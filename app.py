"""The Talent Matrix — FastAPI application."""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import database as db
from core.chat import answer_question, append_turn
from core.interview import (
    build_interview_pack,
    migrate_legacy_pack,
    normalize_stored_pack,
    pack_to_export_text,
)
from core.llm_client import LLMClient, LLMError, get_client
from core.matcher import (
    compute_match_fingerprint,
    enrich_better_fit,
    extract_jd_requirements,
    score_all_resumes,
    sort_results_by_verdict,
)
from core.pdf_loader import PDFLoadError, extract_text
from core import rag
from core.text_normalize import normalize_jd_text, normalize_resume_text

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH, override=True)

STATIC_DIR = Path(__file__).parent / "static"
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    try:
        for resume in db.list_resumes():
            if resume.get("text", "").strip():
                rag.index_resume(resume["id"], resume["text"])
    except Exception:
        pass
    yield


app = FastAPI(title="The Talent Matrix", version=APP_VERSION, lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong. Please try again."},
    )


class JdTextUpload(BaseModel):
    title: str
    text: str


def _better_fit_enabled() -> bool:
    return os.getenv("BETTER_FIT_ENABLED", "false").lower() in ("1", "true", "yes")


def _match_payload(
    jd_id: str,
    results: list,
    requirements: dict | None,
    resumes: list,
    shortlist: set[str] | None = None,
    *,
    cached: bool = False,
) -> dict:
    llm_count = sum(1 for r in results if r.get("llm_scored"))
    return {
        "results": results,
        "jd_id": jd_id,
        "requirements": requirements,
        "cached": cached,
        "scoring_weights": {
            "must_haves": "50%",
            "nice_to_haves": "15%",
            "criteria": "35%",
        },
        "rag_meta": {
            "total_resumes": len(resumes),
            "llm_scored_count": llm_count,
            "semantic_shortlist_count": len(shortlist) if shortlist is not None else llm_count,
            "from_cache": cached,
        },
    }


class MatchRequest(BaseModel):
    jd_id: str
    force: bool = False


class ChatRequest(BaseModel):
    candidate_id: str
    message: str


class InterviewRequest(BaseModel):
    candidate_id: str
    jd_id: str | None = None


class InterviewUpdateRequest(BaseModel):
    interviewer_notes: str | None = None
    final_recommendation: str | None = None
    sections: dict[str, list[dict]] | None = None


def _resume_list_item(r: dict) -> dict:
    return {
        "id": r["id"],
        "name": r["filename"],
        "size_kb": r.get("size_kb", 0),
        "ocr_used": r.get("ocr_used", False),
        "uploaded_at": r.get("uploaded_at"),
    }


def _jd_list_item(j: dict) -> dict:
    return {"id": j["id"], "title": j["title"]}


def _pdf_filename(filename: str | None, default: str) -> str:
    return (filename or default).strip()


def _read_pdf_bytes(content: bytes, filename: str) -> tuple[str, bool] | None:
    try:
        return extract_text(content, filename)
    except PDFLoadError:
        return None


@app.post("/api/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    filename = _pdf_filename(file.filename, "resume.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        return {"warning": f"Could not read {filename}. Skipping."}

    extracted = _read_pdf_bytes(content, filename)
    if extracted is None:
        return {"warning": f"Could not read {filename}. Skipping."}
    text, ocr_used = extracted
    clean_text, norm_warnings = normalize_resume_text(text)
    if not clean_text.strip():
        clean_text = text
    resume_id = f"res-{uuid.uuid4().hex[:8]}"
    size_kb = max(1, len(content) // 1024)
    db.create_resume(resume_id, filename, clean_text, content, size_kb, ocr_used)
    db.clear_all_match_results()
    try:
        rag.index_resume(resume_id, clean_text)
    except Exception:
        pass
    result = {"id": resume_id, "name": filename, "size": size_kb, "ocr_used": ocr_used}
    for w in norm_warnings:
        result.setdefault("warnings", []).append(w)
    if len(text.strip()) < 100 and not ocr_used:
        result["warning"] = (
            f"Limited text extracted from {filename}. "
            "Add MISTRAL_API_KEY for scanned PDF OCR."
        )
    elif ocr_used:
        result["info"] = f"Scanned PDF processed with Mistral OCR: {filename}"
    return result


@app.post("/api/upload/jd")
async def upload_jd(file: UploadFile = File(...)):
    filename = _pdf_filename(file.filename, "job_description.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        return {"warning": f"Could not read {filename}. Skipping."}

    extracted = _read_pdf_bytes(content, filename)
    if extracted is None:
        return {"warning": f"Could not read {filename}. Skipping."}
    text, ocr_used = extracted
    clean_text = normalize_jd_text(text) or text
    jd_id = f"jd-{uuid.uuid4().hex[:8]}"
    title = filename.replace(".pdf", "", 1).replace("_", " ").replace("-", " ").title()
    db.create_jd(jd_id, title, clean_text, content, ocr_used)
    result = {"id": jd_id, "title": title, "ocr_used": ocr_used}
    if len(text.strip()) < 100 and not ocr_used:
        result["warning"] = (
            f"Limited text extracted from {filename}. "
            "Add MISTRAL_API_KEY for scanned PDF OCR."
        )
    elif ocr_used:
        result["info"] = f"Scanned PDF processed with Mistral OCR: {filename}"
    return result


def _key_status() -> dict:
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    keys = {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "openai": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "gemini": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "mistral": bool(os.getenv("MISTRAL_API_KEY", "").strip()),
    }
    model_env = {
        "anthropic": "ANTHROPIC_MODEL",
        "openai": "OPENAI_MODEL",
        "gemini": "GEMINI_MODEL",
    }
    default_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
    }
    return {
        "provider": provider,
        "keys": keys,
        "active_key_set": keys.get(provider, False),
        "model": os.getenv(model_env.get(provider, ""), default_models.get(provider, "")),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>The Talent Matrix</h1><p>static/index.html not found</p>")


@app.post("/api/upload/jd-text")
async def upload_jd_text(body: JdTextUpload):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")
    jd_id = f"jd-{uuid.uuid4().hex[:8]}"
    title = body.title.strip() or "Untitled JD"
    clean_text = normalize_jd_text(body.text) or body.text
    db.create_jd(jd_id, title, clean_text)
    return {"id": jd_id, "title": title}


@app.get("/api/files")
async def list_files():
    return {
        "resumes": [_resume_list_item(r) for r in db.list_resumes()],
        "jds": [_jd_list_item(j) for j in db.list_jds()],
    }


@app.get("/api/resume/{resume_id}")
async def get_resume(resume_id: str):
    r = db.get_resume(resume_id)
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found.")
    return {
        "id": r["id"],
        "name": r["filename"],
        "text": r.get("text", ""),
        "size_kb": r.get("size_kb", 0),
        "ocr_used": r.get("ocr_used", False),
    }


@app.delete("/api/resume/{resume_id}")
async def delete_resume(resume_id: str):
    if not db.delete_resume(resume_id):
        raise HTTPException(status_code=404, detail="Resume not found.")
    rag.delete_resume(resume_id)
    return {"ok": True}


@app.delete("/api/jd/{jd_id}")
async def delete_jd(jd_id: str):
    if not db.delete_jd(jd_id):
        raise HTTPException(status_code=404, detail="Job description not found.")
    return {"ok": True}


@app.get("/api/match/{jd_id}")
async def get_match_results(jd_id: str):
    jd = db.get_jd(jd_id)
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found.")
    results, requirements, _fp = db.get_match_results(jd_id)
    if not requirements and jd.get("requirements"):
        requirements = jd["requirements"]
    llm_count = sum(1 for r in results if r.get("llm_scored"))
    return {
        "results": results,
        "jd_id": jd_id,
        "requirements": requirements,
        "has_results": bool(results),
        "cached": bool(results),
        "scoring_weights": {
            "must_haves": "50%",
            "nice_to_haves": "15%",
            "criteria": "35%",
        },
        "rag_meta": {
            "total_resumes": len(db.list_resumes()),
            "llm_scored_count": llm_count,
            "from_cache": True,
        },
    }


@app.post("/api/match")
async def run_match(body: MatchRequest):
    jd = db.get_jd(body.jd_id)
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found.")
    resumes = db.list_resumes()
    if not resumes:
        raise HTTPException(status_code=400, detail="Upload at least one resume first.")

    jd_text = jd.get("text", "")
    jd_title = jd.get("title", "")
    requirements = jd.get("requirements")

    if not body.force:
        cached_results, cached_req, stored_fp = db.get_match_results(body.jd_id)
        if cached_results and stored_fp:
            fp = compute_match_fingerprint(
                body.jd_id, jd_text, resumes, cached_req or requirements
            )
            if fp == stored_fp:
                req = cached_req or requirements
                return _match_payload(
                    body.jd_id,
                    cached_results,
                    req,
                    resumes,
                    cached=True,
                )

    shortlist: set[str] = {r["id"] for r in resumes}

    try:
        if not requirements or not requirements.get("must_haves"):
            requirements = extract_jd_requirements(jd_text)
            db.update_jd_requirements(body.jd_id, requirements)

        resume_ids = [r["id"] for r in resumes]
        query = rag.build_match_query(jd_text, requirements, jd_title)
        semantic_scores = rag.semantic_scores_for_resumes(query, resume_ids)
        shortlist = rag.pick_shortlist(semantic_scores, len(resumes))
        to_score = [r for r in resumes if r["id"] in shortlist]

        llm_results, requirements = score_all_resumes(
            to_score,
            jd_text,
            requirements=requirements,
            max_workers=min(len(to_score), 8),
        )
        db.update_jd_requirements(body.jd_id, requirements)
        results = rag.merge_llm_and_semantic(
            llm_results, resumes, semantic_scores, shortlist
        )
        all_jds = db.get_all_jds_map()
        if _better_fit_enabled() and len(all_jds) > 1:
            llm_only = [r for r in results if r.get("llm_scored")]
            enriched = enrich_better_fit(
                llm_only,
                resumes,
                body.jd_id,
                all_jds,
            )
            enriched_map = {r["candidate_id"]: r for r in enriched}
            for row in results:
                if row.get("llm_scored") and row["candidate_id"] in enriched_map:
                    alt = enriched_map[row["candidate_id"]].get("better_fit_jd")
                    if alt:
                        row["better_fit_jd"] = alt
            results = sort_results_by_verdict(results)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="AI matching failed. Please try again in a moment.",
        ) from exc

    fingerprint = compute_match_fingerprint(
        body.jd_id, jd_text, resumes, requirements
    )
    db.save_match_results(body.jd_id, results, requirements, fingerprint=fingerprint)
    return _match_payload(
        body.jd_id, results, requirements, resumes, shortlist, cached=False
    )


@app.get("/api/chat/{candidate_id}")
async def get_chat_history(candidate_id: str):
    if not db.get_resume(candidate_id):
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return {"history": db.get_chat_history(candidate_id)}


@app.post("/api/chat")
async def chat(body: ChatRequest):
    resume = db.get_resume(body.candidate_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history = db.get_chat_history(body.candidate_id)

    try:
        reply = answer_question(resume.get("text", ""), message, history)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Chat failed. Please try again in a moment.",
        ) from exc

    history = append_turn(history, "user", message)
    history = append_turn(history, "assistant", reply)
    db.save_chat_history(body.candidate_id, history)

    return {"reply": reply, "history": history}


def _match_row_for_candidate(candidate_id: str, jd_id: str | None) -> dict | None:
    if not jd_id:
        return None
    results, _, _ = db.get_match_results(jd_id)
    for row in results:
        if row.get("candidate_id") == candidate_id:
            return row
    return None


def _shortlisted_rows(jd_id: str) -> list[dict]:
    results, _, _ = db.get_match_results(jd_id)
    out = []
    for row in results:
        overall = int(row.get("overall_score") or 0)
        verdict = row.get("verdict") or ""
        if verdict == "TOP_MATCH" or overall >= 70:
            out.append(row)
    return out


def _weaknesses_for_candidate(candidate_id: str, jd_id: str | None) -> list[str]:
    row = _match_row_for_candidate(candidate_id, jd_id)
    return (row or {}).get("weaknesses") or []


def _resolve_jd_id(jd_id: str | None) -> str | None:
    if jd_id and db.get_jd(jd_id):
        return jd_id
    jds = db.list_jds()
    if len(jds) == 1:
        return jds[0]["id"]
    return jd_id if jd_id and db.get_jd(jd_id) else None


@app.get("/api/interview/shortlisted")
async def interview_shortlisted(jd_id: str = Query(..., alias="jd_id")):
    jd = db.get_jd(jd_id)
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found.")
    results, requirements, _ = db.get_match_results(jd_id)
    if not results:
        raise HTTPException(
            status_code=400,
            detail="Run AI Matching for this role first to shortlist candidates.",
        )
    shortlisted = _shortlisted_rows(jd_id)
    return {
        "jd_id": jd_id,
        "jd_title": jd.get("title", ""),
        "candidates": shortlisted,
        "count": len(shortlisted),
        "requirements": requirements or jd.get("requirements"),
    }


@app.get("/api/interview/{candidate_id}")
async def get_interview_questions(candidate_id: str):
    if not db.get_resume(candidate_id):
        raise HTTPException(status_code=404, detail="Candidate not found.")
    raw = db.get_interview_questions(candidate_id)
    pack = normalize_stored_pack(raw, candidate_id)
    if pack and raw and (
        raw.get("version") != 2
        or not isinstance(raw.get("sections"), dict)
        or set(raw.get("sections", {})) != set(pack.get("sections", {}))
    ):
        db.save_interview_questions(candidate_id, pack)
    return {"pack": pack, "questions": (pack or {}).get("sections")}


@app.put("/api/interview/{candidate_id}")
async def update_interview_pack(candidate_id: str, body: InterviewUpdateRequest):
    if not db.get_resume(candidate_id):
        raise HTTPException(status_code=404, detail="Candidate not found.")
    raw = db.get_interview_questions(candidate_id)
    pack = normalize_stored_pack(raw, candidate_id)
    if not pack:
        raise HTTPException(
            status_code=404,
            detail="Generate an interview pack first.",
        )
    if body.interviewer_notes is not None:
        pack["interviewer_notes"] = body.interviewer_notes[:8000]
    if body.final_recommendation is not None:
        allowed = {"", "hire", "hold", "no_hire", "strong_hire"}
        rec = body.final_recommendation.strip().lower()
        if rec not in allowed:
            raise HTTPException(
                status_code=400,
                detail="final_recommendation must be hire, hold, no_hire, strong_hire, or empty.",
            )
        pack["final_recommendation"] = rec
    if body.sections:
        for section, items in body.sections.items():
            if section not in pack.get("sections", {}):
                continue
            if not isinstance(items, list):
                continue
            existing = pack["sections"][section]
            for i, patch in enumerate(items):
                if i >= len(existing) or not isinstance(patch, dict):
                    continue
                if "asked" in patch:
                    existing[i]["asked"] = bool(patch["asked"])
                if patch.get("rating") in (1, 2, 3, 4, 5, None):
                    existing[i]["rating"] = patch.get("rating")
                if "interviewer_note" in patch:
                    existing[i]["interviewer_note"] = str(patch["interviewer_note"])[:2000]
    db.save_interview_questions(candidate_id, pack)
    return {"pack": pack, "ok": True}


@app.post("/api/interview")
async def generate_interview(body: InterviewRequest):
    resume = db.get_resume(body.candidate_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    jd_id = _resolve_jd_id(body.jd_id)
    if not jd_id:
        raise HTTPException(
            status_code=400,
            detail="Select a job description on the Matching page first.",
        )

    match_row = _match_row_for_candidate(body.candidate_id, jd_id)
    if not match_row:
        raise HTTPException(
            status_code=400,
            detail="Candidate has no match results for this role. Run AI Matching first.",
        )
    overall = int(match_row.get("overall_score") or 0)
    if match_row.get("verdict") != "TOP_MATCH" and overall < 70:
        raise HTTPException(
            status_code=400,
            detail=(
                "Interview packs are for shortlisted candidates only (Overall ≥ 70). "
                "This candidate is not shortlisted for the selected role."
            ),
        )

    jd = db.get_jd(jd_id)
    jd_text = jd.get("text", "") if jd else ""
    requirements = jd.get("requirements") if jd else None
    if not requirements:
        _, requirements, _ = db.get_match_results(jd_id)

    try:
        pack = build_interview_pack(
            resume.get("text", ""),
            jd_text,
            candidate_id=body.candidate_id,
            candidate_name=match_row.get("candidate_name")
            or resume["filename"].replace(".pdf", "").replace("_", " ").title(),
            jd_id=jd_id,
            jd_title=jd.get("title", "") if jd else "",
            weaknesses=match_row.get("weaknesses"),
            must_haves=(requirements or {}).get("must_haves"),
            nice_to_haves=(requirements or {}).get("nice_to_haves"),
            must_matched=match_row.get("must_have_matched"),
            must_missing=match_row.get("must_have_missing")
            or match_row.get("missing_must_haves"),
            evidence=match_row.get("evidence"),
            match_summary=match_row.get("summary") or "",
            overall_score=overall,
        )
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Interview generation failed. Please try again in a moment.",
        ) from exc

    db.save_interview_questions(body.candidate_id, pack)
    return {
        "pack": pack,
        "questions": pack["sections"],
        "candidate_id": body.candidate_id,
        "jd_id": jd_id,
    }


@app.get("/api/interview/export")
async def export_interview(candidate_id: str = Query(...)):
    resume = db.get_resume(candidate_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    raw = db.get_interview_questions(candidate_id)
    pack = normalize_stored_pack(raw, candidate_id)
    if not pack:
        raise HTTPException(status_code=404, detail="Generate interview questions first.")
    name = (
        pack.get("candidate_name")
        or resume["filename"].replace(".pdf", "").replace("_", " ").title()
    )
    content = pack_to_export_text(pack, name)
    filename = f"interview_{candidate_id}.txt"
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/settings")
async def get_settings():
    return _key_status()


@app.post("/api/llm/test")
async def test_llm():
    """Quick connectivity check for the configured LLM provider."""
    try:
        client = get_client()
        reply = client.complete(
            "Reply with exactly one word: OK",
            system="You are a concise assistant.",
            json_mode=False,
        )
        return {"ok": True, "provider": client.provider, "reply": reply.strip()[:200]}
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="AI service is temporarily unavailable. Check your API key and try again.",
        ) from exc


@app.delete("/api/clear")
async def clear_all():
    db.clear_all_data()
    rag.clear_index()
    LLMClient.clear_cache()
    return {"ok": True, "message": "All data cleared."}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

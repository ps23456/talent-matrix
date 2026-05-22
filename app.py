"""TalentMatch AI — FastAPI application."""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.chat import answer_question, append_turn
from core.interview import generate_questions
from core.llm_client import LLMClient, LLMError, get_client
from core.matcher import score_all_resumes
from core.pdf_loader import PDFLoadError, extract_text

load_dotenv(Path(__file__).parent / ".env")

STATIC_DIR = Path(__file__).parent / "static"
APP_VERSION = "1.0.0"

store: dict = {
    "resumes": {},
    "jds": {},
    "match_results": {},
    "chat_history": {},
    "interview_qs": {},
}

app = FastAPI(title="TalentMatch AI", version=APP_VERSION)


class JdTextUpload(BaseModel):
    title: str
    text: str


class MatchRequest(BaseModel):
    jd_id: str


class ChatRequest(BaseModel):
    candidate_id: str
    message: str


class InterviewRequest(BaseModel):
    candidate_id: str
    jd_id: str | None = None


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
    resume_id = f"res-{uuid.uuid4().hex[:8]}"
    size_kb = max(1, len(content) // 1024)
    store["resumes"][resume_id] = {
        "id": resume_id,
        "filename": filename,
        "text": text,
        "bytes": content,
        "size_kb": size_kb,
        "ocr_used": ocr_used,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    result = {"id": resume_id, "name": filename, "size": size_kb, "ocr_used": ocr_used}
    if len(text.strip()) < 100:
        result["warning"] = (
            f"Limited text extracted from {filename}. "
            "Scanned PDFs may need OCR (enable MISTRAL_API_KEY in Step 10)."
        )
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
    jd_id = f"jd-{uuid.uuid4().hex[:8]}"
    title = filename.replace(".pdf", "", 1).replace("_", " ").replace("-", " ").title()
    store["jds"][jd_id] = {
        "id": jd_id,
        "title": title,
        "text": text,
        "ocr_used": ocr_used,
    }
    result = {"id": jd_id, "title": title, "ocr_used": ocr_used}
    if len(text.strip()) < 100:
        result["warning"] = (
            f"Limited text extracted from {filename}. "
            "Scanned PDFs may need OCR (enable MISTRAL_API_KEY in Step 10)."
        )
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
    return HTMLResponse("<h1>TalentMatch AI</h1><p>static/index.html not found</p>")


@app.post("/api/upload/jd-text")
async def upload_jd_text(body: JdTextUpload):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")
    jd_id = f"jd-{uuid.uuid4().hex[:8]}"
    store["jds"][jd_id] = {"id": jd_id, "title": body.title.strip() or "Untitled JD", "text": body.text}
    return {"id": jd_id, "title": store["jds"][jd_id]["title"]}


@app.get("/api/files")
async def list_files():
    return {
        "resumes": [_resume_list_item(r) for r in store["resumes"].values()],
        "jds": [_jd_list_item(j) for j in store["jds"].values()],
    }


@app.get("/api/resume/{resume_id}")
async def get_resume(resume_id: str):
    if resume_id not in store["resumes"]:
        raise HTTPException(status_code=404, detail="Resume not found.")
    r = store["resumes"][resume_id]
    return {
        "id": r["id"],
        "name": r["filename"],
        "text": r.get("text", ""),
        "size_kb": r.get("size_kb", 0),
        "ocr_used": r.get("ocr_used", False),
    }


@app.delete("/api/resume/{resume_id}")
async def delete_resume(resume_id: str):
    if resume_id not in store["resumes"]:
        raise HTTPException(status_code=404, detail="Resume not found.")
    del store["resumes"][resume_id]
    store["chat_history"].pop(resume_id, None)
    store["interview_qs"].pop(resume_id, None)
    return {"ok": True}


@app.delete("/api/jd/{jd_id}")
async def delete_jd(jd_id: str):
    if jd_id not in store["jds"]:
        raise HTTPException(status_code=404, detail="Job description not found.")
    del store["jds"][jd_id]
    store["match_results"].pop(jd_id, None)
    return {"ok": True}


@app.post("/api/match")
async def run_match(body: MatchRequest):
    if body.jd_id not in store["jds"]:
        raise HTTPException(status_code=404, detail="Job description not found.")
    if not store["resumes"]:
        raise HTTPException(status_code=400, detail="Upload at least one resume first.")

    jd_text = store["jds"][body.jd_id].get("text", "")
    resumes = list(store["resumes"].values())

    try:
        results = score_all_resumes(resumes, jd_text)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="AI matching failed. Please try again in a moment.",
        ) from exc

    store["match_results"][body.jd_id] = results
    return {"results": results, "jd_id": body.jd_id}


@app.get("/api/chat/{candidate_id}")
async def get_chat_history(candidate_id: str):
    if candidate_id not in store["resumes"]:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return {"history": store["chat_history"].get(candidate_id, [])}


@app.post("/api/chat")
async def chat(body: ChatRequest):
    if body.candidate_id not in store["resumes"]:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    resume = store["resumes"][body.candidate_id]
    history = store["chat_history"].get(body.candidate_id, [])

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
    store["chat_history"][body.candidate_id] = history

    return {"reply": reply, "history": history}


def _weaknesses_for_candidate(candidate_id: str, jd_id: str | None) -> list[str]:
    if not jd_id:
        return []
    for row in store["match_results"].get(jd_id, []):
        if row.get("candidate_id") == candidate_id:
            return row.get("weaknesses") or []
    return []


def _resolve_jd_id(jd_id: str | None) -> str | None:
    if jd_id and jd_id in store["jds"]:
        return jd_id
    if len(store["jds"]) == 1:
        return next(iter(store["jds"]))
    return jd_id if jd_id in store["jds"] else None


@app.get("/api/interview/{candidate_id}")
async def get_interview_questions(candidate_id: str):
    if candidate_id not in store["resumes"]:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return {"questions": store["interview_qs"].get(candidate_id)}


@app.post("/api/interview")
async def generate_interview(body: InterviewRequest):
    if body.candidate_id not in store["resumes"]:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    jd_id = _resolve_jd_id(body.jd_id)
    if not jd_id:
        raise HTTPException(
            status_code=400,
            detail="Select a job description on the Matching page first.",
        )

    resume = store["resumes"][body.candidate_id]
    jd_text = store["jds"][jd_id].get("text", "")
    weaknesses = _weaknesses_for_candidate(body.candidate_id, jd_id)

    try:
        questions = generate_questions(
            resume.get("text", ""),
            jd_text,
            weaknesses,
        )
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Interview generation failed. Please try again.",
        ) from exc

    store["interview_qs"][body.candidate_id] = questions
    return {"questions": questions, "candidate_id": body.candidate_id, "jd_id": jd_id}


@app.get("/api/interview/export")
async def export_interview(candidate_id: str = Query(...)):
    if candidate_id not in store["resumes"]:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    questions = store["interview_qs"].get(candidate_id)
    if not questions:
        raise HTTPException(status_code=404, detail="Generate interview questions first.")
    name = store["resumes"][candidate_id]["filename"].replace(".pdf", "").replace("_", " ").title()
    lines = [f"Interview Kit — {name}", "=" * 40, ""]
    section_labels = {
        "technical": "Technical Validation",
        "problem_solving": "Problem Solving",
        "behavioral": "Behavioral",
        "gap_probing": "Gap Probing",
        "role_specific": "Role-Specific",
    }
    for key, label in section_labels.items():
        lines.append(f"## {label}")
        for i, item in enumerate(questions.get(key, []), 1):
            lines.append(f"{i}. {item['q']}")
            lines.append(f"   Why: {item['why_ask']}")
        lines.append("")
    content = "\n".join(lines)
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
    for key in store:
        store[key].clear()
    LLMClient.clear_cache()
    return {"ok": True, "message": "All data cleared."}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

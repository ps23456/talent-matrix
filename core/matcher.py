"""Resume ↔ JD scoring via LLM."""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.llm_client import LLMError, get_client

MATCH_JD_CHAR_LIMIT = int(os.getenv("MATCH_JD_CHAR_LIMIT", "8000"))
MATCH_RESUME_CHAR_LIMIT = int(os.getenv("MATCH_RESUME_CHAR_LIMIT", "6000"))
MATCH_MAX_WORKERS = int(os.getenv("MATCH_MAX_WORKERS", "8"))

SCORING_SYSTEM = """You are an expert technical recruiter with 15 years of experience. You evaluate resumes using semantic analysis and contextual evidence — not keyword matching. You produce structured JSON scoring."""

JD_EXTRACT_SYSTEM = (
    "You extract structured hiring requirements from job descriptions. "
    "Return JSON only."
)

JD_EXTRACT_TEMPLATE = """From this job description, extract Must-Haves (deal-breakers, high bar) and Nice-to-Haves (bonus, low bar).

=== JOB DESCRIPTION ===
{jd_text}

Return ONLY valid JSON:
{{
  "must_haves": ["item1", "item2"],
  "nice_to_haves": ["item1", "item2"]
}}

Rules:
- must_haves: 4–8 concrete requirements (skills, years of experience, education, location, certifications explicitly required).
- nice_to_haves: 3–6 optional bonuses (tools, extra certs, preferred but not required skills).
- Use short phrases (e.g. "Python", "3+ years backend", "SQL").
- If the JD does not list nice-to-haves, infer 3–4 reasonable optional items for the role."""

SCORING_USER_TEMPLATE = """Score this candidate's resume against the job description and requirement lists below.

=== JOB DESCRIPTION ===
{jd_text}

=== MUST-HAVES (High weight — missing any hurts rank significantly) ===
{must_haves}

=== NICE-TO-HAVES (Low weight — bonuses only) ===
{nice_to_haves}

=== CANDIDATE RESUME ===
{resume_text}

Return ONLY valid JSON, no preamble, no markdown fences:
{{
  "candidate_name": "extracted full name",
  "must_have_score": <0-100, percent of must-haves clearly met>,
  "nice_to_have_score": <0-100, percent of nice-to-haves met>,
  "must_have_matched": ["items clearly present in resume"],
  "must_have_missing": ["must-haves NOT evidenced in resume"],
  "nice_to_have_matched": ["nice-to-haves present"],
  "nice_to_have_missing": ["nice-to-haves not present"],
  "scores": {{
    "mandatory_skills":      {{"score": 0-100, "weight": 25, "reason": "1 line"}},
    "relevant_experience":   {{"score": 0-100, "weight": 20, "reason": "1 line"}},
    "domain_experience":     {{"score": 0-100, "weight": 10, "reason": "1 line"}},
    "education":             {{"score": 0-100, "weight": 5,  "reason": "1 line"}},
    "certification_fit":     {{"score": 0-100, "weight": 10, "reason": "1 line"}},
    "recent_usage":          {{"score": 0-100, "weight": 10, "reason": "1 line"}},
    "seniority_fit":         {{"score": 0-100, "weight": 15, "reason": "1 line"}},
    "location_availability": {{"score": 0-100, "weight": 5,  "reason": "1 line"}}
  }},
  "criteria_score": <weighted average of 8 criteria, integer 0-100>,
  "evidence": [
    {{"requirement": "must-have or nice-to-have label", "matched": true, "quote": "short exact phrase from resume", "assessment": "1 line on behavioral evidence"}}
  ],
  "authenticity_score": <0-100, concrete metrics and real experience = high; generic AI fluff = low>,
  "authenticity_flags": ["flag if vague bullets, no numbers, keyword stuffing, inconsistent timeline"],
  "summary": "2-sentence executive summary",
  "strengths": ["...", "...", "..."],
  "weaknesses": ["...", "...", "..."]
}}

Rules:
- SEMANTIC ANALYSIS: Do NOT match keywords alone. Require evidence of behavior (e.g. leadership = team size, budget, outcomes; Python = projects built, scale, impact).
- criteria_score = sum(score_i * weight_i / 100) for all 8 criteria.
- If a must-have is missing, must_have_score cannot exceed 55 and mandatory_skills cannot exceed 50.
- must_have_matched only if evidence supports it; must_have_missing lists gaps.
- evidence: 4-8 items covering key must-haves; quote real resume phrases.
- authenticity_score: penalize generic templates, buzzword lists without metrics, no dates/numbers, reads like AI dump.
- Be strict but fair."""

# Final rank score: must-haves dominate, nice-to-haves bonus, criteria still matter
MUST_WEIGHT = 0.50
NICE_WEIGHT = 0.15
CRITERIA_WEIGHT = 0.35

JSON_RETRY_SUFFIX = (
    "\n\nYour last response was not valid JSON. Return ONLY the JSON object."
)

CRITERIA_WEIGHTS = {
    "mandatory_skills": 25,
    "relevant_experience": 20,
    "domain_experience": 10,
    "education": 5,
    "certification_fit": 10,
    "recent_usage": 10,
    "seniority_fit": 15,
    "location_availability": 5,
}


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response.")
    return json.loads(text[start : end + 1])


def _call_llm_json(prompt: str, retry: bool = False) -> dict[str, Any]:
    client = get_client()
    full_prompt = prompt + (JSON_RETRY_SUFFIX if retry else "")
    raw = client.complete(full_prompt, system=SCORING_SYSTEM, json_mode=True)
    return _extract_json(raw)


def _weighted_overall(scores: dict[str, Any]) -> int:
    total = 0.0
    for key, weight in CRITERIA_WEIGHTS.items():
        entry = scores.get(key) or {}
        score = max(0, min(100, int(entry.get("score", 0))))
        total += score * weight / 100
    return round(total)


def _clamp_score(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _evidence_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "requirement": str(item.get("requirement", ""))[:120],
                "matched": bool(item.get("matched")),
                "quote": str(item.get("quote", ""))[:300],
                "assessment": str(item.get("assessment", ""))[:200],
            }
        )
    return out


def _role_fit_score(must: int, nice: int, criteria: int) -> int:
    return round(
        must * MUST_WEIGHT + nice * NICE_WEIGHT + criteria * CRITERIA_WEIGHT
    )


def extract_jd_requirements(jd_text: str) -> dict[str, list[str]]:
    """Parse must-have and nice-to-have lists from a job description (one LLM call)."""
    if not jd_text.strip():
        raise LLMError("Job description text is empty.")

    prompt = JD_EXTRACT_TEMPLATE.format(jd_text=jd_text[:12000])
    client = get_client()
    try:
        raw = client.complete(prompt, system=JD_EXTRACT_SYSTEM, json_mode=True)
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError, KeyError):
        raw = client.complete(
            prompt + JSON_RETRY_SUFFIX,
            system=JD_EXTRACT_SYSTEM,
            json_mode=True,
        )
        data = _extract_json(raw)

    must = _str_list(data.get("must_haves"))
    nice = _str_list(data.get("nice_to_haves"))
    if not must:
        must = ["Core skills and experience required by the role"]
    if not nice:
        nice = ["Additional preferred skills for the role"]
    return {"must_haves": must[:10], "nice_to_haves": nice[:8]}


VERDICT_SORT_ORDER = {
    "TOP_MATCH": 0,   # Shortlisted — overall score >= 70
    "HOLD": 1,        # In review — overall score 50–69
    "REJECT": 2,      # Rejected — overall score < 50
    "SEMANTIC_ONLY": 3,
}


def _verdict_from_score(overall: int) -> str:
    if overall >= 70:
        return "TOP_MATCH"
    if overall >= 50:
        return "HOLD"
    return "REJECT"


def compute_match_fingerprint(
    jd_id: str,
    jd_text: str,
    resumes: list[dict[str, Any]],
    requirements: dict[str, list[str]] | None = None,
) -> str:
    """Stable hash for cache: same JD + resumes + requirement lists → same fingerprint."""
    parts = [jd_id, hashlib.sha256(jd_text.encode("utf-8")).hexdigest()[:20]]
    if requirements:
        req_blob = json.dumps(
            {
                "must_haves": requirements.get("must_haves") or [],
                "nice_to_haves": requirements.get("nice_to_haves") or [],
            },
            sort_keys=True,
        )
        parts.append(hashlib.sha256(req_blob.encode("utf-8")).hexdigest()[:20])
    for row in sorted(resumes, key=lambda r: r.get("id", "")):
        text = row.get("text") or ""
        parts.append(
            f"{row.get('id', '')}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:20]}"
        )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def sort_results_by_verdict(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order: Shortlisted → In review → Rejected → Semantic only; then by score."""
    sorted_rows = sorted(
        results,
        key=lambda x: (
            VERDICT_SORT_ORDER.get(x.get("verdict"), 99),
            0 if x.get("llm_scored") else 1,
            -x.get("overall_score", 0),
            -x.get("semantic_score", 0) if x.get("semantic_score") is not None else 0,
            -x.get("must_have_score", 0),
        ),
    )
    for i, row in enumerate(sorted_rows, start=1):
        row["rank"] = i
    return sorted_rows


def _normalize_result(data: dict[str, Any], candidate_id: str, filename: str) -> dict[str, Any]:
    scores = data.get("scores") or {}
    for key, weight in CRITERIA_WEIGHTS.items():
        entry = scores.setdefault(key, {})
        entry["weight"] = weight
        entry["score"] = _clamp_score(entry.get("score", 0))
        entry["reason"] = str(entry.get("reason", ""))[:200]

    criteria = _clamp_score(data.get("criteria_score"))
    if criteria == 0:
        criteria = _weighted_overall(scores)

    must_score = _clamp_score(data.get("must_have_score"))
    nice_score = _clamp_score(data.get("nice_to_have_score"))
    overall = _role_fit_score(must_score, nice_score, criteria)
    verdict = _verdict_from_score(overall)

    name = (data.get("candidate_name") or "").strip()
    if not name:
        name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

    must_missing = _str_list(data.get("must_have_missing"))
    return {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "role_title": data.get("role_title") or "—",
        "years_experience": int(data.get("years_experience") or 0),
        "location": data.get("location") or "—",
        "overall_score": overall,
        "must_have_score": must_score,
        "nice_to_have_score": nice_score,
        "criteria_score": criteria,
        "verdict": verdict,
        "rank": 0,
        "summary": data.get("summary") or "",
        "scores": scores,
        "strengths": _str_list(data.get("strengths")),
        "weaknesses": _str_list(data.get("weaknesses")),
        "must_have_matched": _str_list(data.get("must_have_matched")),
        "must_have_missing": must_missing,
        "nice_to_have_matched": _str_list(data.get("nice_to_have_matched")),
        "nice_to_have_missing": _str_list(data.get("nice_to_have_missing")),
        "missing_must_haves": must_missing,
        "evidence": _evidence_list(data.get("evidence")),
        "authenticity_score": _clamp_score(data.get("authenticity_score")),
        "authenticity_flags": _str_list(data.get("authenticity_flags")),
        "semantic_score": None,
        "semantic_score_pct": None,
        "llm_scored": True,
        "better_fit_jd": None,
    }


def _assign_ranks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in results:
        if row.get("llm_scored", True) and row.get("verdict") not in VERDICT_SORT_ORDER:
            row["verdict"] = _verdict_from_score(int(row.get("overall_score", 0)))
    return sort_results_by_verdict(results)


def score_single_resume(
    candidate_id: str,
    filename: str,
    resume_text: str,
    jd_text: str,
    requirements: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if not resume_text.strip():
        raise LLMError(f"Resume '{filename}' has no extractable text.")

    req = requirements or {"must_haves": [], "nice_to_haves": []}
    must_lines = "\n".join(f"- {m}" for m in req.get("must_haves", [])) or "- (see JD)"
    nice_lines = "\n".join(f"- {n}" for n in req.get("nice_to_haves", [])) or "- (see JD)"

    prompt = SCORING_USER_TEMPLATE.format(
        jd_text=jd_text[:MATCH_JD_CHAR_LIMIT],
        resume_text=resume_text[:MATCH_RESUME_CHAR_LIMIT],
        must_haves=must_lines,
        nice_to_haves=nice_lines,
    )
    try:
        parsed = _call_llm_json(prompt, retry=False)
    except (json.JSONDecodeError, ValueError, KeyError):
        parsed = _call_llm_json(prompt, retry=True)

    return _normalize_result(parsed, candidate_id, filename)


def score_all_resumes(
    resumes: list[dict[str, Any]],
    jd_text: str,
    requirements: dict[str, list[str]] | None = None,
    max_workers: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if not jd_text.strip():
        raise LLMError("Job description text is empty.")

    req = requirements if requirements else extract_jd_requirements(jd_text)

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    workers = max_workers if max_workers is not None else min(len(resumes), MATCH_MAX_WORKERS)
    workers = max(1, workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                score_single_resume,
                r["id"],
                r.get("filename", "resume.pdf"),
                r.get("text", ""),
                jd_text,
                req,
            ): r
            for r in resumes
        }
        for future in as_completed(futures):
            resume = futures[future]
            try:
                results.append(future.result())
            except LLMError as exc:
                errors.append(f"{resume.get('filename', 'resume')}: {exc}")
            except Exception:
                errors.append(
                    f"{resume.get('filename', 'resume')}: scoring failed unexpectedly."
                )

    if not results and errors:
        raise LLMError(errors[0] if len(errors) == 1 else "; ".join(errors[:3]))

    return _assign_ranks(results), req


BETTER_FIT_SYSTEM = (
    "You are an expert recruiter doing a quick fit check. "
    "Answer with JSON only."
)

BETTER_FIT_PROMPT = """Given this resume and this job description, would this candidate likely score >= 75 if matched against this JD?

=== JOB DESCRIPTION ({jd_title}) ===
{jd_text}

=== RESUME ===
{resume_text}

Return ONLY valid JSON, no markdown fences:
{{"fits": true or false, "jd_title": "{jd_title}"}}"""


def _check_better_fit(resume_text: str, jd_text: str, jd_title: str) -> dict[str, Any]:
    prompt = BETTER_FIT_PROMPT.format(
        jd_text=jd_text[:8000],
        resume_text=resume_text[:8000],
        jd_title=jd_title,
    )
    client = get_client()
    try:
        raw = client.complete(prompt, system=BETTER_FIT_SYSTEM, json_mode=True)
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError, KeyError):
        raw = client.complete(
            prompt + JSON_RETRY_SUFFIX,
            system=BETTER_FIT_SYSTEM,
            json_mode=True,
        )
        data = _extract_json(raw)

    fits = data.get("fits")
    if isinstance(fits, str):
        fits = fits.strip().lower() in ("true", "yes", "1")
    return {
        "fits": bool(fits),
        "jd_title": str(data.get("jd_title") or jd_title).strip(),
    }


def enrich_better_fit(
    results: list[dict[str, Any]],
    resumes: list[dict[str, Any]],
    current_jd_id: str,
    all_jds: dict[str, dict[str, Any]],
    max_candidates: int = 5,
    max_other_jds: int = 3,
) -> list[dict[str, Any]]:
    """
    For HOLD/REJECT candidates, check up to 3 other JDs each (max 5 candidates).
    Sets better_fit_jd when another role is a stronger match.
    """
    other_jds = [
        (jd_id, jd)
        for jd_id, jd in all_jds.items()
        if jd_id != current_jd_id and jd.get("text", "").strip()
    ]
    if not other_jds:
        return results

    resume_map = {r["id"]: r for r in resumes}
    hold_reject = [
        r for r in results if r.get("verdict") in ("HOLD", "REJECT")
    ]
    hold_reject.sort(key=lambda x: x["overall_score"], reverse=True)
    targets = hold_reject[:max_candidates]

    def _find_fit(candidate: dict[str, Any]) -> tuple[str, str | None]:
        resume = resume_map.get(candidate["candidate_id"])
        if not resume:
            return candidate["candidate_id"], None
        resume_text = resume.get("text", "")
        for jd_id, jd in other_jds[:max_other_jds]:
            try:
                check = _check_better_fit(
                    resume_text,
                    jd.get("text", ""),
                    jd.get("title", "Other role"),
                )
                if check.get("fits"):
                    return candidate["candidate_id"], check.get("jd_title")
            except Exception:
                continue
        return candidate["candidate_id"], None

    if not targets:
        return results

    with ThreadPoolExecutor(max_workers=3) as pool:
        fits = list(pool.map(_find_fit, targets))

    fit_by_id = {cid: title for cid, title in fits if title}
    for row in results:
        alt = fit_by_id.get(row.get("candidate_id"))
        if alt:
            row["better_fit_jd"] = alt

    return results

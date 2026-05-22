"""Resume ↔ JD scoring via LLM."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.llm_client import LLMError, get_client

SCORING_SYSTEM = """You are an expert technical recruiter with 15 years of experience. You evaluate resumes against job descriptions objectively and produce structured JSON scoring."""

SCORING_USER_TEMPLATE = """Score this candidate's resume against the job description below across 8 weighted criteria. Weights sum to 100.

=== JOB DESCRIPTION ===
{jd_text}

=== CANDIDATE RESUME ===
{resume_text}

Return ONLY valid JSON, no preamble, no markdown fences:
{{
  "candidate_name": "extracted full name",
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
  "overall_score": <weighted average integer 0-100>,
  "verdict": "TOP_MATCH" | "HOLD" | "REJECT",
  "summary": "2-sentence executive summary",
  "strengths": ["...", "...", "..."],
  "weaknesses": ["...", "...", "..."],
  "missing_must_haves": ["skill1", "skill2"]
}}

Rules:
- overall_score = sum(score_i * weight_i / 100) for all 8 criteria
- TOP_MATCH = overall_score >= 75
- HOLD = 55–74
- REJECT < 55
- If a must-have skill is missing, mandatory_skills score cannot exceed 50.
- certification_fit: score based on relevant certifications, courses, or formal training listed. Score 50 if no certifications mentioned but not required by JD.
- Be strict but fair."""

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


def _verdict_from_score(overall: int) -> str:
    if overall >= 75:
        return "TOP_MATCH"
    if overall >= 55:
        return "HOLD"
    return "REJECT"


def _normalize_result(data: dict[str, Any], candidate_id: str, filename: str) -> dict[str, Any]:
    scores = data.get("scores") or {}
    for key, weight in CRITERIA_WEIGHTS.items():
        entry = scores.setdefault(key, {})
        entry["weight"] = weight
        entry["score"] = max(0, min(100, int(entry.get("score", 0))))
        entry["reason"] = str(entry.get("reason", ""))[:200]

    overall = _weighted_overall(scores)
    verdict = _verdict_from_score(overall)

    name = (data.get("candidate_name") or "").strip()
    if not name:
        name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

    return {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "role_title": data.get("role_title") or "—",
        "years_experience": int(data.get("years_experience") or 0),
        "location": data.get("location") or "—",
        "overall_score": overall,
        "verdict": verdict,
        "summary": data.get("summary") or "",
        "scores": scores,
        "strengths": data.get("strengths") or [],
        "weaknesses": data.get("weaknesses") or [],
        "missing_must_haves": data.get("missing_must_haves") or [],
        "better_fit_jd": None,
    }


def score_single_resume(
    candidate_id: str,
    filename: str,
    resume_text: str,
    jd_text: str,
) -> dict[str, Any]:
    if not resume_text.strip():
        raise LLMError(f"Resume '{filename}' has no extractable text.")

    prompt = SCORING_USER_TEMPLATE.format(jd_text=jd_text, resume_text=resume_text)
    try:
        parsed = _call_llm_json(prompt, retry=False)
    except (json.JSONDecodeError, ValueError, KeyError):
        parsed = _call_llm_json(prompt, retry=True)

    return _normalize_result(parsed, candidate_id, filename)


def score_all_resumes(
    resumes: list[dict[str, Any]],
    jd_text: str,
    max_workers: int = 5,
) -> list[dict[str, Any]]:
    if not jd_text.strip():
        raise LLMError("Job description text is empty.")

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                score_single_resume,
                r["id"],
                r.get("filename", "resume.pdf"),
                r.get("text", ""),
                jd_text,
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

    results.sort(key=lambda x: x["overall_score"], reverse=True)
    return results

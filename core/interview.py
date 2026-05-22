"""Tailored interview question generator."""

from __future__ import annotations

import json
from typing import Any

from core.llm_client import LLMError, get_client
from core.matcher import _extract_json

INTERVIEW_SYSTEM = (
    "You are a senior interviewer designing a targeted interview plan."
)

INTERVIEW_USER_TEMPLATE = """Generate exactly 10 interview questions. Return ONLY valid JSON, no markdown fences.

JD: {jd_text}
RESUME: {resume_text}
WEAKNESSES TO PROBE: {weaknesses}

{{
  "technical":      [{{"q": "...", "why_ask": "..."}}, ...],
  "problem_solving":[{{"q": "...", "why_ask": "..."}}, ...],
  "behavioral":     [{{"q": "...", "why_ask": "..."}}, ...],
  "gap_probing":    [{{"q": "...", "why_ask": "..."}}, ...],
  "role_specific":  [{{"q": "...", "why_ask": "..."}}]
}}

Requirements:
- technical: exactly 3 questions
- problem_solving: exactly 2 questions
- behavioral: exactly 2 questions
- gap_probing: exactly 2 questions (probe the weaknesses listed)
- role_specific: exactly 1 question"""

JSON_RETRY_SUFFIX = (
    "\n\nYour last response was not valid JSON. Return ONLY the JSON object."
)

SECTION_LIMITS = {
    "technical": 3,
    "problem_solving": 2,
    "behavioral": 2,
    "gap_probing": 2,
    "role_specific": 1,
}


def _normalize_section(items: Any, expected: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        items = []
    normalized = []
    for item in items[:expected]:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q", "")).strip()
        why = str(item.get("why_ask", "")).strip()
        if q:
            normalized.append({"q": q, "why_ask": why, "asked": False})
    while len(normalized) < expected:
        normalized.append(
            {
                "q": "Tell me more about your relevant experience for this role.",
                "why_ask": "Fallback question — regenerate for better targeting.",
                "asked": False,
            }
        )
    return normalized[:expected]


def _normalize_questions(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        section: _normalize_section(data.get(section), limit)
        for section, limit in SECTION_LIMITS.items()
    }


def generate_questions(
    resume_text: str,
    jd_text: str,
    weaknesses: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not resume_text.strip():
        raise LLMError("Resume text is empty — cannot generate interview questions.")
    if not jd_text.strip():
        raise LLMError("Job description text is empty — add a JD first.")

    weakness_str = ", ".join(weaknesses) if weaknesses else "None identified"
    prompt = INTERVIEW_USER_TEMPLATE.format(
        jd_text=jd_text,
        resume_text=resume_text,
        weaknesses=weakness_str,
    )

    client = get_client()
    try:
        raw = client.complete(prompt, system=INTERVIEW_SYSTEM, json_mode=True)
        parsed = _extract_json(raw)
    except (json.JSONDecodeError, ValueError, KeyError):
        raw = client.complete(
            prompt + JSON_RETRY_SUFFIX,
            system=INTERVIEW_SYSTEM,
            json_mode=True,
        )
        parsed = _extract_json(raw)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(
            "Could not generate interview questions. Please try again."
        ) from exc

    return _normalize_questions(parsed)

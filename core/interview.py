"""Tailored interview pack — unique questions from resume claims vs JD (no fixed DB)."""

from __future__ import annotations

import json
import os
from typing import Any

from core.llm_client import LLMError, get_client
from core.matcher import _extract_json

INTERVIEW_SYSTEM = (
    "You are a senior hiring manager and interviewer. You design unique interview "
    "questions from each candidate's resume claims and the job description — never "
    "reuse generic question banks. Favor practical scenario depth and gap validation."
)

INTERVIEW_USER_TEMPLATE = """Build a tailored interview pack for this shortlisted candidate.

=== JOB DESCRIPTION ===
{jd_text}

=== MUST-HAVES (JD) ===
{must_haves}

=== NICE-TO-HAVES (JD) ===
{nice_to_haves}

=== MUST-HAVES MET (resume evidence) ===
{must_matched}

=== MUST-HAVES MISSING (gaps to validate) ===
{must_missing}

=== WEAKNESSES FROM AI MATCH ===
{weaknesses}

=== RESUME EVIDENCE SNIPPETS ===
{evidence}

=== FULL RESUME ===
{resume_text}

Return ONLY valid JSON, no markdown fences:
{{
  "focus_skills": ["skills/claims from resume to probe, e.g. microservices, Node.js, Redis, AWS, leadership"],
  "suggestions": [
    "3-5 short Interview Pack Behavior tips for the hiring manager (how to run this interview, what to listen for)"
  ],
  "scenario_depth": [
    {{
      "q": "scenario question testing practical depth on a claimed skill",
      "why_ask": "what evidence this validates",
      "targets": ["skill or claim from resume"],
      "question_type": "scenario_depth"
    }}
  ],
  "gap_validation": [
    {{
      "q": "question for a MISSING must-have — test transferable knowledge, not trivia",
      "why_ask": "which gap and what good answer looks like",
      "targets": ["missing must-have"],
      "question_type": "gap_validation"
    }}
  ],
  "behavioral": [
    {{
      "q": "behavioral / leadership / collaboration scenario",
      "why_ask": "competency tested",
      "targets": ["behavioral theme"],
      "question_type": "behavioral"
    }}
  ],
  "role_specific": [
    {{
      "q": "question tied directly to this JD role outcomes",
      "why_ask": "role fit",
      "targets": ["JD theme"],
      "question_type": "role_specific"
    }}
  ]
}}

Rules:
- Do NOT use a fixed question database — every question must reference THIS resume and THIS JD.
- scenario_depth: exactly 3 questions. If resume claims microservices, Node.js, Redis, cloud, DevOps, or leadership, write hands-on scenarios (incident, scale, trade-offs, team outcomes) — not definitions.
- gap_validation: exactly 2 questions — one per important missing must-have; probe transferable experience.
- behavioral: exactly 2 questions — ownership, conflict, mentoring, delivery under pressure.
- role_specific: exactly 2 questions — tie to JD must-haves and seniority.
- focus_skills: list 4-8 concrete technologies/themes claimed in the resume.
- suggestions: actionable guidance for the interviewer (tone, red flags, follow-ups)."""

JSON_RETRY_SUFFIX = (
    "\n\nYour last response was not valid JSON. Return ONLY the JSON object."
)

SECTION_LIMITS = {
    "scenario_depth": 3,
    "gap_validation": 2,
    "behavioral": 2,
    "role_specific": 2,
}

# LLM often returns legacy section names — map into v2 buckets
SECTION_ALIASES: dict[str, str] = {
    "technical": "scenario_depth",
    "problem_solving": "scenario_depth",
    "technical_validation": "scenario_depth",
    "gap_probing": "gap_validation",
    "gap_probing_questions": "gap_validation",
    "role_specific_questions": "role_specific",
}

SECTION_LABELS = {
    "scenario_depth": "Scenario depth (resume claims)",
    "gap_validation": "Gap validation (missing must-haves)",
    "behavioral": "Behavioral & leadership",
    "role_specific": "Role-specific (JD)",
}

MATCH_RESUME_CHAR_LIMIT = int(os.getenv("MATCH_RESUME_CHAR_LIMIT", "6000"))
MATCH_JD_CHAR_LIMIT = int(os.getenv("MATCH_JD_CHAR_LIMIT", "8000"))


def _blank_question(qtype: str) -> dict[str, Any]:
    return {
        "q": "Regenerate pack for a tailored question.",
        "why_ask": "Placeholder — prior generation incomplete.",
        "targets": [],
        "question_type": qtype,
        "asked": False,
        "rating": None,
        "interviewer_note": "",
    }


def _normalize_question(item: Any, qtype: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        return _blank_question(qtype)
    q = str(item.get("q", "")).strip()
    if not q:
        return _blank_question(qtype)
    targets = item.get("targets")
    if not isinstance(targets, list):
        targets = [str(targets)] if targets else []
    return {
        "q": q,
        "why_ask": str(item.get("why_ask", "")).strip()[:500],
        "targets": [str(t).strip()[:80] for t in targets if str(t).strip()][:5],
        "question_type": str(item.get("question_type") or qtype),
        "asked": bool(item.get("asked")),
        "rating": item.get("rating") if item.get("rating") in (1, 2, 3, 4, 5) else None,
        "interviewer_note": str(item.get("interviewer_note", ""))[:2000],
    }


def _is_real_question(q: dict[str, Any]) -> bool:
    text = q.get("q", "")
    return bool(text) and "Regenerate pack" not in text


def _normalize_section(
    items: Any,
    section: str,
    expected: int,
    *,
    pad_blanks: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        items = []
    normalized = [_normalize_question(item, section) for item in items if item]
    normalized = [q for q in normalized if _is_real_question(q)]
    if pad_blanks:
        while len(normalized) < expected:
            normalized.append(_blank_question(section))
    return normalized[:expected]


def _collect_sections_from_parsed(parsed: dict[str, Any]) -> dict[str, list[Any]]:
    """Merge v2 and legacy LLM keys into one bucket per section."""
    buckets: dict[str, list[Any]] = {key: [] for key in SECTION_LIMITS}

    def _add_items(target: str, items: Any) -> None:
        if not isinstance(items, list):
            return
        buckets[target].extend(items)

    nested = parsed.get("sections")
    if isinstance(nested, dict):
        for key, items in nested.items():
            key_l = str(key).lower().replace(" ", "_")
            if key_l in buckets:
                _add_items(key_l, items)
            elif key_l in SECTION_ALIASES:
                _add_items(SECTION_ALIASES[key_l], items)

    for key, items in parsed.items():
        if key in ("version", "sections", "suggestions", "focus_skills"):
            continue
        key_l = str(key).lower().replace(" ", "_")
        if key_l in buckets:
            _add_items(key_l, items)
        elif key_l in SECTION_ALIASES:
            _add_items(SECTION_ALIASES[key_l], items)

    return buckets


def _build_sections(parsed: dict[str, Any], *, pad_blanks: bool = False) -> dict[str, list[dict[str, Any]]]:
    buckets = _collect_sections_from_parsed(parsed)
    return {
        section: _normalize_section(buckets[section], section, limit, pad_blanks=pad_blanks)
        for section, limit in SECTION_LIMITS.items()
    }


def _sections_missing_real(sections: dict[str, list[dict[str, Any]]]) -> list[str]:
    missing = []
    for section, limit in SECTION_LIMITS.items():
        real = [q for q in sections.get(section, []) if _is_real_question(q)]
        if len(real) < limit:
            missing.append(section)
    return missing


SECTION_RETRY_TEMPLATE = """The interview pack JSON was incomplete. Generate ONLY these missing sections for the same candidate.

Missing sections: {missing}

JD excerpt:
{jd_text}

Resume excerpt:
{resume_text}

Must-haves missing: {must_missing}

Return ONLY valid JSON with these keys and the required question counts:
{schema}
"""


def _format_evidence(evidence: list[dict[str, Any]] | None) -> str:
    if not evidence:
        return "None"
    lines = []
    for item in evidence[:8]:
        req = item.get("requirement", "")
        quote = item.get("quote", "")
        matched = "yes" if item.get("matched") else "no"
        lines.append(f"- [{matched}] {req}: \"{quote}\"")
    return "\n".join(lines) or "None"


def _format_list(items: list[str] | None, fallback: str = "None") -> str:
    if not items:
        return fallback
    return "\n".join(f"- {x}" for x in items[:12])


def build_interview_pack(
    resume_text: str,
    jd_text: str,
    *,
    candidate_id: str,
    candidate_name: str,
    jd_id: str,
    jd_title: str = "",
    weaknesses: list[str] | None = None,
    must_haves: list[str] | None = None,
    nice_to_haves: list[str] | None = None,
    must_matched: list[str] | None = None,
    must_missing: list[str] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    match_summary: str = "",
    overall_score: int | None = None,
) -> dict[str, Any]:
    """Generate full interview pack v2 with sections + HM suggestions."""
    if not resume_text.strip():
        raise LLMError("Resume text is empty — cannot generate interview questions.")
    if not jd_text.strip():
        raise LLMError("Job description text is empty — add a JD first.")

    prompt = INTERVIEW_USER_TEMPLATE.format(
        jd_text=jd_text[:MATCH_JD_CHAR_LIMIT],
        resume_text=resume_text[:MATCH_RESUME_CHAR_LIMIT],
        must_haves=_format_list(must_haves),
        nice_to_haves=_format_list(nice_to_haves),
        must_matched=_format_list(must_matched),
        must_missing=_format_list(must_missing, "All must-haves appear met or unclear"),
        weaknesses=", ".join(weaknesses) if weaknesses else "None identified",
        evidence=_format_evidence(evidence),
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

    sections = _build_sections(parsed, pad_blanks=False)

    missing = _sections_missing_real(sections)
    if missing:
        schema_parts = []
        for sec in missing:
            limit = SECTION_LIMITS[sec]
            schema_parts.append(f'"{sec}": [{{"q":"...","why_ask":"...","targets":[]}}] x{limit}')
        retry_prompt = SECTION_RETRY_TEMPLATE.format(
            missing=", ".join(missing),
            jd_text=jd_text[:4000],
            resume_text=resume_text[:4000],
            must_missing=_format_list(must_missing, "None"),
            schema="{ " + ", ".join(schema_parts) + " }",
        )
        try:
            retry_raw = client.complete(
                retry_prompt + JSON_RETRY_SUFFIX,
                system=INTERVIEW_SYSTEM,
                json_mode=True,
            )
            retry_parsed = _extract_json(retry_raw)
            retry_sections = _build_sections(retry_parsed, pad_blanks=False)
            for sec in missing:
                if retry_sections.get(sec):
                    existing = sections.get(sec, [])
                    seen = {q["q"] for q in existing}
                    for q in retry_sections[sec]:
                        if _is_real_question(q) and q["q"] not in seen:
                            existing.append(q)
                            seen.add(q["q"])
                    sections[sec] = existing[: SECTION_LIMITS[sec]]
        except Exception:
            pass

    sections = {
        section: _normalize_section(
            sections.get(section, []),
            section,
            SECTION_LIMITS[section],
            pad_blanks=False,
        )
        for section in SECTION_LIMITS
    }

    focus = parsed.get("focus_skills")
    if not isinstance(focus, list):
        focus = []
    focus_skills = [str(x).strip() for x in focus if str(x).strip()][:10]

    suggestions = parsed.get("suggestions")
    if not isinstance(suggestions, list):
        suggestions = []
    suggestions = [str(s).strip() for s in suggestions if str(s).strip()][:8]

    return {
        "version": 2,
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "jd_id": jd_id,
        "jd_title": jd_title,
        "overall_score": overall_score,
        "match_summary": match_summary[:800],
        "focus_skills": focus_skills,
        "suggestions": suggestions,
        "sections": sections,
        "interviewer_notes": "",
        "final_recommendation": "",
    }


def migrate_legacy_pack(legacy: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Convert pre-v2 flat section keys into pack v2."""
    merged_input = dict(legacy)
    if legacy.get("version") == 2 and isinstance(legacy.get("sections"), dict):
        merged_input = {**legacy, **legacy["sections"]}

    sections = _build_sections(merged_input, pad_blanks=False)

    return {
        "version": 2,
        "candidate_id": legacy.get("candidate_id") or candidate_id,
        "candidate_name": legacy.get("candidate_name", ""),
        "jd_id": legacy.get("jd_id", ""),
        "jd_title": legacy.get("jd_title", ""),
        "focus_skills": legacy.get("focus_skills") or [],
        "suggestions": legacy.get("suggestions") or [],
        "sections": sections,
        "interviewer_notes": legacy.get("interviewer_notes", ""),
        "final_recommendation": legacy.get("final_recommendation", ""),
        "match_summary": legacy.get("match_summary", ""),
        "overall_score": legacy.get("overall_score"),
    }


def normalize_stored_pack(raw: dict[str, Any] | None, candidate_id: str) -> dict[str, Any] | None:
    if not raw:
        return None
    return migrate_legacy_pack(raw, candidate_id)


def generate_questions(
    resume_text: str,
    jd_text: str,
    weaknesses: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-compatible entry — returns sections only for old callers."""
    pack = build_interview_pack(
        resume_text,
        jd_text,
        candidate_id=kwargs.get("candidate_id", ""),
        candidate_name=kwargs.get("candidate_name", ""),
        jd_id=kwargs.get("jd_id", ""),
        weaknesses=weaknesses,
        must_haves=kwargs.get("must_haves"),
        nice_to_haves=kwargs.get("nice_to_haves"),
        must_matched=kwargs.get("must_matched"),
        must_missing=kwargs.get("must_missing"),
        evidence=kwargs.get("evidence"),
    )
    return pack["sections"]


def pack_to_export_text(pack: dict[str, Any], candidate_label: str) -> str:
    lines = [
        f"Interview Pack — {candidate_label}",
        f"Role: {pack.get('jd_title') or '—'}",
        "=" * 48,
        "",
    ]
    if pack.get("match_summary"):
        lines.append(f"Match summary: {pack['match_summary']}")
        lines.append("")
    if pack.get("focus_skills"):
        lines.append("Focus skills: " + ", ".join(pack["focus_skills"]))
        lines.append("")
    if pack.get("suggestions"):
        lines.append("## Interview Pack Behavior (suggestions)")
        for i, s in enumerate(pack["suggestions"], 1):
            lines.append(f"{i}. {s}")
        lines.append("")

    sections = pack.get("sections") or pack
    for key, label in SECTION_LABELS.items():
        items = sections.get(key) if isinstance(sections, dict) else None
        if not items:
            continue
        lines.append(f"## {label}")
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item.get('q', '')}")
            lines.append(f"   Why: {item.get('why_ask', '')}")
            if item.get("targets"):
                lines.append(f"   Targets: {', '.join(item['targets'])}")
            if item.get("rating"):
                lines.append(f"   Rating: {item['rating']}/5")
            if item.get("interviewer_note"):
                lines.append(f"   Note: {item['interviewer_note']}")
            if item.get("asked"):
                lines.append("   [Asked]")
        lines.append("")

    if pack.get("interviewer_notes"):
        lines.append("## Interviewer notes")
        lines.append(pack["interviewer_notes"])
        lines.append("")
    rec = pack.get("final_recommendation")
    if rec:
        lines.append(f"## Final recommendation: {rec}")
    return "\n".join(lines)

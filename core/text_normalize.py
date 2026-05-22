"""Clean extracted PDF text before storage and embedding."""

from __future__ import annotations

import re
from collections import Counter

# Lines that are only keywords / punctuation (common stuffing pattern)
_KEYWORD_LINE = re.compile(
    r"^[\s,;|/\-•·]+(?:python|java|sql|aws|docker|kubernetes|leadership|"
    r"agile|scrum|excel|react|node|api|git|ci/cd)[\s,;|/\-•·]*$",
    re.IGNORECASE,
)
_REPEAT_WORDS = re.compile(r"\b(\w{2,})\b(?:\s+\1\b){4,}", re.IGNORECASE)


def normalize_resume_text(text: str) -> tuple[str, list[str]]:
    """
    Normalize resume text and return (clean_text, warnings).
    """
    warnings: list[str] = []
    if not text or not text.strip():
        return "", warnings

    lines = text.splitlines()
    cleaned: list[str] = []
    dropped_keyword_lines = 0

    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if len(raw) < 4 and not re.search(r"\d", raw):
            continue
        if _KEYWORD_LINE.match(raw):
            dropped_keyword_lines += 1
            continue
        cleaned.append(raw)

    if dropped_keyword_lines > 0:
        warnings.append(
            f"Removed {dropped_keyword_lines} suspicious keyword-only line(s)."
        )

    body = "\n".join(cleaned)
    body = _REPEAT_WORDS.sub(r"\1", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    words = re.findall(r"\b[a-z]{2,}\b", body.lower())
    if words:
        counts = Counter(words)
        top = counts.most_common(1)[0]
        if top[1] > max(12, len(words) // 8) and top[0] not in {
            "the",
            "and",
            "for",
            "with",
            "from",
        }:
            warnings.append(
                f"Possible keyword stuffing: '{top[0]}' appears {top[1]} times."
            )

    return body, warnings


def normalize_jd_text(text: str) -> str:
    """Lighter normalization for job descriptions."""
    if not text:
        return ""
    body = re.sub(r"[ \t]+", " ", text)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()

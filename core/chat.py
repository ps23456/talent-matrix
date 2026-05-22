"""Resume Q&A chat grounded in candidate text."""

from __future__ import annotations

from typing import Any

from core.llm_client import LLMError, get_client

CHAT_SYSTEM = """You are an AI assistant helping a hiring manager analyze a candidate's resume.
Answer only from the resume content below. If the answer isn't in the resume, say "Not mentioned in this resume." Be concise."""

CHAT_USER_TEMPLATE = """RESUME:
{resume_text}

CONVERSATION HISTORY:
{history}

USER: {question}"""


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(none)"
    lines = []
    for turn in history[:-1]:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(none)"


def answer_question(
    resume_text: str,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    if not resume_text.strip():
        raise LLMError("This resume has no extractable text to chat about.")

    question = question.strip()
    if not question:
        raise LLMError("Message cannot be empty.")

    history = list(history or [])
    prompt = CHAT_USER_TEMPLATE.format(
        resume_text=resume_text,
        history=_format_history(history + [{"role": "user", "content": question}]),
        question=question,
    )

    client = get_client()
    try:
        return client.complete(prompt, system=CHAT_SYSTEM, json_mode=False).strip()
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(
            "Could not get a response right now. Please try again."
        ) from exc


def append_turn(
    history: list[dict[str, str]],
    role: str,
    content: str,
) -> list[dict[str, str]]:
    updated = list(history)
    updated.append({"role": role, "content": content})
    return updated

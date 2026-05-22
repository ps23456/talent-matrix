"""SQLite persistence for resumes, JDs, and related workspace data."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Integer, String, Text, create_engine, delete, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "talentmatch.db"
UPLOADS_DIR = DATA_DIR / "uploads"
RESUME_FILES_DIR = UPLOADS_DIR / "resumes"
JD_FILES_DIR = UPLOADS_DIR / "jds"

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class ResumeRow(Base):
    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    text: Mapped[str] = mapped_column(Text, default="")
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    size_kb: Mapped[int] = mapped_column(Integer, default=0)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    uploaded_at: Mapped[str] = mapped_column(String(64))


class JdRow(Base):
    __tablename__ = "jds"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    text: Mapped[str] = mapped_column(Text, default="")
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    requirements_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(64))


class MatchResultsRow(Base):
    __tablename__ = "match_results"

    jd_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    results_json: Mapped[str] = mapped_column(Text, default="[]")
    requirements_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[str] = mapped_column(String(64))


class ChatHistoryRow(Base):
    __tablename__ = "chat_history"

    candidate_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    history_json: Mapped[str] = mapped_column(Text, default="[]")


class InterviewRow(Base):
    __tablename__ = "interview_questions"

    candidate_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    questions_json: Mapped[str] = mapped_column(Text, default="{}")


def _migrate_match_fingerprint_column() -> None:
    insp = inspect(engine)
    if not insp.has_table("match_results"):
        return
    cols = {c["name"] for c in insp.get_columns("match_results")}
    if "fingerprint" not in cols:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE match_results ADD COLUMN fingerprint VARCHAR(64)")
            )


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_FILES_DIR.mkdir(parents=True, exist_ok=True)
    JD_FILES_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    _migrate_match_fingerprint_column()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_pdf(folder: Path, record_id: str, content: bytes) -> str:
    path = folder / f"{record_id}.pdf"
    path.write_bytes(content)
    return str(path.relative_to(ROOT_DIR))


def _delete_file(rel_path: str | None) -> None:
    if not rel_path:
        return
    full = ROOT_DIR / rel_path
    if full.is_file():
        full.unlink(missing_ok=True)


def _resume_to_dict(row: ResumeRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "filename": row.filename,
        "text": row.text,
        "size_kb": row.size_kb,
        "ocr_used": row.ocr_used,
        "uploaded_at": row.uploaded_at,
        "file_path": row.file_path,
    }


def _jd_to_dict(row: JdRow) -> dict[str, Any]:
    requirements = None
    if row.requirements_json:
        try:
            requirements = json.loads(row.requirements_json)
        except json.JSONDecodeError:
            requirements = None
    return {
        "id": row.id,
        "title": row.title,
        "text": row.text,
        "ocr_used": row.ocr_used,
        "requirements": requirements,
        "file_path": row.file_path,
    }


def create_resume(
    resume_id: str,
    filename: str,
    text: str,
    content: bytes | None,
    size_kb: int,
    ocr_used: bool,
) -> dict[str, Any]:
    file_path = None
    if content:
        file_path = _save_pdf(RESUME_FILES_DIR, resume_id, content)
    row = ResumeRow(
        id=resume_id,
        filename=filename,
        text=text,
        file_path=file_path,
        size_kb=size_kb,
        ocr_used=ocr_used,
        uploaded_at=_now_iso(),
    )
    with SessionLocal() as session:
        session.merge(row)
        session.commit()
    return _resume_to_dict(row)


def list_resumes() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        rows = session.scalars(select(ResumeRow).order_by(ResumeRow.uploaded_at)).all()
        return [_resume_to_dict(r) for r in rows]


def get_resume(resume_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        row = session.get(ResumeRow, resume_id)
        return _resume_to_dict(row) if row else None


def clear_all_match_results() -> None:
    """Drop cached match rows when the resume pool changes."""
    with SessionLocal() as session:
        session.execute(delete(MatchResultsRow))
        session.commit()


def delete_resume(resume_id: str) -> bool:
    with SessionLocal() as session:
        row = session.get(ResumeRow, resume_id)
        if not row:
            return False
        _delete_file(row.file_path)
        session.delete(row)
        chat = session.get(ChatHistoryRow, resume_id)
        if chat:
            session.delete(chat)
        interview = session.get(InterviewRow, resume_id)
        if interview:
            session.delete(interview)
        session.commit()
        clear_all_match_results()
        return True


def create_jd(
    jd_id: str,
    title: str,
    text: str,
    content: bytes | None = None,
    ocr_used: bool = False,
) -> dict[str, Any]:
    file_path = None
    if content:
        file_path = _save_pdf(JD_FILES_DIR, jd_id, content)
    row = JdRow(
        id=jd_id,
        title=title,
        text=text,
        file_path=file_path,
        ocr_used=ocr_used,
        created_at=_now_iso(),
    )
    with SessionLocal() as session:
        session.merge(row)
        session.commit()
    return _jd_to_dict(row)


def list_jds() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        rows = session.scalars(select(JdRow).order_by(JdRow.created_at)).all()
        return [_jd_to_dict(r) for r in rows]


def get_jd(jd_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        row = session.get(JdRow, jd_id)
        return _jd_to_dict(row) if row else None


def get_all_jds_map() -> dict[str, dict[str, Any]]:
    return {j["id"]: j for j in list_jds()}


def update_jd_requirements(jd_id: str, requirements: dict[str, Any]) -> None:
    with SessionLocal() as session:
        row = session.get(JdRow, jd_id)
        if row:
            row.requirements_json = json.dumps(requirements)
            session.commit()


def delete_jd(jd_id: str) -> bool:
    with SessionLocal() as session:
        row = session.get(JdRow, jd_id)
        if not row:
            return False
        _delete_file(row.file_path)
        session.delete(row)
        match = session.get(MatchResultsRow, jd_id)
        if match:
            session.delete(match)
        session.commit()
        return True


def save_match_results(
    jd_id: str,
    results: list[dict[str, Any]],
    requirements: dict[str, Any] | None,
    fingerprint: str | None = None,
) -> None:
    with SessionLocal() as session:
        session.merge(
            MatchResultsRow(
                jd_id=jd_id,
                results_json=json.dumps(results),
                requirements_json=json.dumps(requirements) if requirements else None,
                fingerprint=fingerprint,
                updated_at=_now_iso(),
            )
        )
        session.commit()


def get_match_results(
    jd_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    with SessionLocal() as session:
        row = session.get(MatchResultsRow, jd_id)
        if not row:
            return [], None, None
        try:
            results = json.loads(row.results_json)
        except json.JSONDecodeError:
            results = []
        requirements = None
        if row.requirements_json:
            try:
                requirements = json.loads(row.requirements_json)
            except json.JSONDecodeError:
                requirements = None
        return results, requirements, row.fingerprint


def get_chat_history(candidate_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        row = session.get(ChatHistoryRow, candidate_id)
        if not row:
            return []
        try:
            return json.loads(row.history_json)
        except json.JSONDecodeError:
            return []


def save_chat_history(candidate_id: str, history: list[dict[str, Any]]) -> None:
    with SessionLocal() as session:
        session.merge(
            ChatHistoryRow(
                candidate_id=candidate_id,
                history_json=json.dumps(history),
            )
        )
        session.commit()


def get_interview_questions(candidate_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        row = session.get(InterviewRow, candidate_id)
        if not row:
            return None
        try:
            return json.loads(row.questions_json)
        except json.JSONDecodeError:
            return None


def save_interview_questions(candidate_id: str, questions: dict[str, Any]) -> None:
    with SessionLocal() as session:
        session.merge(
            InterviewRow(
                candidate_id=candidate_id,
                questions_json=json.dumps(questions),
            )
        )
        session.commit()


def clear_all_data() -> None:
    with SessionLocal() as session:
        for model in (
            InterviewRow,
            ChatHistoryRow,
            MatchResultsRow,
            JdRow,
            ResumeRow,
        ):
            session.execute(delete(model))
        session.commit()
    if UPLOADS_DIR.exists():
        shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
    RESUME_FILES_DIR.mkdir(parents=True, exist_ok=True)
    JD_FILES_DIR.mkdir(parents=True, exist_ok=True)

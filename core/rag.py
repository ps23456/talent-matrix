"""RAG: chunk resumes, embed, and semantic shortlist for matching."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from core.embeddings import embed_query, embed_texts
from core.matcher import _verdict_from_score, sort_results_by_verdict

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
COLLECTION_NAME = "resume_chunks"

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TOP_FRACTION = float(os.getenv("RAG_TOP_FRACTION", "0.20"))
MIN_SHORTLIST = int(os.getenv("RAG_MIN_SHORTLIST", "3"))
MAX_SHORTLIST = int(os.getenv("RAG_MAX_SHORTLIST", "30"))
ALWAYS_LLM_BELOW = int(os.getenv("RAG_ALWAYS_LLM_MAX_RESUMES", "6"))

_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _chunk_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 40]


def index_resume(resume_id: str, text: str) -> None:
    """Embed and store resume chunks. Replaces prior chunks for this resume."""
    delete_resume(resume_id)
    chunks = _chunk_text(text)
    if not chunks:
        return
    try:
        vectors = embed_texts(chunks)
    except Exception:
        return
    col = _get_collection()
    ids = [f"{resume_id}::{i}" for i in range(len(chunks))]
    metadatas = [{"resume_id": resume_id, "chunk_index": i} for i in range(len(chunks))]
    col.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)


def delete_resume(resume_id: str) -> None:
    try:
        col = _get_collection()
        existing = col.get(where={"resume_id": resume_id})
        if existing and existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        pass


def clear_index() -> None:
    global _collection
    try:
        import chromadb
        from chromadb.config import Settings

        if CHROMA_DIR.exists():
            import shutil

            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        _collection = None


def build_match_query(
    jd_text: str,
    requirements: dict[str, list[str]] | None,
    jd_title: str = "",
) -> str:
    req = requirements or {}
    must = req.get("must_haves") or []
    nice = req.get("nice_to_haves") or []
    parts = []
    if jd_title:
        parts.append(f"Job title: {jd_title}")
    parts.append(jd_text[:6000])
    if must:
        parts.append("Must-haves: " + "; ".join(must))
    if nice:
        parts.append("Nice-to-haves: " + "; ".join(nice))
    return "\n\n".join(parts)


def semantic_scores_for_resumes(
    query: str,
    resume_ids: list[str],
) -> dict[str, float]:
    """
    Return resume_id -> similarity score in [0, 1] (higher = better match).
    """
    if not resume_ids or not query.strip():
        return {rid: 0.0 for rid in resume_ids}

    try:
        qvec = embed_query(query)
        col = _get_collection()
        n_results = min(100, max(20, len(resume_ids) * 6))
        try:
            results = col.query(
                query_embeddings=[qvec],
                n_results=n_results,
                where={"resume_id": {"$in": resume_ids}},
                include=["metadatas", "distances"],
            )
        except Exception:
            results = col.query(
                query_embeddings=[qvec],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
    except Exception:
        return {rid: 0.0 for rid in resume_ids}

    best: dict[str, float] = {rid: 0.0 for rid in resume_ids}
    ids = results.get("ids") or [[]]
    distances = results.get("distances") or [[]]
    metas = results.get("metadatas") or [[]]

    if not ids or not ids[0]:
        return best

    allowed = set(resume_ids)
    for _id, dist, meta in zip(ids[0], distances[0], metas[0]):
        rid = (meta or {}).get("resume_id")
        if not rid or rid not in allowed:
            continue
        # Chroma cosine distance: 0 = identical, 2 = opposite
        similarity = max(0.0, 1.0 - (float(dist) / 2.0))
        best[rid] = max(best[rid], similarity)

    return best


def pick_shortlist(
    scores: dict[str, float],
    total: int,
) -> set[str]:
    """Choose resume IDs for full LLM scoring."""
    if total <= ALWAYS_LLM_BELOW:
        return set(scores.keys())

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    n = max(MIN_SHORTLIST, math.ceil(total * TOP_FRACTION))
    n = min(n, MAX_SHORTLIST, total)
    return {rid for rid, _ in ranked[:n]}


def build_semantic_only_result(
    resume: dict[str, Any],
    semantic_score: float,
) -> dict[str, Any]:
    """Placeholder result for candidates not deep-scored by LLM."""
    name = (resume.get("filename") or "Candidate").replace(".pdf", "")
    name = name.replace("_", " ").replace("-", " ").title()
    sem_pct = round(semantic_score * 100)
    return {
        "candidate_id": resume["id"],
        "candidate_name": name,
        "role_title": "—",
        "years_experience": 0,
        "location": "—",
        "overall_score": sem_pct,
        "verdict": _verdict_from_score(sem_pct),
        "must_have_score": 0,
        "nice_to_have_score": 0,
        "criteria_score": 0,
        "semantic_score": round(semantic_score, 4),
        "semantic_score_pct": sem_pct,
        "llm_scored": False,
        "rank": 0,
        "summary": (
            f"Semantic match {sem_pct}% — not in top shortlist for full AI scoring. "
            "Run with fewer resumes or lower RAG threshold to include."
        ),
        "scores": {},
        "strengths": [],
        "weaknesses": [],
        "must_have_matched": [],
        "must_have_missing": [],
        "nice_to_have_matched": [],
        "nice_to_have_missing": [],
        "missing_must_haves": [],
        "evidence": [],
        "authenticity_score": None,
        "authenticity_flags": ["Full AI evaluation skipped (semantic pre-filter)"],
        "better_fit_jd": None,
    }


def merge_llm_and_semantic(
    llm_results: list[dict[str, Any]],
    all_resumes: list[dict[str, Any]],
    semantic_scores: dict[str, float],
    shortlist: set[str],
) -> list[dict[str, Any]]:
    """Combine deep LLM rows with semantic-only rows for everyone else."""
    llm_by_id = {r["candidate_id"]: r for r in llm_results}
    merged: list[dict[str, Any]] = []

    for resume in all_resumes:
        rid = resume["id"]
        sem = semantic_scores.get(rid, 0.0)
        if rid in llm_by_id:
            row = dict(llm_by_id[rid])
            row["semantic_score"] = round(sem, 4)
            row["semantic_score_pct"] = round(sem * 100)
            row["llm_scored"] = True
            merged.append(row)
        else:
            merged.append(build_semantic_only_result(resume, sem))

    for row in merged:
        if row.get("llm_scored"):
            row["verdict"] = _verdict_from_score(int(row.get("overall_score", 0)))
        elif not row.get("verdict"):
            row["verdict"] = "SEMANTIC_ONLY"
    return sort_results_by_verdict(merged)

# The Talent Matrix

**Hire smarter, not harder.** AI-powered resume ↔ job description matching for recruiters — upload PDFs, score candidates, chat, and generate interview kits in one workspace.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) — lands on the **Overview** dashboard (welcome + pipeline stats).

### `.env` (minimum)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...    # required for scanned PDF OCR
```

## Features

- 📄 **Upload** — Multi-resume PDF + JD upload or paste; **SQLite** + on-disk PDF storage
- 🔍 **Mistral OCR** — Automatic fallback for scanned/image PDFs
- 🎯 **AI matching** — Must/nice + 8 criteria; **RAG semantic shortlist** + evidence/authenticity scoring
- 📊 **Compare top 3** — Radar chart across all criteria
- 🔀 **Better-fit tags** — Optional (`BETTER_FIT_ENABLED=true`); off by default for faster matching
- ⚡ **Stable fast matching** — `temperature=0`, parallel scoring, instant re-run when files unchanged (Shift+click Run to force re-score)
- 💬 **Resume chat** — Grounded Q&A per candidate
- 🎤 **Interview pack (shortlisted only)** — Unique LLM questions from resume claims vs JD (scenarios, gap validation, behavioral); ratings, notes, recommendation, `.txt` export
- ⚙️ **Settings** — Swappable LLM (OpenAI / Anthropic / Gemini) + connection test

## Project structure

```
app.py              # All API routes (single entry point)
core/
  pdf_loader.py     # pypdf + Mistral OCR
  llm_client.py     # Anthropic / OpenAI / Gemini
  matcher.py        # Must/nice scoring + better-fit
  database.py       # SQLite (data/talentmatch.db + PDF uploads)
  text_normalize.py # Resume text cleanup (anti keyword-stuffing)
  embeddings.py     # OpenAI embeddings
  rag.py            # Chroma index + semantic shortlist
  chat.py           # Resume Q&A
  interview.py      # Interview questions
static/index.html   # Full UI (HTML + CSS + JS)
```

## API routes

| Method | Route |
|--------|--------|
| GET | `/` — App UI |
| POST | `/api/upload/resume`, `/api/upload/jd`, `/api/upload/jd-text` |
| GET | `/api/files`, `/api/resume/{id}` |
| DELETE | `/api/resume/{id}`, `/api/jd/{id}`, `/api/clear` |
| POST | `/api/match` |
| GET/POST | `/api/chat/{id}`, `/api/chat` |
| GET | `/api/interview/shortlisted?jd_id=` — Shortlisted candidates for HM |
| GET/POST/PUT | `/api/interview/{id}`, `/api/interview`, `/api/interview/export` |
| GET | `/api/settings` |
| POST | `/api/llm/test` |

## Demo flow (hackathon checklist)

1. Upload 3–5 resume PDFs + 2 JDs (one can be scanned for OCR badge)
2. **AI Matching** → pick JD → Run → see ranked cards with animated scores
3. **Compare Top 3** on HOLD/REJECT with 2+ JDs → check better-fit pills
4. **Chat** → ask resume-specific questions
5. **Interview Kit** → generate + export
6. **Settings** → Test AI Connection

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Vanilla JS](https://img.shields.io/badge/Frontend-Vanilla_JS-F7DF1E?logo=javascript&logoColor=black)
![Chart.js](https://img.shields.io/badge/Charts-Chart.js-FF6384)

FastAPI · uvicorn · pypdf · Mistral OCR · OpenAI / Anthropic / Gemini

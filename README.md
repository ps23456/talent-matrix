# TalentMatch AI

**Hire smarter, not harder.** AI-powered resume ↔ job description matching for recruiters — score, rank, chat, and generate interview kits in one sleek workspace.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # or edit your existing .env
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

Set `LLM_PROVIDER` to `anthropic`, `openai`, or `gemini` and add the matching API key. For scanned PDFs, add `MISTRAL_API_KEY`.

## Features

- 📄 **Multi-resume upload** — PDFs in memory, instant list with OCR badge when needed
- 🎯 **8-criteria AI scoring** — weighted match (100 pts total), ranked candidate cards
- 📊 **Compare top 3** — radar chart across all criteria
- 💬 **Resume Q&A** — grounded chat per candidate
- 🎤 **Interview Kit** — 10 tailored questions in 5 sections, export to `.txt`
- 🔀 **Better-fit detection** — flags candidates who may fit another JD
- ⚙️ **Swappable LLMs** — Anthropic, OpenAI, or Gemini via env

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Vanilla JS](https://img.shields.io/badge/Frontend-Vanilla_JS-F7DF1E?logo=javascript&logoColor=black)
![Chart.js](https://img.shields.io/badge/Charts-Chart.js-FF6384)

FastAPI · uvicorn · pypdf · Mistral OCR · Anthropic / OpenAI / Gemini

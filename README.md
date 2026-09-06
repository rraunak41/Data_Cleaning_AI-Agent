# AI Data Cleaning & Preprocessing Agent

A simple, hybrid data pipeline that combines **deterministic** data cleaning
(Pandas / Scikit-learn) with an **AI layer** (OpenAI) that explains what
happened in plain English. Built to be easy to demo and explain in a
fresher-level interview.

## Architecture (high level)

```
                 ┌────────────────────┐
   CSV / API /   │   Ingestion Layer   │
   PostgreSQL ─▶ │  (scripts/data_    │
                 │   ingestion.py)     │
                 └──────────┬─────────┘
                            ▼
                 ┌────────────────────┐
                 │   Cleaning Layer    │
                 │ (scripts/data_      │
                 │  cleaning.py)       │
                 │ - snake_case cols   │
                 │ - dedupe (skip IDs) │
                 │ - null normalize    │
                 │ - median/mode fill  │
                 └──────────┬─────────┘
                            ▼
                 ┌────────────────────┐
                 │ Preprocessing Layer │
                 │ (scripts/           │
                 │  preprocessing.py)  │
                 │ - IQR outlier cap   │
                 │ - one-hot encode    │
                 │ - z-score scaling   │
                 └──────────┬─────────┘
                            ▼
                 ┌────────────────────┐
                 │   AI Layer (LLM)    │
                 │ (scripts/           │
                 │  ai_agent.py)       │
                 │ - audit summary     │
                 │ - recommendations   │
                 │ - graceful fallback │
                 └──────────┬─────────┘
                            ▼
        ┌───────────────────┴────────────────────┐
        ▼                                         ▼
┌───────────────┐                        ┌────────────────────┐
│ FastAPI Backend│                        │ Streamlit Dashboard │
│ (scripts/      │◀──── optional  ─────▶ │ (app/app.py)        │
│  backend.py)   │      REST calls        │ runs pipeline       │
└───────────────┘                        │ directly for demos   │
                                          └────────────────────┘
```

The Streamlit app calls the pipeline functions directly (simplest to run
and demo). The FastAPI backend exposes the same logic as REST endpoints,
so you can show both a "notebook-style" UI and a "real service" API in
the same project.

## Project structure

```
AI_Data_Cleaning/
├── .env.example
├── requirements.txt
├── README.md
├── INTERVIEW_NOTES.md
├── scripts/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── ai_agent.py
│   └── backend.py
└── app/
    └── app.py
```

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

- `OPENAI_API_KEY` is optional — if left blank, the AI Audit tab still
  works, just using a rule-based fallback summary instead of the LLM.
- `DB_*` variables are only needed if you want to test `ingest_db()`
  against a real PostgreSQL database.

### 4. (Optional) Set up a local PostgreSQL test database

```bash
# using psql
createdb ai_data_cleaning
psql ai_data_cleaning -c "CREATE TABLE sample (id SERIAL PRIMARY KEY, name TEXT, age INT);"
psql ai_data_cleaning -c "INSERT INTO sample (name, age) VALUES ('Alice', 30), ('Bob', NULL);"
```

## Running the project

### Streamlit dashboard (main demo, no backend needed)

```bash
streamlit run app/app.py
```

### FastAPI backend (optional, for the REST API)

**PowerShell (run in a separate terminal):**
```powershell
uvicorn scripts.backend:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for interactive Swagger docs.

### Running both concurrently (PowerShell)

```powershell
Start-Process powershell -ArgumentList "uvicorn scripts.backend:app --reload --port 8000"
streamlit run app/app.py
```

## How to use it

1. Open the Streamlit app and upload any CSV (e.g. the classic Titanic
   dataset works well for a demo — it has nulls, duplicates, outliers,
   and categorical columns).
2. Walk through the tabs in order: Ingestion → Cleaning → Preprocessing
   → AI Audit → Export.
3. Download the final processed CSV from the Export tab.

## Why this project is a good interview talking point

- Shows you understand the **difference between deterministic ETL logic
  and AI-assisted analysis**, and when to use each.
- Demonstrates **graceful degradation** — a production mindset (the app
  never crashes just because an API key is missing).
- Touches a realistic **full-stack slice**: ingestion, cleaning, feature
  engineering, a REST API, and a UI.

See `INTERVIEW_NOTES.md` for deeper technical explanations and sample
Q&A you can use to prepare.

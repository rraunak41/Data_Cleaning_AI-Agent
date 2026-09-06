"""
backend.py
------------------
FastAPI REST API exposing the data pipeline as endpoints:
  GET  /health     - simple liveness check
  POST /ingest      - upload a CSV file, get back JSON records
  POST /clean        - clean a JSON payload of records
  POST /preprocess    - run outlier capping + encoding + scaling
  POST /ai-audit    - generate an AI (or fallback) summary of a cleaning run

Run with:
  uvicorn scripts.backend:app --reload --port 8000
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scripts.data_cleaning import clean_data
from scripts.preprocessing import encode_categoricals, handle_outliers_iqr, scale_features
from scripts.ai_agent import build_df_profile, generate_data_audit_summary, recommend_cleaning_strategies

app = FastAPI(title="AI Data Cleaning & Preprocessing Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic schemas ----------

class RecordsPayload(BaseModel):
    records: List[Dict[str, Any]]


class PreprocessPayload(BaseModel):
    records: List[Dict[str, Any]]
    exclude_cols: Optional[List[str]] = None


class AuditPayload(BaseModel):
    records: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# ---------- Endpoints ----------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")

    return {"row_count": len(df), "records": df.to_dict(orient="records")}


@app.post("/clean")
def clean(payload: RecordsPayload) -> Dict[str, Any]:
    df = pd.DataFrame(payload.records)
    if df.empty:
        raise HTTPException(status_code=400, detail="No records provided.")

    cleaned_df, metadata = clean_data(df)
    return {"records": cleaned_df.to_dict(orient="records"), "metadata": metadata}


@app.post("/preprocess")
def preprocess(payload: PreprocessPayload) -> Dict[str, Any]:
    df = pd.DataFrame(payload.records)
    if df.empty:
        raise HTTPException(status_code=400, detail="No records provided.")

    df, capped_counts = handle_outliers_iqr(df)
    df = encode_categoricals(df)
    df = scale_features(df, exclude_cols=payload.exclude_cols)

    return {"records": df.to_dict(orient="records"), "outliers_capped": capped_counts}


@app.post("/ai-audit")
def ai_audit(payload: AuditPayload) -> Dict[str, Any]:
    df = pd.DataFrame(payload.records)
    if df.empty:
        raise HTTPException(status_code=400, detail="No records provided.")

    summary = generate_data_audit_summary(payload.metadata, df)
    profile = build_df_profile(df)
    recommendations = recommend_cleaning_strategies(profile)

    return {"summary": summary, "recommendations": recommendations}

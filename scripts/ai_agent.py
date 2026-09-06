"""
ai_agent.py
------------------
Thin wrapper around the OpenAI API that turns dataset metadata into
plain-English insights. If no valid API key is configured, every
function falls back to a simple rule-based summary instead of crashing
the app — this is the "graceful degradation" the assignment asks for.
"""

import os
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gpt-4o-mini"


def _get_client():
    """Return an OpenAI client, or None if no API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() in ("", "your_openai_api_key_here"):
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _fallback_audit(metadata: Dict[str, Any]) -> str:
    """Rule-based summary used when the OpenAI API is unavailable."""
    lines = ["[Fallback mode — no OpenAI API key configured]", ""]
    lines.append(f"- Rows: {metadata.get('initial_rows')} -> {metadata.get('final_rows')}")
    lines.append(f"- Duplicates removed: {metadata.get('duplicates_dropped')}")
    lines.append(
        f"- Missing values: {metadata.get('nulls_before')} -> {metadata.get('nulls_after')} "
        "(filled using median/mode imputation)"
    )
    lines.append(f"- Columns: {', '.join(metadata.get('columns', []))}")
    return "\n".join(lines)


def generate_data_audit_summary(metadata: Dict[str, Any], sample_df: pd.DataFrame) -> str:
    """Produce an executive-style summary of the cleaning pass. Uses the
    LLM when available, otherwise a deterministic fallback."""
    client = _get_client()
    if client is None:
        return _fallback_audit(metadata)

    sample_preview = sample_df.head(5).to_dict(orient="records")

    prompt = f"""
You are a data quality analyst. Write a short, executive-ready summary
(4-6 sentences, plain English, no bullet lists) of a data cleaning run.

Metadata: {metadata}
Sample rows (first 5): {sample_preview}

Mention: how many rows/duplicates were affected, how missing values
were handled, and one practical observation about the data quality.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return _fallback_audit(metadata) + f"\n\n[AI call failed, showing fallback. Error: {exc}]"


def _fallback_recommendations(df_profile: Dict[str, Any]) -> str:
    lines = ["[Fallback mode — no OpenAI API key configured]", ""]
    for col, info in df_profile.items():
        lines.append(f"- {col}: dtype={info.get('dtype')}, missing={info.get('missing_pct')}%")
    lines.append("\nGeneral suggestion: scale numeric features, one-hot encode low-cardinality "
                 "text columns, and review high-missingness columns manually.")
    return "\n".join(lines)


def recommend_cleaning_strategies(df_profile: Dict[str, Any]) -> str:
    """Given a per-column profile (dtype, missing %, cardinality, etc.),
    ask the LLM for tailored preprocessing recommendations."""
    client = _get_client()
    if client is None:
        return _fallback_recommendations(df_profile)

    prompt = f"""
You are a senior data engineer. Given this column-level profile of a
dataset, suggest concrete preprocessing steps per column (imputation
strategy, encoding, scaling, or flags for manual review). Keep it to a
short bullet list, one line per column.

Column profile: {df_profile}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return _fallback_recommendations(df_profile) + f"\n\n[AI call failed, showing fallback. Error: {exc}]"


def build_df_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Helper: build the per-column profile dict expected by
    recommend_cleaning_strategies()."""
    profile = {}
    for col in df.columns:
        profile[col] = {
            "dtype": str(df[col].dtype),
            "missing_pct": round(float(df[col].isna().mean()) * 100, 2),
            "unique_values": int(df[col].nunique(dropna=True)),
        }
    return profile

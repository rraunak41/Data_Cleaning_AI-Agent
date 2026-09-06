"""
app.py
------------------
Streamlit dashboard for the AI Data Cleaning & Preprocessing Agent.
Runs the whole pipeline locally (no need for the FastAPI backend to be
running) so it is simple to demo end-to-end: Ingestion -> Cleaning ->
Preprocessing -> AI Audit -> Export.

Run with:
  streamlit run app/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

# Allow importing from the sibling "scripts" package when run as
# `streamlit run app/app.py` from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_cleaning import clean_data
from scripts.preprocessing import encode_categoricals, handle_outliers_iqr, scale_features
from scripts.ai_agent import build_df_profile, generate_data_audit_summary, recommend_cleaning_strategies

st.set_page_config(page_title="AI Data Cleaning Agent", layout="wide")

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Configuration")

api_key_set = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
st.sidebar.markdown("**OpenAI API Key:** " + ("🟢 Configured" if api_key_set else "🔴 Not set (fallback mode)"))

st.sidebar.markdown("---")
st.sidebar.subheader("Database (optional)")
st.sidebar.text_input("DB Host", value=os.getenv("DB_HOST", "localhost"), disabled=True)
st.sidebar.text_input("DB Name", value=os.getenv("DB_NAME", "ai_data_cleaning"), disabled=True)

st.title("🧹 AI Data Cleaning & Preprocessing Agent")
st.caption("Hybrid Deterministic + LLM pipeline — upload a CSV to get started.")

# ---------- Session state ----------
for key in ("raw_df", "cleaned_df", "clean_meta", "processed_df", "outliers_capped"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------- Tabs ----------
tab_ingest, tab_clean, tab_preprocess, tab_audit, tab_export = st.tabs(
    ["1️⃣ Ingestion", "2️⃣ Cleaning", "3️⃣ Preprocessing", "4️⃣ AI Audit", "5️⃣ Export"]
)

# --- Tab 1: Ingestion ---
with tab_ingest:
    st.subheader("Upload a CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.session_state.raw_df = pd.read_csv(uploaded_file)

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        c1, c2 = st.columns(2)
        c1.metric("Total Rows", len(df))
        c2.metric("Total Columns", len(df.columns))
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("Upload a CSV to begin.")

# --- Tab 2: Cleaning ---
with tab_clean:
    st.subheader("Deterministic Data Cleaning")
    if st.session_state.raw_df is None:
        st.warning("Please upload a file in the Ingestion tab first.")
    else:
        if st.button("Run Cleaning Step"):
            cleaned_df, meta = clean_data(st.session_state.raw_df)
            st.session_state.cleaned_df = cleaned_df
            st.session_state.clean_meta = meta

        if st.session_state.cleaned_df is not None:
            meta = st.session_state.clean_meta
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", meta["final_rows"], delta=meta["final_rows"] - meta["initial_rows"])
            c2.metric("Duplicates Removed", meta["duplicates_dropped"])
            c3.metric("Nulls Before", meta["nulls_before"])
            c4.metric("Nulls After", meta["nulls_after"])
            st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True)

# --- Tab 3: Preprocessing ---
with tab_preprocess:
    st.subheader("Feature Preprocessing")
    if st.session_state.cleaned_df is None:
        st.warning("Please run the Cleaning step first.")
    else:
        exclude_cols_input = st.text_input(
            "Columns to exclude from scaling (comma-separated, e.g. target,survived)", value=""
        )
        if st.button("Run Preprocessing Step"):
            exclude_cols = [c.strip() for c in exclude_cols_input.split(",") if c.strip()]
            df, capped = handle_outliers_iqr(st.session_state.cleaned_df)
            df = encode_categoricals(df)
            df = scale_features(df, exclude_cols=exclude_cols)
            st.session_state.processed_df = df
            st.session_state.outliers_capped = capped

        if st.session_state.processed_df is not None:
            total_capped = sum(st.session_state.outliers_capped.values()) if st.session_state.outliers_capped else 0
            c1, c2 = st.columns(2)
            c1.metric("Outliers Capped (total values)", total_capped)
            c2.metric("Final Feature Columns", len(st.session_state.processed_df.columns))

            if st.session_state.outliers_capped:
                st.write("Outliers capped per column:", st.session_state.outliers_capped)

            st.dataframe(st.session_state.processed_df.head(50), use_container_width=True)

# --- Tab 4: AI Audit ---
with tab_audit:
    st.subheader("AI Data Quality Audit & Recommendations")
    if st.session_state.cleaned_df is None:
        st.warning("Please run the Cleaning step first.")
    else:
        if st.button("Generate AI Audit"):
            with st.spinner("Analyzing dataset..."):
                summary = generate_data_audit_summary(
                    st.session_state.clean_meta, st.session_state.cleaned_df
                )
                profile = build_df_profile(st.session_state.cleaned_df)
                recommendations = recommend_cleaning_strategies(profile)

            st.markdown("### 📋 Executive Summary")
            st.write(summary)
            st.markdown("### 💡 Recommendations")
            st.write(recommendations)

# --- Tab 5: Export ---
with tab_export:
    st.subheader("Export Final Data")
    final_df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.cleaned_df

    if final_df is None:
        st.warning("Run the pipeline first to generate exportable data.")
    else:
        st.dataframe(final_df.head(50), use_container_width=True)
        csv_bytes = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
        )

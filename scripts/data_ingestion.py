"""
data_ingestion.py
------------------
Handles reading raw data into a pandas DataFrame from three sources:
  1. A local CSV file
  2. A REST API endpoint that returns JSON
  3. A SQL table/query against a PostgreSQL database (via SQLAlchemy)

Kept intentionally simple (no connection pooling tuning, no retries) so it
is easy to explain line-by-line in an interview.
"""

import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load variables from a .env file (if present) into the environment
load_dotenv()


def ingest_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV file from disk into a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")
    return pd.read_csv(file_path)


def ingest_api(api_url: str) -> pd.DataFrame:
    """Call a REST API that returns JSON (a list of records) and load it
    into a DataFrame."""
    response = requests.get(api_url, timeout=15)
    response.raise_for_status()  # raises for 4xx/5xx responses
    data = response.json()

    # Some APIs wrap the list of records inside a key like "results" or "data"
    if isinstance(data, dict):
        for key in ("results", "data", "records"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    return pd.DataFrame(data)


def _build_db_url() -> str:
    """Build a PostgreSQL connection URL from environment variables."""
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "ai_data_cleaning")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def ingest_db(query: str, db_url: str = None) -> pd.DataFrame:
    """Run a SQL query against PostgreSQL and return the result as a
    DataFrame. If db_url is not supplied, it is built from env vars."""
    url = db_url or _build_db_url()
    engine = create_engine(url, pool_pre_ping=True)

    with engine.connect() as connection:
        result_df = pd.read_sql(text(query), connection)

    return result_df

"""Dashboard data loading utilities.

Uses DuckDB for memory-efficient queries on large Parquet files
and Streamlit caching for optimal performance within the free-tier
1 GB RAM limit.

Data artifacts are loaded locally when present (local dev / Git LFS).
When running on Streamlit Cloud, files are downloaded from Google Drive
on first access using a lightweight ``requests``-based downloader
(no ``gdown`` dependency).
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml

# Configuration helpers

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# Google Drive file IDs — used as fallback when files aren't present on disk.
_GDRIVE_IDS: dict[str, str] = {
    "canonical_products_e5.parquet": "11_cnPiDfttEPjiPLZreuQYSdFt5eYDN4",
    "feature_engineered_data.parquet": "16FH8zRgLZK68cFlcqSUQnWWu3TGnrQi_",
    "price_predictor_lgbm.joblib": "1vBzTXy7PvVKIXS1G7HjlNHCOSkhGrXhi",
    "shap_sample_data.parquet": "1nrkuLNfuBkd7XC1hfEP9UkXRfXLN8dW3",
    "shap_values.npy": "1kqwR3ailFxNH2Tfn9MWyNGgsZVS3edOx",
    "shap_base_value.txt": "1fczoPcm3JqreXfUZn6Cz_DnpQFeqJnXG",
    "market_dispersion.parquet": "15OH7gaHFK6G9aMt3N2g-RKfNNS5MNQYv",
    "price_leadership.parquet": "1HJhsw9SV6tp4TYi0MPeRJo79Zc_Da67i",
}


@st.cache_data
def _load_config() -> dict:
    """Load config.yaml once and cache."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve(rel_path: str) -> Path:
    return _PROJECT_ROOT / rel_path


# Google Drive download helper 


def _download_from_gdrive(file_id: str, destination: Path) -> None:
    """Download a file from Google Drive using requests (no gdown needed).

    Handles the large-file confirmation page automatically.
    """
    if destination.exists() and destination.stat().st_size > 100:
        return  # Already present and not an LFS pointer

    destination.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner(f"Downloading {destination.name}… (first run only)"):
        url = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(url, params={"id": file_id}, stream=True)

        # Handle virus-scan confirmation for large files
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                response = session.get(
                    url, params={"id": file_id, "confirm": value}, stream=True
                )
                break

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

    st.success(f"✓ Downloaded {destination.name}")


def _ensure_file(path: Path, gdrive_key: str | None = None) -> Path:
    """Ensure a data file exists — download from GDrive if missing.

    Parameters
    ----------
    path : Path
        Expected local path to the file.
    gdrive_key : str, optional
        Key into ``_GDRIVE_IDS``. If ``None``, uses the filename.

    Returns
    -------
    Path
        Verified path to the file.
    """
    key = gdrive_key or path.name

    if path.exists() and path.stat().st_size > 100:
        return path

    # Try to download from Google Drive
    file_id = _GDRIVE_IDS.get(key)
    if file_id:
        _download_from_gdrive(file_id, path)
        return path

    st.error(f"⚠️ Data file not found: `{path}`\n\nRun `python run.py precompute` locally to generate it.")
    st.stop()


# DuckDB connection (shared, cached)


@st.cache_resource
def _get_duckdb_conn() -> duckdb.DuckDBPyConnection:
    """Create a shared in-memory DuckDB connection."""
    return duckdb.connect(database=":memory:")


# Canonical products (9.5M rows — DuckDB queries)


@st.cache_data(ttl=3600)
def load_canonical_data() -> pd.DataFrame:
    """Load canonical products with only the columns needed for the dashboard.

    Uses DuckDB to read only required columns from the Parquet file,
    drastically reducing memory footprint.
    """
    cfg = _load_config()
    parquet_path = _resolve(cfg["data"]["processed_dir"]) / cfg["matching"]["output_filename"]
    _ensure_file(parquet_path, "canonical_products_e5.parquet")

    conn = _get_duckdb_conn()
    query = f"""
        SELECT supermarket, prices, canonical_name, own_brand, date
        FROM read_parquet('{parquet_path.as_posix()}')
    """
    df = conn.execute(query).fetchdf()

    # Optimise types
    df["supermarket"] = df["supermarket"].astype("category")
    df["own_brand"] = df["own_brand"].astype("category")
    df["canonical_name"] = df["canonical_name"].astype("category")
    df["prices"] = pd.to_numeric(df["prices"], downcast="float")
    df["date"] = pd.to_datetime(df["date"])
    return df


# Feature data (for price predictor)


@st.cache_data(ttl=3600)
def get_raw_features_df() -> pd.DataFrame:
    """Load a sample of the feature-engineered data for the Price Predictor."""
    cfg = _load_config()
    parquet_path = _resolve(cfg["data"]["processed_dir"]) / cfg["features"]["output_filename"]
    _ensure_file(parquet_path, "feature_engineered_data.parquet")

    conn = _get_duckdb_conn()
    query = f"SELECT * FROM read_parquet('{parquet_path.as_posix()}') LIMIT 10000"
    return conn.execute(query).fetchdf()


# Model


@st.cache_resource
def load_model():
    """Load the trained LightGBM model from disk."""
    cfg = _load_config()
    model_path = _resolve(cfg["model"]["output_dir"]) / cfg["model"]["model_filename"]
    _ensure_file(model_path, "price_predictor_lgbm.joblib")
    return joblib.load(model_path)


# SHAP pre-computed artifacts


@st.cache_data(ttl=3600)
def load_shap_sample_data() -> pd.DataFrame | None:
    """Load pre-computed SHAP sample data."""
    cfg = _load_config()
    path = _resolve(cfg["shap"]["output_dir"]) / "shap_sample_data.parquet"
    _ensure_file(path, "shap_sample_data.parquet")
    return pd.read_parquet(path, engine="pyarrow")


@st.cache_data(ttl=3600)
def load_shap_values() -> tuple[np.ndarray | None, float | None]:
    """Load pre-computed SHAP values and base value."""
    cfg = _load_config()
    shap_dir = _resolve(cfg["shap"]["output_dir"])

    shap_file = shap_dir / "shap_values.npy"
    base_file = shap_dir / "shap_base_value.txt"

    _ensure_file(shap_file, "shap_values.npy")
    _ensure_file(base_file, "shap_base_value.txt")

    shap_values = np.load(shap_file)
    with open(base_file, "r") as f:
        base_value = float(f.read().strip())

    return shap_values, base_value


# Market dynamics pre-computed artifacts


@st.cache_data(ttl=3600)
def load_market_dispersion() -> pd.Series | None:
    """Load pre-computed market dispersion time series."""
    cfg = _load_config()
    path = _resolve(cfg["market_dynamics"]["output_dir"]) / "market_dispersion.parquet"
    _ensure_file(path, "market_dispersion.parquet")
    df = pd.read_parquet(path, engine="pyarrow")
    return df["dispersion"]


@st.cache_data(ttl=3600)
def load_price_leadership() -> pd.DataFrame | None:
    """Load pre-computed price leadership data."""
    cfg = _load_config()
    path = _resolve(cfg["market_dynamics"]["output_dir"]) / "price_leadership.parquet"
    _ensure_file(path, "price_leadership.parquet")
    return pd.read_parquet(path, engine="pyarrow")
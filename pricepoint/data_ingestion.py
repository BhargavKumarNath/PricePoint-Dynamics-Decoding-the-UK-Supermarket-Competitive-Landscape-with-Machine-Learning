"""Data ingestion pipeline.

Reads raw retailer CSVs, cleans them, validates against the Pandera
schema, and writes the cleaned interim dataset to Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pricepoint.config import Settings
from pricepoint.schemas import RAW_DATA_SCHEMA

logger = logging.getLogger(__name__)


def load_raw_csvs(settings: Settings) -> pd.DataFrame:
    """Load and concatenate all raw retailer CSV files.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    pd.DataFrame
        Concatenated raw data.
    """
    raw_dir = settings.data.raw_dir
    frames: list[pd.DataFrame] = []

    for filename in settings.data.raw_files:
        filepath = raw_dir / filename
        if not filepath.exists():
            logger.warning("Raw file not found, skipping: %s", filepath)
            continue

        logger.info("Loading %s …", filepath.name)
        df = pd.read_csv(filepath, low_memory=False)
        frames.append(df)
        logger.info("  → %s rows loaded.", f"{len(df):,}")

    if not frames:
        raise FileNotFoundError(
            f"No raw CSV files found in {raw_dir}. "
            f"Expected: {settings.data.raw_files}"
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total raw records: %s", f"{len(combined):,}")
    return combined


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning transformations to raw data.

    - Coerce date column
    - Strip whitespace from string columns
    - Remove rows with null prices
    - Standardise supermarket names

    Parameters
    ----------
    df : pd.DataFrame
        Raw concatenated data.

    Returns
    -------
    pd.DataFrame
        Cleaned data.
    """
    logger.info("Cleaning raw data …")
    df = df.copy()

    # Coerce dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Strip string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    # Coerce prices
    if "prices" in df.columns:
        df["prices"] = pd.to_numeric(df["prices"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["prices"])
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %s rows with null/invalid prices.", f"{dropped:,}")

    logger.info("Cleaning complete. %s rows remaining.", f"{len(df):,}")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the raw data schema.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data.

    Returns
    -------
    pd.DataFrame
        Validated data (unchanged if valid).

    Raises
    ------
    pandera.errors.SchemaError
        If validation fails.
    """
    logger.info("Validating against RAW_DATA_SCHEMA …")
    validated = RAW_DATA_SCHEMA.validate(df, lazy=True)
    logger.info("Validation passed. ✓")
    return validated


def run_ingestion(settings: Settings) -> Path:
    """Execute the full ingestion pipeline.

    1. Load raw CSVs
    2. Clean
    3. Validate
    4. Save to interim Parquet

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the output Parquet file.
    """
    df = load_raw_csvs(settings)
    df = clean_raw_data(df)
    df = validate_data(df)

    output_dir = settings.data.interim_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned_supermarket_data.parquet"

    logger.info("Writing cleaned data to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info("Ingestion complete. Output: %s", output_path)

    return output_path

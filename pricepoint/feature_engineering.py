"""Feature engineering pipeline.

Computes temporal, momentum, and competitive features from
canonical product price data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pricepoint.config import Settings

logger = logging.getLogger(__name__)


def add_temporal_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    lag_days: list[int],
) -> pd.DataFrame:
    """Add rolling statistics and lag features.

    Parameters
    ----------
    df : pd.DataFrame
        Price data sorted by product and date.
    rolling_windows : list[int]
        Window sizes for rolling statistics (e.g., [7, 14, 30]).
    lag_days : list[int]
        Lag periods in days (e.g., [1, 7]).

    Returns
    -------
    pd.DataFrame
        Data with new temporal columns.
    """
    logger.info("Adding temporal features (windows=%s, lags=%s) …", rolling_windows, lag_days)
    df = df.sort_values(["canonical_name", "supermarket", "date"]).copy()

    group_cols = ["canonical_name", "supermarket"]

    for window in rolling_windows:
        grp = df.groupby(group_cols, observed=True)["prices"]
        df[f"price_rol_mean_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"price_rol_std_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f"price_rol_max_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        df[f"price_rol_min_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )

    for lag in lag_days:
        df[f"price_lag_{lag}d"] = df.groupby(group_cols, observed=True)["prices"].shift(lag)

    # Momentum: daily price change
    df["price_diff_1d"] = df.groupby(group_cols, observed=True)["prices"].diff(1)

    return df


def add_competitive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-retailer competitive context features.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with canonical_name and date columns.

    Returns
    -------
    pd.DataFrame
        Data with competitive features.
    """
    logger.info("Adding competitive features …")
    df = df.copy()

    # Daily market average per product
    market_avg = df.groupby(["canonical_name", "date"], observed=True)["prices"].transform("mean")
    df["price_vs_market_avg"] = df["prices"] - market_avg

    # Daily price rank (1 = cheapest)
    df["price_rank"] = df.groupby(["canonical_name", "date"], observed=True)["prices"].rank(
        method="dense"
    )

    # Is cheapest flag
    df["is_cheapest_in_market"] = (df["price_rank"] == 1).astype(int)

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical date encodings.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a ``date`` column.

    Returns
    -------
    pd.DataFrame
        Data with cyclical features.
    """
    logger.info("Adding cyclical date features …")
    df = df.copy()
    dt = df["date"].dt

    df["day_of_week_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)
    df["day_of_month_sin"] = np.sin(2 * np.pi * dt.day / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * dt.day / 31)
    df["week_of_year_sin"] = np.sin(2 * np.pi * dt.isocalendar().week.astype(int) / 52)
    df["week_of_year_cos"] = np.cos(2 * np.pi * dt.isocalendar().week.astype(int) / 52)

    return df


def run_feature_engineering(settings: Settings) -> Path:
    """Execute the full feature engineering pipeline.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the output feature-engineered Parquet file.
    """
    canonical_path = settings.data.processed_dir / settings.matching.output_filename
    if not canonical_path.exists():
        raise FileNotFoundError(
            f"Canonical products not found at {canonical_path}. Run matching first."
        )

    logger.info("Loading canonical products from %s …", canonical_path)
    df = pd.read_parquet(canonical_path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])

    df = add_temporal_features(df, settings.features.rolling_windows, settings.features.lag_days)
    df = add_competitive_features(df)
    df = add_cyclical_features(df)

    output_dir = settings.data.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.features.output_filename

    logger.info("Writing feature data to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info(
        "Feature engineering complete. %s rows × %s columns. Output: %s",
        f"{len(df):,}",
        len(df.columns),
        output_path,
    )

    return output_path

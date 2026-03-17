"""Semantic product matching pipeline.

Provides text normalization, Sentence-BERT embedding generation,
FAISS similarity search, and canonical product assignment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pricepoint.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation (migrated from src/data_processing.py)
# ---------------------------------------------------------------------------

_BRANDS_TO_REMOVE = frozenset(
    ["tesco", "asda", "sainsburys", "saintsburys", "morrisons", "aldi"]
)

_UNIT_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(kg|g|ml|l|m|pack|pk|x\d+(\.\d+)?\s?(kg|g|ml|l)?|x)\b",
    re.IGNORECASE,
)
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalise_product_name(name: str | None) -> str:
    """Aggressively normalise a product name for matching.

    Steps:
    1. Lowercase
    2. Strip units (e.g., ``500g``, ``1.5l``, ``6x``)
    3. Remove punctuation
    4. Remove known retailer brand names
    5. Collapse whitespace

    Parameters
    ----------
    name : str or None
        Raw product name.

    Returns
    -------
    str
        Normalised product name (empty string if input is invalid).
    """
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = _UNIT_PATTERN.sub("", name)
    name = _PUNCTUATION_PATTERN.sub("", name)
    for brand in _BRANDS_TO_REMOVE:
        name = name.replace(brand, "")
    name = _WHITESPACE_PATTERN.sub(" ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Embedding & matching pipeline
# ---------------------------------------------------------------------------


def generate_embeddings(
    product_names: pd.Series,
    model_name: str = "intfloat/e5-large",
    batch_size: int = 256,
) -> np.ndarray:
    """Generate Sentence-BERT embeddings for product names.

    Parameters
    ----------
    product_names : pd.Series
        Series of product name strings.
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape ``(n_products, embedding_dim)``.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    names = product_names.tolist()
    logger.info("Encoding %s product names …", f"{len(names):,}")
    embeddings = model.encode(
        names, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    logger.info("Embeddings generated. Shape: %s", embeddings.shape)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """Build a FAISS inner-product index from embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Normalised embedding matrix.

    Returns
    -------
    faiss.IndexFlatIP
        FAISS index ready for search.
    """
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    logger.info("FAISS index built with %s vectors (dim=%s).", f"{index.ntotal:,}", dim)
    return index


def find_canonical_matches(
    df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Run the full product matching pipeline.

    1. Normalise product names
    2. Generate embeddings
    3. Build FAISS index
    4. Search for nearest neighbours
    5. Assign canonical product IDs

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned supermarket data with a ``product_name`` column.
    settings : Settings
        Application settings.

    Returns
    -------
    pd.DataFrame
        Data with ``canonical_name`` column added.
    """
    df = df.copy()
    df["normalised_name"] = df["product_name"].apply(normalise_product_name)

    unique_names = df["normalised_name"].drop_duplicates().reset_index(drop=True)
    logger.info("Unique normalised names: %s", f"{len(unique_names):,}")

    embeddings = generate_embeddings(unique_names, model_name=settings.matching.model_name)
    index = build_faiss_index(embeddings)

    threshold = settings.matching.similarity_threshold
    logger.info("Searching for matches (threshold=%.2f) …", threshold)

    # Simple greedy clustering: assign each product to its nearest canonical
    k = 2  # self + nearest neighbour
    distances, indices = index.search(embeddings.astype(np.float32), k)

    canonical_map: dict[str, str] = {}
    for i, name in enumerate(unique_names):
        if name in canonical_map:
            continue
        canonical_map[name] = name  # self is canonical
        for j in range(1, k):
            neighbour_idx = indices[i, j]
            neighbour_name = unique_names[neighbour_idx]
            if distances[i, j] >= threshold and neighbour_name not in canonical_map:
                canonical_map[neighbour_name] = name

    df["canonical_name"] = df["normalised_name"].map(canonical_map)
    matched = df["canonical_name"].notna().sum()
    logger.info("Canonical matching complete. %s / %s matched.", f"{matched:,}", f"{len(df):,}")

    return df


def run_matching(settings: Settings) -> Path:
    """Execute the full product matching pipeline.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the output canonical products Parquet file.
    """
    interim_path = settings.data.interim_dir / "cleaned_supermarket_data.parquet"
    if not interim_path.exists():
        raise FileNotFoundError(
            f"Interim data not found at {interim_path}. Run ingestion first."
        )

    logger.info("Loading interim data from %s …", interim_path)
    df = pd.read_parquet(interim_path, engine="pyarrow")

    df = find_canonical_matches(df, settings)

    output_dir = settings.data.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.matching.output_filename

    logger.info("Writing canonical products to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info("Product matching complete. Output: %s", output_path)

    return output_path

"""
Utilities for working with locally bundled datasets.

We keep heavy population CSVs zipped in the repository to stay under Git
hosting limits. At runtime we extract individual CSVs on-demand so existing
code that expects `data/mwi_<dataset>_2020.csv` continues to work.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path("data")

# Ordered list mirrors the options exposed in the Streamlit app
POPULATION_DATASETS: Iterable[str] = [
    "general",
    "women",
    "men",
    "children_under_five",
    "youth_15_24",
    "elderly_60_plus",
    "women_of_reproductive_age_15_49",
]


def _population_csv_name(dataset_name: str) -> str:
    return f"mwi_{dataset_name}_2020.csv"


def ensure_population_csv(dataset_name: str) -> Path:
    """
    Ensure the uncompressed CSV for a population dataset exists locally.

    If a zipped version (`*.csv.zip`) is present and the raw CSV is missing,
    the archive is extracted into `data/`. The resulting CSV path is returned.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_name = _population_csv_name(dataset_name)
    csv_path = DATA_DIR / csv_name
    if csv_path.exists():
        return csv_path

    zip_path = DATA_DIR / f"{csv_name}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extract(csv_name, path=DATA_DIR)
        return csv_path

    raise FileNotFoundError(
        f"Population dataset '{dataset_name}' not found. "
        f"Expected either '{csv_path}' or '{zip_path}'."
    )


def ensure_all_population_csvs() -> None:
    """
    Extract all population CSVs that have a zipped counterpart.

    Useful for local development or maintenance scripts that expect every CSV
    to be present up-front.
    """
    missing_archives = []

    for dataset_name in POPULATION_DATASETS:
        try:
            ensure_population_csv(dataset_name)
        except FileNotFoundError as exc:
            missing_archives.append(str(exc))

    if missing_archives:
        raise FileNotFoundError(
            "One or more population datasets are missing:\n"
            + "\n".join(f"- {msg}" for msg in missing_archives)
        )


def hash_dataframe_for_cache(df: pd.DataFrame) -> str:
    """
    Provide a stable cache hash for pandas DataFrames that may contain
    otherwise unhashable dtypes (e.g., numpy arrays).
    """
    key = df.attrs.get("_gaia_cache_key")
    if key is not None:
        return str(key)
    columns_signature = "|".join(map(str, df.columns))
    rows, cols = df.shape
    return f"shape={rows}x{cols};cols={columns_signature}"


def prepare_population_dataframe(
    df: pd.DataFrame, dataset_name: str
) -> pd.DataFrame:
    """
    Normalize population DataFrame columns so they are safe to cache and
    attach a deterministic cache key used by `hash_dataframe_for_cache`.
    """
    drop_columns = [col for col in ("color",) if col in df.columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)

    pop_field = f"mwi_{dataset_name}_2020"
    total_pop = float(df[pop_field].sum()) if pop_field in df.columns else 0.0
    df.attrs["_gaia_cache_key"] = (
        f"population:{dataset_name}:{len(df)}:{total_pop:.0f}"
    )
    return df


POPULATION_CACHE_HASH_FUNCS = {pd.DataFrame: hash_dataframe_for_cache}


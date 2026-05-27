"""
LOAD & CLEAN MODULE — ACIS Insurance Dataset
=============================================
Responsibilities:
    1. Load raw pipe-delimited dataset with optimized dtypes
    2. Fix known data quality issues
    3. Handle missing values by strategy
    4. Save cleaned data as Parquet for fast downstream use

Usage:
    from scripts.load_and_clean import load_insurance_data

    df = load_insurance_data("../data/raw/MachineLearningRating_v3.txt")
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

PROCESSED_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed"
CLEANED_PARQUET = PROCESSED_DIR / "cleaned_data.parquet"

# Optimized column dtypes — reduces memory significantly
# str columns with few unique values → category
DTYPE_MAP: dict = {
    # Special: mixed types — converted to numeric after load
    "CapitalOutstanding":  "string",

    # Owner / policy holder
    "Citizenship":         "category",
    "LegalType":           "category",
    "Title":               "category",
    "Language":            "category",
    "Bank":                "category",
    "AccountType":         "category",
    "MaritalStatus":       "category",
    "Gender":              "category",

    # Location
    "Country":             "category",
    "Province":            "category",
    "PostalCode":          "category",
    "MainCrestaZone":      "category",
    "SubCrestaZone":       "category",

    # Vehicle
    "ItemType":            "category",
    "VehicleType":         "category",
    "make":                "category",
    "Model":               "category",
    "bodytype":            "category",
    "AlarmImmobiliser":    "category",
    "TrackingDevice":      "category",
    "WrittenOff":          "category",
    "Rebuilt":             "category",
    "Converted":           "category",

    # Policy / cover
    "TermFrequency":       "category",
    "ExcessSelected":      "category",
    "CoverCategory":       "category",
    "CoverType":           "category",
    "CoverGroup":          "category",
    "Section":             "category",
    "Product":             "category",
    "StatutoryClass":      "category",
    "StatutoryRiskType":   "category",
}

# Columns to parse as dates
PARSE_DATES: list[str] = ["TransactionMonth"]

# ── Missing value strategy ─────────────────────────────────────────────────────

# 100% or near-total missing — drop entirely
COLS_TO_DROP: list[str] = [
    "NewVehicle",               # 0 non-null  (100% missing)
    "NumberOfVehiclesInFleet",  # 0 non-null  (100% missing)
    "CrossBorder",              # 99.93% missing
]

# High-missing categoricals — fill with "Unknown" to preserve rows
COLS_FILL_UNKNOWN: list[str] = [
    "WrittenOff",    # 64.18% missing
    "Rebuilt",       # 64.18% missing
    "Converted",     # 64.18% missing
    "Bank",          # 14.59% missing
]

# Low-missing categoricals (≤5%) — fill with most frequent value
COLS_FILL_MODE: list[str] = [
    "VehicleType",
    "make",
    "Model",
    "bodytype",
    "VehicleIntroDate",
    "Gender",
    "MaritalStatus",
    "AccountType",
]

# Low-missing numericals (≤1%) — fill with median
COLS_FILL_MEDIAN: list[str] = [
    "mmcode",
    "Cylinders",
    "cubiccapacity",
    "kilowatts",
    "NumberOfDoors",
    "CapitalOutstanding",
    "CustomValueEstimate",
]


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _log_shape(df: pd.DataFrame, label: str) -> None:
    logger.info(f"{label}: {df.shape[0]:,} rows × {df.shape[1]} cols")


def _drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in COLS_TO_DROP if c in df.columns]
    dropped  = [c for c in COLS_TO_DROP if c not in df.columns]
    if dropped:
        logger.warning(f"Already absent — skipping drop: {dropped}")
    df = df.drop(columns=existing)
    logger.info(f"Dropped {len(existing)} unusable columns: {existing}")
    return df


def _fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    # CapitalOutstanding: string → numeric
    if "CapitalOutstanding" in df.columns:
        df["CapitalOutstanding"] = pd.to_numeric(
            df["CapitalOutstanding"], errors="coerce"
        ).astype("Float64")
        logger.info("CapitalOutstanding → Float64")

    # make: strip stray whitespace
    if "make" in df.columns:
        df["make"] = df["make"].astype(str).str.strip()
        logger.info("make → whitespace stripped")

    if "VehicleIntroDate" in df.columns:
    # Standardize mixed formats → datetime → extract year-month string
        df["VehicleIntroDate"] = pd.to_datetime(
        df["VehicleIntroDate"],
        format="mixed",
        errors="coerce"           # unparseable → NaT
    )
    logger.info("VehicleIntroDate → datetime (mixed formats standardized)")

    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based helper columns. Stored as str for Parquet compatibility."""
    if "TransactionMonth" in df.columns:
        df["TransactionYearMonth"] = (
            df["TransactionMonth"].dt.to_period("M").astype(str)
        )
        logger.info("TransactionYearMonth (str YYYY-MM) → added")
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values by column strategy.

    >70% missing  → dropped earlier
    10–70% missing → fill categorical with 'Unknown'
    <10%  missing  → fill categorical with mode / numerical with median
    """
    # Fill with "Unknown"
    for col in COLS_FILL_UNKNOWN:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].cat.add_categories("Unknown") if hasattr(df[col], "cat") else df[col]
                df[col] = df[col].fillna("Unknown")
                logger.info(f"{col}: filled {n:,} nulls → 'Unknown'")

    # Fill with mode
    for col in COLS_FILL_MODE:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                mode_val = df[col].mode(dropna=True)
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                    logger.info(f"{col}: filled {n:,} nulls → mode='{mode_val.iloc[0]}'")

    # Fill with median
    for col in COLS_FILL_MEDIAN:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                median_val = df[col].median(skipna=True)
                df[col] = df[col].fillna(median_val)
                logger.info(f"{col}: filled {n:,} nulls → median={median_val:.4f}")

    return df


def _validate_no_missing(df: pd.DataFrame) -> None:
    critical = ["TotalPremium", "TotalClaims", "SumInsured", "CalculatedPremiumPerTerm"]
    for col in critical:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"CRITICAL: '{col}' contains nulls after cleaning.")

    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if remaining.empty:
        logger.info("✅ No missing values remaining after cleaning")
    else:
        logger.warning(f"⚠️  Remaining nulls:\n{remaining.to_string()}")


def _save_parquet(df: pd.DataFrame) -> Path:
    """
    Save to Parquet — 5× smaller, 10× faster than CSV, preserves dtypes.
    Period dtype is converted to str (PyArrow limitation).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()

    # Convert any remaining Period columns → str (PyArrow can't serialize Period)
    period_cols = [
        col for col in df_to_save.columns
        if isinstance(df_to_save[col].dtype, pd.PeriodDtype)
    ]
    for col in period_cols:
        df_to_save[col] = df_to_save[col].astype(str)
        logger.info(f"{col}: Period → str for Parquet compatibility")

    df_to_save.to_parquet(CLEANED_PARQUET, index=False, engine="pyarrow")
    size_mb = CLEANED_PARQUET.stat().st_size / 1e6
    logger.info(f"💾 Saved → {CLEANED_PARQUET}  ({size_mb:.1f} MB)")
    return CLEANED_PARQUET


def _print_summary(df: pd.DataFrame) -> None:
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    cat_cols = [c for c in df.columns if str(df[c].dtype) == "category"]
    print("\n" + "=" * 55)
    print("  DATA LOADED & CLEANED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Rows        : {df.shape[0]:>12,}")
    print(f"  Columns     : {df.shape[1]:>12,}")
    print(f"  Memory      : {mem_mb:>11.2f} MB")
    print(f"  Categories  : {len(cat_cols):>12,} columns")
    print(f"  Saved to    : {CLEANED_PARQUET.name}")
    print("=" * 55 + "\n")


# =============================================================================
# PUBLIC API
# =============================================================================

def load_insurance_data(path: str | Path) -> pd.DataFrame:
    """
    Load, clean, and return the ACIS insurance dataset.

    Pipeline
    --------
    1. Read raw pipe-delimited file with optimized dtypes
    2. Fix data types (CapitalOutstanding, make)
    3. Add derived time columns (TransactionYearMonth)
    4. Drop unusable columns (100% missing)
    5. Impute missing values by strategy
    6. Validate no critical columns have nulls
    7. Save cleaned data to data/processed/cleaned_data.parquet

    Parameters
    ----------
    path : str | Path
        Path to raw MachineLearningRating_v3.txt

    Returns
    -------
    pd.DataFrame
        Fully cleaned dataset ready for EDA and modeling.

    Example
    -------
    >>> df = load_insurance_data("../data/raw/MachineLearningRating_v3.txt")
    >>> df.shape
    (1000098, 50)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Run 'dvc pull' to fetch data, or check your path."
        )

    # Step 1: Load
    logger.info(f"Loading: {path.name}")
    df = pd.read_csv(
        path,
        sep="|",
        dtype=DTYPE_MAP,
        parse_dates=PARSE_DATES,
        low_memory=False,
        na_values=["", "NULL", "NaN"],
    )
    _log_shape(df, "After load")

    # Step 2: Fix data types
    df = _fix_data_types(df)

    # Step 3: Add derived columns
    df = _add_derived_columns(df)

    # Step 4: Drop unusable columns
    df = _drop_unusable_columns(df)
    _log_shape(df, "After drop")

    # Step 5: Handle missing values
    df = _handle_missing_values(df)
    _log_shape(df, "After imputation")

    # Step 6: Validate
    _validate_no_missing(df)

    # Step 7: Save to Parquet
    _save_parquet(df)

    # Summary
    _print_summary(df)

    return df


def load_cleaned_data() -> pd.DataFrame:
    """
    Fast loader — reads pre-cleaned Parquet directly.
    Use this in all EDA and modeling notebooks after the first run.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset from Parquet cache.

    Example
    -------
    >>> from scripts.load_and_clean import load_cleaned_data
    >>> df = load_cleaned_data()
    """
    if not CLEANED_PARQUET.exists():
        raise FileNotFoundError(
            f"Cleaned Parquet not found at: {CLEANED_PARQUET}\n"
            "Run load_insurance_data() first to generate it."
        )
    logger.info(f"Loading cleaned data from: {CLEANED_PARQUET.name}")
    df = pd.read_parquet(CLEANED_PARQUET, engine="pyarrow")
    _log_shape(df, "Loaded from Parquet")
    return df

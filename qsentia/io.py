from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from . import utils


PathLike = Union[str, Path]
FileOrBuffer = Union[PathLike, io.BytesIO, io.StringIO]


def _resolve_path(path: PathLike) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path).expanduser().resolve()


def load_csv_or_parquet(source: FileOrBuffer) -> pd.DataFrame:
    """
    Load a CSV or Parquet file into a DataFrame.

    Parameters
    ----------
    source:
        Either a filesystem path or an in-memory buffer compatible with
        Streamlit's `UploadedFile`.
    """
    if isinstance(source, (str, Path)):
        path = _resolve_path(source)
        ext = path.suffix.lower()
        if ext in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    elif hasattr(source, "read"):
        # Streamlit's UploadedFile exposes .name and .type but behaves like BytesIO.
        name = getattr(source, "name", "uploaded_file").lower()
        buffer = io.BytesIO(source.read())  # copy for repeated reads
        buffer.seek(0)
        if name.endswith((".parquet", ".pq")):
            df = pd.read_parquet(buffer)
        else:
            df = pd.read_csv(buffer)
    else:
        raise TypeError(f"Unsupported input type: {type(source)}")

    df.columns = [str(c) for c in df.columns]
    return df


def load_sample() -> pd.DataFrame:
    """Return the bundled synthetic dataset for quickstarts."""
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_roundtrips.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample dataset missing at {sample_path}")
    return pd.read_csv(sample_path, parse_dates=["timestamp_open", "timestamp_close"])


def load_default(path: Optional[PathLike] = None) -> pd.DataFrame:
    """Load the primary dataset shipped with the app."""
    if path is not None:
        default_path = _resolve_path(path)
        if not default_path.exists():
            raise FileNotFoundError(f"Default dataset missing at {default_path}")
    else:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        default_path = None
        for candidate in ("roundtrips.csv", "roundtrip.csv"):
            candidate_path = data_dir / candidate
            if candidate_path.exists():
                default_path = candidate_path
                break
        if default_path is None:
            raise FileNotFoundError(f"No default dataset found in {data_dir}")
    return pd.read_csv(default_path)


def validate_schema(df: pd.DataFrame) -> Tuple[bool, utils.ColumnMap, pd.DataFrame]:
    """
    Validate the dataframe schema and return a (is_valid, column_map, report) tuple.

    The report lists expected canonical fields alongside the detected column names.
    Validation ensures critical fields (symbol, pnl, timestamp_close) are present.
    """
    column_map = utils.infer_columns(df)
    report_records = []
    for field in column_map.__dataclass_fields__:  # type: ignore[attr-defined]
        report_records.append(
            {
                "expected": field,
                "detected": getattr(column_map, field),
            }
        )
    report = pd.DataFrame(report_records)
    required_fields = ("symbol", "pnl", "timestamp_close")
    is_valid = all(getattr(column_map, field) is not None for field in required_fields)
    return is_valid, column_map, report


def schema_feedback(report: pd.DataFrame) -> str:
    """Render a friendly message summarising detected vs expected columns."""
    missing = report[report["detected"].isna()]
    if missing.empty:
        return "All required fields detected."
    missing_fields = ", ".join(missing["expected"])
    return f"Missing essential columns: {missing_fields}"


def coerce_dtypes(df: pd.DataFrame, column_map: utils.ColumnMap) -> pd.DataFrame:
    """
    Ensure timestamps and numerics use consistent dtypes.
    This homogenises data ahead of analytics and mirrors the original script's casting.
    """
    df = df.copy()

    for ts_col in (column_map.timestamp_open, column_map.timestamp_close):
        if ts_col and ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    if column_map.holding_period and column_map.holding_period in df.columns:
        df[column_map.holding_period] = pd.to_numeric(df[column_map.holding_period], errors="coerce")

    for num_col in (column_map.pnl, column_map.ret, column_map.qty, column_map.price_open, column_map.price_close):
        if num_col and num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    return df

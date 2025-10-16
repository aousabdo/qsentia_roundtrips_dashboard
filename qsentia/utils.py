from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Canonical schema used across the dashboard. The loader validates that at least
# the required subset is present; optional fields are leveraged when available.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "timestamp_open",
    "timestamp_close",
    "symbol",
    "side",
    "qty",
    "price_open",
    "price_close",
    "pnl",
    "return",
    "holding_period",
    "strategy",
    "run_id",
)

OPTIONAL_COLUMNS: Tuple[str, ...] = (
    "trade_id",
    "entry_ts",
    "exit_ts",
    "hold_days",
    "ret_pct",
    "ret_mean",
    "pnl_usd",
)

# Column inference candidates (lower-case names) reused from the legacy script.
COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "pnl": ("pnl", "pnl_usd", "profit", "pl", "p&l", "pnl_total"),
    "return": ("ret", "ret_pct", "return", "return_pct", "pct_return"),
    "symbol": ("symbol", "ticker", "asset"),
    "timestamp_open": (
        "timestamp_open",
        "ts_entry",
        "entry_ts",
        "date_entry",
        "time_open",
        "timestamp_entry",
    ),
    "timestamp_close": (
        "timestamp_close",
        "ts_exit",
        "exit_ts",
        "date_exit",
        "time_close",
        "timestamp_exit",
    ),
    "holding_period": ("hold_days", "holding_period", "holding_period_days"),
    "strategy": ("strategy", "strat", "model"),
    "run_id": ("run_id", "run", "batch_id"),
    "side": ("side", "position", "direction"),
    "qty": ("qty", "quantity", "size", "shares", "qty_closed"),
    "price_open": ("price_open", "entry_price", "open_price"),
    "price_close": ("price_close", "exit_price", "close_price"),
    "trade_id": ("trade_id", "id", "roundtrip_id"),
}

DEFAULT_WINDOWS: Dict[str, int] = {
    "trading_year": 252,
    "rolling_window": 20,
    "equity_smoothing": 5,
}

PLOTLY_CONFIG: Dict[str, object] = {
    "displaylogo": False,
    "responsive": True,
}


@dataclass(frozen=True)
class ColumnMap:
    pnl: Optional[str] = None
    ret: Optional[str] = None
    symbol: Optional[str] = None
    timestamp_open: Optional[str] = None
    timestamp_close: Optional[str] = None
    holding_period: Optional[str] = None
    strategy: Optional[str] = None
    run_id: Optional[str] = None
    side: Optional[str] = None
    qty: Optional[str] = None
    price_open: Optional[str] = None
    price_close: Optional[str] = None
    trade_id: Optional[str] = None


@dataclass
class FilterParams:
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    symbols: Optional[Sequence[str]] = None
    strategies: Optional[Sequence[str]] = None
    runs: Optional[Sequence[str]] = None
    min_trades_per_symbol: int = 0
    hold_days_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    pnl_clip: Optional[Tuple[Optional[float], Optional[float]]] = None
    winsor_pct: Optional[Tuple[float, float]] = None  # expressed as quantiles (0-1)


def infer_columns(df: pd.DataFrame) -> ColumnMap:
    """Infer canonical columns from the provided dataframe."""
    lower_map = {c.lower(): c for c in df.columns}

    def match(key: str) -> Optional[str]:
        for candidate in COLUMN_CANDIDATES.get(key, ()):
            if candidate in lower_map:
                return lower_map[candidate]
        return None

    return ColumnMap(
        pnl=match("pnl"),
        ret=match("return"),
        symbol=match("symbol"),
        timestamp_open=match("timestamp_open"),
        timestamp_close=match("timestamp_close"),
        holding_period=match("holding_period"),
        strategy=match("strategy"),
        run_id=match("run_id"),
        side=match("side"),
        qty=match("qty"),
        price_open=match("price_open"),
        price_close=match("price_close"),
        trade_id=match("trade_id"),
    )


def fmt_pct(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.2f}%"


def fmt_currency(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.2f}K"
    return f"${x:.2f}"


def winsorize(series: pd.Series, limits: Optional[Tuple[Optional[float], Optional[float]]]) -> pd.Series:
    """Winsorize a series using absolute bounds or quantile tuples."""
    if limits is None:
        return series
    lower, upper = limits
    s = series.copy()
    if lower is not None and upper is not None and 0 <= lower <= 1 and 0 <= upper <= 1 and upper > lower:
        lo = s.quantile(lower)
        hi = s.quantile(upper)
        return s.clip(lo, hi)
    if lower is not None:
        s = s.clip(lower=lower)
    if upper is not None:
        s = s.clip(upper=upper)
    return s


def lorenz_gini_from_abs(values: Iterable[float]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Return Lorenz curve coordinates and Gini coefficient for |values|."""
    x = np.asarray(list(values), dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    if x.size == 0:
        return None, None
    x = np.sort(np.abs(x))
    total = x.sum()
    if total <= 0:
        return None, None
    lorenz = np.cumsum(x) / total
    lorenz = np.insert(lorenz, 0, 0.0)
    gini = 1 - 2 * np.trapz(lorenz, dx=1 / (len(lorenz) - 1))
    return lorenz, float(gini)


def summarize_filters(filters: FilterParams) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    if filters.start or filters.end:
        start = filters.start.strftime("%Y-%m-%d") if filters.start else "start"
        end = filters.end.strftime("%Y-%m-%d") if filters.end else "end"
        summary["Date range"] = f"{start} → {end}"
    if filters.symbols:
        summary["Symbols"] = f"{len(filters.symbols)} selected"
    if filters.strategies:
        summary["Strategies"] = f"{', '.join(filters.strategies)}"
    if filters.runs:
        summary["Runs"] = f"{', '.join(filters.runs)}"
    if filters.min_trades_per_symbol:
        summary["Min trades/symbol"] = str(filters.min_trades_per_symbol)
    if filters.hold_days_range:
        lo, hi = filters.hold_days_range
        summary["Hold days"] = f"{lo or 0:.1f} ↔ {hi or '∞'}"
    if filters.pnl_clip:
        summary["P&L clip"] = f"[{filters.pnl_clip[0] or 'auto'}, {filters.pnl_clip[1] or 'auto'}]"
    return summary

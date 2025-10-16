from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import utils
from .io import coerce_dtypes


BOOTSTRAP_B = 3000
RNG_SEED = 42

# Tiering thresholds copied from the legacy notebook for continuity.
WIN_MIN_T1 = 0.65
WIN_MIN_T2 = 0.55
MILD_TAIL_P5 = -4.0
MIN_TRADES_FOR_TRUST = 20

SIZE_MULT_T1 = 1.25
SIZE_MULT_T2 = 0.75
SIZE_MULT_T3 = 0.50
SIZE_MULT_OUT = 0.25

CAP_T1 = 1.00
CAP_T2 = 0.60
CAP_T3 = 0.30
CAP_OUT = 0.20


def prepare_dataframe(df: pd.DataFrame, column_map: utils.ColumnMap) -> Tuple[pd.DataFrame, utils.ColumnMap]:
    """
    Normalise column dtypes and ensure holding period is present.

    Returns the mutated dataframe (copy) and an updated column map.
    """
    df = coerce_dtypes(df, column_map)

    # Compute holding period if missing/empty
    hold_col = column_map.holding_period
    if hold_col is None or hold_col not in df.columns or df[hold_col].fillna(0).eq(0).all():
        if column_map.timestamp_open and column_map.timestamp_close:
            hold_col = "holding_period_days"
            df[hold_col] = (
                df[column_map.timestamp_close] - df[column_map.timestamp_open]
            ).dt.total_seconds() / (60 * 60 * 24)
            column_map = replace(column_map, holding_period=hold_col)

    # Convenience normalised columns used throughout downstream analytics.
    if column_map.pnl and column_map.pnl in df.columns:
        df["_pnl"] = pd.to_numeric(df[column_map.pnl], errors="coerce")
    else:
        df["_pnl"] = np.nan

    if column_map.ret and column_map.ret in df.columns:
        df["_ret"] = pd.to_numeric(df[column_map.ret], errors="coerce")
    else:
        df["_ret"] = np.nan

    if column_map.holding_period and column_map.holding_period in df.columns:
        df["_hold_days"] = pd.to_numeric(df[column_map.holding_period], errors="coerce")
    else:
        df["_hold_days"] = np.nan

    if column_map.timestamp_close and column_map.timestamp_close in df.columns:
        df["_trade_date"] = df[column_map.timestamp_close].dt.normalize()
    else:
        df["_trade_date"] = pd.NaT

    return df, column_map


def apply_filters(df: pd.DataFrame, filters: utils.FilterParams, column_map: utils.ColumnMap) -> pd.DataFrame:
    """Filter the roundtrips DataFrame according to sidebar controls."""
    data = df.copy()

    # Base winsorised pnl column initialised before filters for reuse in charts.
    data["_pnl_w"] = data["_pnl"]

    if column_map.timestamp_close and filters.start is not None:
        data = data[data[column_map.timestamp_close] >= filters.start]
    if column_map.timestamp_close and filters.end is not None:
        data = data[data[column_map.timestamp_close] <= filters.end]

    if column_map.symbol and filters.symbols:
        data = data[data[column_map.symbol].isin(filters.symbols)]

    if column_map.strategy and filters.strategies:
        data = data[data[column_map.strategy].isin(filters.strategies)]

    if column_map.run_id and filters.runs:
        data = data[data[column_map.run_id].isin(filters.runs)]

    if filters.hold_days_range and column_map.holding_period:
        lo, hi = filters.hold_days_range
        series = pd.to_numeric(data[column_map.holding_period], errors="coerce")
        if lo is not None:
            data = data[series >= lo]
        if hi is not None:
            data = data[series <= hi]

    if filters.pnl_clip and column_map.pnl:
        lo, hi = filters.pnl_clip
        pnl_series = pd.to_numeric(data[column_map.pnl], errors="coerce")
        if lo is not None:
            data = data[pnl_series >= lo]
        if hi is not None:
            data = data[pnl_series <= hi]

    if filters.min_trades_per_symbol and column_map.symbol:
        counts = data[column_map.symbol].value_counts()
        keep = counts[counts >= filters.min_trades_per_symbol].index
        data = data[data[column_map.symbol].isin(keep)]

    if filters.winsor_pct:
        data["_pnl_w"] = utils.winsorize(data["_pnl"], filters.winsor_pct)

    return data.reset_index(drop=True)


def _daily_series(df: pd.DataFrame, column_map: utils.ColumnMap, column: str) -> pd.Series:
    if column_map.timestamp_close is None or df.empty:
        return pd.Series(dtype=float)
    dates = df[column_map.timestamp_close].dt.normalize()
    series = pd.Series(df[column].values, index=dates)
    grouped = series.groupby(level=0).sum().sort_index()
    return grouped


def _wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - half, center + half


def _bootstrap_mean_ci(values: pd.Series, B: int = BOOTSTRAP_B, alpha: float = 0.05) -> Tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if x.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, x.size, size=(B, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def pnl_concentration(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    s = pd.to_numeric(df["_pnl_w"], errors="coerce").dropna()
    if s.empty:
        return {"top5_share": None, "top1pct_share": None, "gini_abs": None}
    n = len(s)
    total = s.sum()
    top5_share = float(s.nlargest(min(5, n)).sum() / total) if total != 0 else np.nan
    top1_count = max(1, int(np.floor(0.01 * n)))
    top1_share = float(s.nlargest(top1_count).sum() / total) if total != 0 else np.nan
    lorenz, gini = utils.lorenz_gini_from_abs(s.values)
    return {
        "top5_share": top5_share,
        "top1pct_share": top1_share,
        "gini_abs": gini,
        "lorenz_curve": lorenz,
    }


def kpis(df: pd.DataFrame, column_map: utils.ColumnMap) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}

    pnl_series = pd.to_numeric(df["_pnl_w"], errors="coerce").dropna()
    if not pnl_series.empty:
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        metrics.update(
            {
                "total_trades": int(len(pnl_series)),
                "total_pnl": float(pnl_series.sum()),
                "avg_pnl": float(pnl_series.mean()),
                "median_pnl": float(pnl_series.median()),
                "win_rate": float((pnl_series > 0).mean()),
                "profit_factor": float(wins.sum() / -losses.sum()) if not losses.empty else np.nan,
                "avg_win": float(wins.mean()) if not wins.empty else np.nan,
                "avg_loss": float(-losses.mean()) if not losses.empty else np.nan,
                "expectancy": float((pnl_series > 0).mean() * (wins.mean() if not wins.empty else 0.0)
                                    - (pnl_series <= 0).mean() * (-losses.mean() if not losses.empty else 0.0)),
            }
        )
        if metrics["avg_loss"] and metrics["avg_loss"] > 0:
            metrics["payoff_ratio"] = metrics["avg_win"] / metrics["avg_loss"] if metrics["avg_win"] else np.nan
        else:
            metrics["payoff_ratio"] = np.nan
    else:
        metrics.update(
            {
                "total_trades": 0,
                "total_pnl": None,
                "avg_pnl": None,
                "median_pnl": None,
                "win_rate": None,
                "profit_factor": None,
                "avg_win": None,
                "avg_loss": None,
                "expectancy": None,
                "payoff_ratio": None,
            }
        )

    ret_series = pd.to_numeric(df["_ret"], errors="coerce").dropna()
    if not ret_series.empty:
        metrics.update(
            {
                "ret_mean": float(ret_series.mean()),
                "ret_median": float(ret_series.median()),
                "ret_std": float(ret_series.std(ddof=1)),
                "ret_skew": float(ret_series.skew()),
                "ret_kurt": float(ret_series.kurt()),
                "var_95": float(np.percentile(ret_series, 5)),
                "es_95": float(ret_series[ret_series <= np.percentile(ret_series, 5)].mean())
                if (ret_series <= np.percentile(ret_series, 5)).any()
                else np.nan,
            }
        )
    else:
        metrics.update(
            {
                "ret_mean": None,
                "ret_median": None,
                "ret_std": None,
                "ret_skew": None,
                "ret_kurt": None,
                "var_95": None,
                "es_95": None,
            }
        )

    concentration = pnl_concentration(df)
    metrics.update(concentration)

    equity = equity_curve(df, column_map)
    if not equity.empty:
        metrics["max_drawdown"] = float(equity["drawdown"].min())
        metrics["avg_hold_days"] = float(df["_hold_days"].mean())
        metrics["median_hold_days"] = float(df["_hold_days"].median())

        # Annualised metrics based on daily returns if available.
        if "daily_return" in equity:
            daily_return = equity["daily_return"].dropna()
            if not daily_return.empty and np.isfinite(daily_return.std(ddof=1)):
                mu = daily_return.mean()
                sigma = daily_return.std(ddof=1)
                metrics["ann_return"] = float(mu * utils.DEFAULT_WINDOWS["trading_year"])
                metrics["ann_vol"] = float(sigma * np.sqrt(utils.DEFAULT_WINDOWS["trading_year"]))
                metrics["sharpe"] = float(mu / sigma * np.sqrt(utils.DEFAULT_WINDOWS["trading_year"])) if sigma > 0 else np.nan
            else:
                metrics["ann_return"] = None
                metrics["ann_vol"] = None
                metrics["sharpe"] = None
        else:
            metrics["ann_return"] = None
            metrics["ann_vol"] = None
            metrics["sharpe"] = None
    else:
        metrics["max_drawdown"] = None
        metrics["avg_hold_days"] = None
        metrics["median_hold_days"] = None
        metrics["ann_return"] = None
        metrics["ann_vol"] = None
        metrics["sharpe"] = None

    return metrics


def equity_curve(df: pd.DataFrame, column_map: utils.ColumnMap) -> pd.DataFrame:
    if column_map.timestamp_close is None or df.empty:
        return pd.DataFrame()

    daily_pnl = _daily_series(df, column_map, "_pnl_w")
    if daily_pnl.empty:
        return pd.DataFrame()

    equity = daily_pnl.cumsum()
    rolling_max = equity.cummax()
    drawdown = equity - rolling_max

    if column_map.ret and column_map.ret in df.columns:
        daily_return = _daily_series(df, column_map, "_ret")
    else:
        daily_return = pd.Series(index=daily_pnl.index, dtype=float)

    base_series = daily_return if not daily_return.dropna().empty else daily_pnl

    window = utils.DEFAULT_WINDOWS["rolling_window"]
    rolling_sharpe = base_series.rolling(window).mean() / base_series.rolling(window).std(ddof=1)
    rolling_sharpe *= np.sqrt(utils.DEFAULT_WINDOWS["trading_year"])

    out = pd.DataFrame(
        {
            "date": daily_pnl.index,
            "daily_pnl": daily_pnl.values,
            "equity": equity.values,
            "rolling_sharpe": rolling_sharpe.values,
            "drawdown": drawdown.values,
        }
    )
    out["daily_return"] = daily_return.reindex(out["date"]).values
    return out


def rolling_metrics(df: pd.DataFrame, column_map: utils.ColumnMap, window: int = utils.DEFAULT_WINDOWS["rolling_window"]) -> Dict[str, pd.Series]:
    if column_map.timestamp_close is None or df.empty:
        return {}
    daily_pnl = _daily_series(df, column_map, "_pnl_w")
    if daily_pnl.empty:
        return {}
    rolling_vol = daily_pnl.rolling(window).std(ddof=1)
    rolling_mean = daily_pnl.rolling(window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rolling_sharpe = rolling_mean / rolling_vol
        rolling_sharpe *= np.sqrt(utils.DEFAULT_WINDOWS["trading_year"])
    return {
        "rolling_vol": rolling_vol,
        "rolling_mean": rolling_mean,
        "rolling_sharpe": rolling_sharpe,
    }


def drawdowns(df: pd.DataFrame, column_map: utils.ColumnMap) -> pd.DataFrame:
    curve = equity_curve(df, column_map)
    if curve.empty:
        return pd.DataFrame()
    underwater = curve[["date", "drawdown"]].copy()
    underwater["underwater"] = underwater["drawdown"] / curve["equity"].replace(0, np.nan)
    return underwater


def calendar_views(df: pd.DataFrame, column_map: utils.ColumnMap) -> Dict[str, object]:
    if column_map.timestamp_close is None or df.empty:
        return {}
    daily_pnl = _daily_series(df, column_map, "_pnl_w")
    if daily_pnl.empty:
        return {}

    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    equity = daily_pnl.cumsum()
    drawdown = equity - equity.cummax()

    daily_df = pd.DataFrame(
        {
            "date": daily_pnl.index,
            "daily_pnl": daily_pnl.values,
            "equity": equity.values,
            "drawdown": drawdown.values,
        }
    )

    best_day = daily_pnl.idxmax()
    worst_day = daily_pnl.idxmin()

    pivot_df = daily_df.copy()
    pivot_df["month"] = pivot_df["date"].dt.to_period("M").astype(str)
    pivot_df["weekday"] = pivot_df["date"].dt.day_name()

    pivot = pivot_df.pivot_table(
        values="daily_pnl",
        index="month",
        columns="weekday",
        aggfunc="sum",
        fill_value=0.0,
        observed=False,
    )

    monthly_series = pivot.sum(axis=1)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_series = pivot.sum(axis=0).reindex([w for w in weekday_order if w in pivot.columns])

    return {
        "daily": daily_df,
        "best_day": (best_day, float(daily_pnl.max())) if not daily_pnl.empty else None,
        "worst_day": (worst_day, float(daily_pnl.min())) if not daily_pnl.empty else None,
        "monthly": monthly_series,
        "weekday": weekday_series,
        "pivot": pivot,
    }


def per_symbol_table(df: pd.DataFrame, column_map: utils.ColumnMap) -> pd.DataFrame:
    if column_map.symbol is None or df.empty:
        return pd.DataFrame()

    grp = df.groupby(column_map.symbol)
    pnl = grp["_pnl_w"]

    trades = pnl.count().rename("trades")
    win_rate = pnl.apply(lambda s: (s > 0).mean()).rename("win_rate")
    total_pnl = pnl.sum().rename("total_pnl")
    avg_pnl = pnl.mean().rename("avg_pnl")
    median_pnl = pnl.median().rename("median_pnl")

    wins = pnl.apply(lambda s: s[s > 0].mean() if (s > 0).any() else np.nan).rename("avg_win")
    losses = pnl.apply(lambda s: -s[s < 0].mean() if (s < 0).any() else np.nan).rename("avg_loss")
    expectancy = (win_rate * wins - (1 - win_rate) * losses).rename("expectancy_usd")

    if "_ret" in df.columns and not df["_ret"].isna().all():
        ret_mean = grp["_ret"].mean().rename("ret_mean")
        ret_std = grp["_ret"].std(ddof=1).rename("ret_std")
        ret_p5 = grp["_ret"].quantile(0.05).rename("ret_p5")
        sharpe_like = ((ret_mean / ret_std).replace({np.inf: np.nan}) * np.sqrt(utils.DEFAULT_WINDOWS["trading_year"])).rename("sharpe_like")
    else:
        ret_mean = pd.Series(index=trades.index, dtype=float, name="ret_mean")
        ret_std = pd.Series(index=trades.index, dtype=float, name="ret_std")
        ret_p5 = pd.Series(index=trades.index, dtype=float, name="ret_p5")
        sharpe_like = pd.Series(index=trades.index, dtype=float, name="sharpe_like")

    hold_days = grp["_hold_days"].mean().rename("avg_hold_days")

    league = pd.concat(
        [
            trades,
            win_rate,
            total_pnl,
            avg_pnl,
            median_pnl,
            wins,
            losses,
            expectancy,
            ret_mean,
            ret_std,
            ret_p5,
            sharpe_like,
            hold_days,
        ],
        axis=1,
    )

    win_ci_lo = []
    win_ci_hi = []
    pnl_ci_lo = []
    pnl_ci_hi = []
    for sym, rows in df.groupby(column_map.symbol):
        pnl_series = rows["_pnl_w"].dropna()
        n = len(pnl_series)
        p = float((pnl_series > 0).mean()) if n else 0.0
        lo, hi = _wilson_ci(p, n)
        win_ci_lo.append(lo)
        win_ci_hi.append(hi)
        ci_lo, ci_hi = _bootstrap_mean_ci(pnl_series)
        pnl_ci_lo.append(ci_lo)
        pnl_ci_hi.append(ci_hi)

    league["win_ci_lo"] = win_ci_lo
    league["win_ci_hi"] = win_ci_hi
    league["avg_pnl_ci_lo"] = pnl_ci_lo
    league["avg_pnl_ci_hi"] = pnl_ci_hi

    # Tiering rules replicated from the legacy script
    def assign_tier(row: pd.Series) -> str:
        n = row["trades"]
        wr = row["win_rate"]
        expct = row["expectancy_usd"]
        tail = row["ret_p5"]
        if n < MIN_TRADES_FOR_TRUST:
            return "Outlier-Hold"
        if (wr < WIN_MIN_T2) or (pd.notna(tail) and tail < MILD_TAIL_P5) or (
            pd.notna(expct) and expct <= 0 and (pd.isna(tail) or tail < -3.5)
        ):
            return "Tier 3"
        if (wr >= WIN_MIN_T1) and (pd.notna(expct) and expct > 0) and (pd.isna(tail) or tail >= MILD_TAIL_P5):
            return "Tier 1"
        return "Tier 2"

    league["tier"] = league.apply(assign_tier, axis=1)

    def size_mult(row: pd.Series) -> float:
        t = row["tier"]
        if t == "Tier 1":
            return SIZE_MULT_T1
        if t == "Tier 2":
            return SIZE_MULT_T2
        if t == "Tier 3":
            return SIZE_MULT_T3
        return SIZE_MULT_OUT

    def size_cap(row: pd.Series) -> float:
        t = row["tier"]
        if t == "Tier 1":
            return CAP_T1
        if t == "Tier 2":
            return CAP_T2
        if t == "Tier 3":
            return CAP_T3
        return CAP_OUT

    league["size_mult"] = league.apply(size_mult, axis=1)
    league["size_cap"] = league.apply(size_cap, axis=1)

    tier_order = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2, "Outlier-Hold": 3}
    league = league.sort_values(
        by=["tier", "total_pnl"],
        key=lambda s: s.map(lambda x: tier_order.get(x, 4)) if s.name == "tier" else s,
        ascending=[True, False],
    )

    return league


def holding_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "_hold_days" not in df:
        return pd.DataFrame()

    hd = df["_hold_days"].clip(lower=0).dropna()
    bins = [-0.1, 1, 2, 3, 5, 10, 30, np.inf]
    labels = ["<1", "1-2", "2-3", "3-5", "5-10", "10-30", ">30"]
    categories = pd.cut(hd, bins=bins, labels=labels)
    out = pd.DataFrame(index=pd.Index(labels, name="holding_period_bin"))
    out["count"] = categories.value_counts().reindex(labels, fill_value=0)
    grouped = df.groupby(categories, observed=False)
    pnl_group = grouped["_pnl_w"]
    out["pnl_mean"] = pnl_group.mean().reindex(labels)
    out["pnl_median"] = pnl_group.median().reindex(labels)
    out["win_rate"] = pnl_group.apply(lambda s: (s > 0).mean()).reindex(labels)
    if "_ret" in df.columns and not df["_ret"].isna().all():
        ret_group = grouped["_ret"]
        out["ret_mean"] = ret_group.mean().reindex(labels)
        out["ret_median"] = ret_group.median().reindex(labels)
    return out


def holding_period_histogram(df: pd.DataFrame) -> pd.Series:
    if df.empty or "_hold_days" not in df:
        return pd.Series(dtype=float)
    return df["_hold_days"].clip(lower=0).dropna()


def holding_period_scatter(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty or "_hold_days" not in df or "_ret" not in df:
        return {}
    x = df["_hold_days"].astype(float)
    y = df["_ret"].astype(float)
    mask = (~x.isna()) & (~y.isna())
    if not mask.any():
        return {}
    x_vals = x[mask].values
    y_vals = y[mask].values
    slope = intercept = None
    if x_vals.size > 1:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
    scatter_df = pd.DataFrame({"hold_days": x_vals, "return": y_vals})
    return {"scatter": scatter_df, "line": (slope, intercept)}


def holding_period_box(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "_hold_days" not in df or "_ret" not in df:
        return pd.DataFrame()
    hd = df["_hold_days"].clip(lower=0)
    bins = [-0.1, 1, 2, 3, 5, 10, 30, np.inf]
    labels = ["<1", "1-2", "2-3", "3-5", "5-10", "10-30", ">30"]
    categories = pd.cut(hd, bins=bins, labels=labels)
    data = pd.DataFrame({"hold_bin": categories, "return": df["_ret"]})
    return data.dropna()


def streak_statistics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if df.empty or "_pnl_w" not in df:
        return {}
    s = (df["_pnl_w"] > 0).astype(int).dropna().values
    if s.size == 0:
        return {}
    streaks = []
    cur = s[0]
    run = 1
    for v in s[1:]:
        if v == cur:
            run += 1
        else:
            streaks.append((cur, run))
            cur = v
            run = 1
    streaks.append((cur, run))
    wins = [length for outcome, length in streaks if outcome == 1]
    losses = [length for outcome, length in streaks if outcome == 0]
    return {
        "longest_win_streak": int(max(wins) if wins else 0),
        "longest_loss_streak": int(max(losses) if losses else 0),
        "avg_win_streak": float(np.mean(wins) if wins else 0.0),
        "avg_loss_streak": float(np.mean(losses) if losses else 0.0),
    }


def pnl_distribution(df: pd.DataFrame) -> pd.Series:
    if df.empty or "_pnl_w" not in df:
        return pd.Series(dtype=float)
    return df["_pnl_w"].dropna()


def bootstrap_returns(df: pd.DataFrame, column_map: utils.ColumnMap, B: int = 5000) -> Dict[str, object]:
    if column_map.ret is None or df.empty:
        return {}
    returns = pd.to_numeric(df["_ret"], errors="coerce").dropna()
    if returns.size < 5:
        return {}
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, returns.size, size=(B, returns.size))
    means = returns.to_numpy()[idx].mean(axis=1)
    medians = np.median(returns.to_numpy()[idx], axis=1)
    ci_mean = np.quantile(means, [0.025, 0.975])
    ci_median = np.quantile(medians, [0.025, 0.975])
    return {
        "means": means.tolist(),
        "medians": medians.tolist(),
        "ci_mean": tuple(ci_mean),
        "ci_median": tuple(ci_median),
    }


def tier_summary_table(league: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if league.empty or "tier" not in league.columns:
        return pd.DataFrame()
    return league.groupby("tier").head(top_n)


def runs_summary(df: pd.DataFrame, column_map: utils.ColumnMap) -> pd.DataFrame:
    if column_map.run_id is None or df.empty:
        return pd.DataFrame()
    grp = df.groupby(column_map.run_id)
    agg = grp["_pnl_w"].agg(["count", "sum", "mean"])
    agg = agg.rename(columns={"count": "trades", "sum": "total_pnl", "mean": "avg_pnl"})
    agg["win_rate"] = grp["_pnl_w"].apply(lambda s: (s > 0).mean())

    if "_ret" in df.columns and not df["_ret"].isna().all():
        returns = grp["_ret"].agg(["mean", "std"])
        agg["ret_mean"] = returns["mean"]
        agg["ret_std"] = returns["std"]
        agg["sharpe_like"] = (returns["mean"] / returns["std"]) * np.sqrt(utils.DEFAULT_WINDOWS["trading_year"])
    else:
        agg["ret_mean"] = np.nan
        agg["ret_std"] = np.nan
        agg["sharpe_like"] = np.nan

    agg["max_pnl"] = grp["_pnl_w"].max()
    agg["min_pnl"] = grp["_pnl_w"].min()
    return agg.sort_values("total_pnl", ascending=False)


def timeline_dataframe(df: pd.DataFrame, column_map: utils.ColumnMap) -> pd.DataFrame:
    if column_map.timestamp_open is None or column_map.timestamp_close is None:
        return pd.DataFrame()
    symbol_col = column_map.symbol
    start_col = column_map.timestamp_open
    end_col = column_map.timestamp_close
    side_col = column_map.side
    pnl_col = column_map.pnl if column_map.pnl else "_pnl"
    hold_col = column_map.holding_period if column_map.holding_period else "_hold_days"

    timeline = pd.DataFrame(
        {
            "symbol": df[symbol_col] if symbol_col else df.index,
            "start": df[start_col],
            "end": df[end_col],
            "pnl": df[pnl_col],
            "hold_days": df[hold_col],
        }
    )
    if side_col and side_col in df.columns:
        timeline["side"] = df[side_col]
    else:
        timeline["side"] = timeline["symbol"]
    return timeline


def generate_report(df: pd.DataFrame, column_map: utils.ColumnMap) -> str:
    metrics = kpis(df, column_map)
    table = per_symbol_table(df, column_map).head(10)
    concentration = pnl_concentration(df)
    report_lines = [
        "# Roundtrips Performance Report",
        "",
        "## Overview KPIs",
    ]
    for key, label in [
        ("total_trades", "Total trades"),
        ("total_pnl", "Total P&L"),
        ("avg_pnl", "Average P&L / trade"),
        ("win_rate", "Win rate"),
        ("ann_return", "Annualised return"),
        ("ann_vol", "Annualised volatility"),
        ("sharpe", "Sharpe ratio"),
        ("max_drawdown", "Max drawdown"),
        ("avg_hold_days", "Average holding days"),
    ]:
        value = metrics.get(key)
        if key in {"win_rate"}:
            formatted = utils.fmt_pct(value)
        elif key in {"total_pnl", "avg_pnl", "max_drawdown"}:
            formatted = utils.fmt_currency(value)
        elif key in {"ann_return"}:
            formatted = utils.fmt_pct(value)
        else:
            formatted = f"{value:.4f}" if value is not None and np.isfinite(value) else "—"
        report_lines.append(f"- **{label}:** {formatted}")

    report_lines.extend(
        [
            "",
            "## Concentration Metrics",
            f"- Top 5 trades share: {utils.fmt_pct(concentration.get('top5_share'))}",
            f"- Top 1% trades share: {utils.fmt_pct(concentration.get('top1pct_share'))}",
            f"- Gini(|P&L|): {concentration.get('gini_abs'):.3f}" if concentration.get("gini_abs") is not None else "- Gini(|P&L|): —",
            "",
            "## Top Symbols",
        ]
    )

    if not table.empty:
        report_lines.append(table.to_markdown())
    else:
        report_lines.append("_No symbols after applying filters._")

    return "\n".join(report_lines)

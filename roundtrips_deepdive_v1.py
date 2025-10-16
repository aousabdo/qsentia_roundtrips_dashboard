#!/usr/bin/env python

import argparse
import io, os, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = Path("plots")

def _safe_plot_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    return safe or "plot"


def save_plot(fig, name: str, ext: str = "png") -> None:
    """Persist figure to the plots directory with a normalized filename."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_plot_name(name)
    path = PLOTS_DIR / f"{safe_name}.{ext}"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved plot: {path}")


def _parse_csv_arg():
    """Return tuple of (csv_path, provided_flag) based on CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Roundtrips deepdive analysis",
        add_help=True,
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to the roundtrips CSV file (defaults to roundtrips.csv)",
    )
    args, unknown = parser.parse_known_args()
    # Preserve unknown args for other consumers (e.g., IPython)
    sys.argv = [sys.argv[0], *unknown]
    provided = args.csv_path is not None
    raw_path = args.csv_path or "roundtrips.csv"
    path = Path(raw_path).expanduser()
    return path, provided


ROUNDTRIPS_CSV_PATH, CSV_ARG_PROVIDED = _parse_csv_arg()

# ------------- Utilities -------------
def in_colab():
    try:
        import google.colab
        return True
    except Exception:
        return False

def print_section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def robust_load_roundtrips(upload_first=True, fallback_path="roundtrips.csv"):
    """
    Returns: DataFrame (roundtrips), dict(info/debug)
    Tries multiple delimiter/encoding combos; prints raw preview and chosen parse.
    """
    info = {}
    buf = None
    fname = None

    # Option A: Upload (Colab)
    if upload_first and in_colab():
        try:
            from google.colab import files
            up = files.upload()
            if len(up):
                fname = list(up.keys())[0]
                raw_bytes = up[fname]
                info["uploaded_name"] = fname
                info["uploaded_bytes"] = len(raw_bytes)
                print(f"Uploaded: {fname}, bytes: {len(raw_bytes)}")
                raw_preview = raw_bytes[:600].decode(errors="ignore")
                print("\n--- RAW PREVIEW (~600 bytes) ---\n", raw_preview)
                buf = io.BytesIO(raw_bytes)
        except Exception as e:
            print("Upload not used:", e)

    # Option B: Path
    if buf is None:
        assert os.path.exists(fallback_path), f"File not found: {fallback_path}"
        print(f"Reading from path: {fallback_path}, size(bytes): {os.path.getsize(fallback_path)}")
        with open(fallback_path, "rb") as f:
            raw = f.read()
        raw_preview = raw[:600].decode(errors="ignore")
        print("\n--- RAW PREVIEW (~600 bytes) ---\n", raw_preview)
        buf = io.BytesIO(raw)

    attempts = [
        {"sep": ",", "encoding": None},
        {"sep": ";", "encoding": None},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
    ]

    dfs = []
    errors = []
    for a in attempts:
        try:
            buf.seek(0)
            df = pd.read_csv(buf, sep=a["sep"], encoding=a["encoding"])
            dfs.append((a, df))
        except Exception as e:
            errors.append((a, str(e)))

    if not dfs:
        print("All parse attempts failed. Errors:")
        for a, err in errors:
            print(a, "->", err)
        return pd.DataFrame(), {"errors": errors}

    # Choose the parse with most columns first, then most rows
    best = None
    best_score = (-1, -1)
    for a, df in dfs:
        score = (df.shape[1], df.shape[0])
        if score > best_score:
            best, best_score = (a, df), score

    a, rt = best
    print("\nSelected parse:", a, "Shape:", rt.shape)
    print("Columns:", list(rt.columns))
    info["selected_parse"] = a
    return rt, info

def ensure_hold_days(df):
    # if hold_days is missing or zeroed, try to compute from timestamps
    if "hold_days" in df.columns and not (df["hold_days"].fillna(0)==0).all():
        return df
    entry_col = next((c for c in df.columns if "entry" in c.lower() and any(k in c.lower() for k in ["ts","time","date"])), None)
    exit_col  = next((c for c in df.columns if "exit"  in c.lower() and any(k in c.lower() for k in ["ts","time","date"])), None)
    if entry_col and exit_col:
        df[entry_col] = pd.to_datetime(df[entry_col], errors="coerce")
        df[exit_col]  = pd.to_datetime(df[exit_col], errors="coerce")
        df["hold_days"] = (df[exit_col] - df[entry_col]).dt.total_seconds()/(60*60*24)
    return df

def pick_col(df, candidates):
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None

def lorenz_gini_from_abs(series):
    x = np.asarray(series, dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    x = np.sort(np.abs(x))
    if x.sum() <= 0:
        return None, None
    lorenz = np.cumsum(x) / x.sum()
    lorenz = np.insert(lorenz, 0, 0.0)  # start at 0
    gini = 1 - 2*np.trapz(lorenz, dx=1/(len(lorenz)-1))
    return lorenz, gini

print_section("Load & Diagnose")
upload_first = in_colab() and not CSV_ARG_PROVIDED
roundtrips, dbg = robust_load_roundtrips(upload_first=upload_first, fallback_path=str(ROUNDTRIPS_CSV_PATH))
if roundtrips.empty:
    print("\nDataframe is empty. Try adjusting sep/encoding or check the file export.")
    sys.exit(0)

# Normalize timestamps if present (helps later)
for c in roundtrips.columns:
    cl = c.lower()
    if any(k in cl for k in ["ts","time","date"]):
        try:
            roundtrips[c] = pd.to_datetime(roundtrips[c], errors="ignore")
        except Exception:
            pass

roundtrips = ensure_hold_days(roundtrips)

# Identify key columns
PNL_CANDS = {"pnl","pnl_usd","profit","pl","p&l"}
RET_CANDS = {"ret","ret_pct","return","return_pct","pct_return"}
SYM_CANDS = {"symbol","ticker","asset"}

pnl_col = pick_col(roundtrips, PNL_CANDS) or "pnl_usd" if "pnl_usd" in roundtrips.columns else None
ret_col = pick_col(roundtrips, RET_CANDS)
sym_col = pick_col(roundtrips, SYM_CANDS)
exit_col = next((c for c in roundtrips.columns if "exit" in c.lower() and any(k in c.lower() for k in ["ts","date","time"])), None)

print("\nGuessed columns ->",
      "pnl_col:", pnl_col,
      "| ret_col:", ret_col,
      "| sym_col:", sym_col,
      "| exit_ts:", exit_col)


# ------------- Core Performance -------------
print_section("Core Performance")
summary = {}
if pnl_col in roundtrips.columns:
    s = pd.to_numeric(roundtrips[pnl_col], errors="coerce").dropna()
    summary.update({
        "trades": int(s.shape[0]),
        "total_pnl_usd": float(s.sum()),
        "mean_pnl_usd": float(s.mean()),
        "median_pnl_usd": float(s.median()),
        "win_rate": float((s > 0).mean())
    })
    gp = s[s>0].sum()
    gl = -s[s<0].sum()
    summary["profit_factor"] = float(gp/gl) if gl > 0 else np.nan
    avg_win = s[s>0].mean() if (s>0).any() else np.nan
    avg_loss = -s[s<0].mean() if (s<0).any() else np.nan
    summary["avg_win_usd"] = float(avg_win) if pd.notna(avg_win) else np.nan
    summary["avg_loss_usd"] = float(avg_loss) if pd.notna(avg_loss) else np.nan
    wr = (s>0).mean()
    summary["payoff_ratio"] = float(avg_win/avg_loss) if (pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss>0) else np.nan
    summary["expectancy_usd_per_trade"] = float(wr*avg_win - (1-wr)*avg_loss) if (pd.notna(avg_win) and pd.notna(avg_loss)) else np.nan
else:
    print("No P&L column found; many metrics will be skipped.")

if ret_col in roundtrips.columns:
    r = pd.to_numeric(roundtrips[ret_col], errors="coerce").dropna()
    summary.update({
        "ret_mean_pct": float(r.mean()),
        "ret_median_pct": float(r.median()),
        "ret_std_pct": float(r.std(ddof=1)),
        "ret_skew": float(r.skew()),
        "ret_kurtosis": float(r.kurt())
    })
    VaR95 = float(np.percentile(r, 5))
    ES95  = float(r[r <= VaR95].mean()) if (r <= VaR95).any() else np.nan
    summary["VaR_95_pct"] = VaR95
    summary["ES_95_pct"]  = ES95
else:
    print("No return% column found; tail metrics based on returns will be skipped.")

print(pd.Series(summary, name="value"))


# ------------- P&L Concentration + Lorenz/Gini -------------
print_section("P&L Concentration")
if pnl_col in roundtrips.columns:
    s = pd.to_numeric(roundtrips[pnl_col], errors="coerce").dropna()
    n = len(s); total = s.sum()
    top5_share = (s.nlargest(5).sum()/total) if total != 0 else np.nan
    top1pct_n = max(1, math.floor(0.01*n))
    top1pct_share = (s.nlargest(top1pct_n).sum()/total) if total != 0 else np.nan
    print(pd.Series({
        "top5_pnl_share": top5_share,
        "top1pct_trades": top1pct_n,
        "top1pct_pnl_share": top1pct_share
    }))

    lorenz, gini = lorenz_gini_from_abs(s.values)
    if lorenz is not None:
        xs = np.linspace(0, 1, len(lorenz))
        fig, ax = plt.subplots()
        ax.plot(xs, lorenz, label="Lorenz (|P&L|)")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Equality")
        ax.set_title("Lorenz Curve of Absolute P&L")
        ax.set_xlabel("Fraction of trades")
        ax.set_ylabel("Fraction of total |P&L|")
        ax.legend()
        save_plot(fig, "lorenz_curve_absolute_pnl")
        plt.show()
        print("Gini (|P&L|):", gini)
else:
    print("P&L column missing; skipping concentration and Lorenz/Gini.")


# ------------- Calendar views (Daily/Monthly), Equity & Drawdown -------------
print_section("Calendar & Drawdown")
if (pnl_col in roundtrips.columns) and (exit_col is not None):
    df = roundtrips.copy()
    df[exit_col] = pd.to_datetime(df[exit_col], errors="coerce")
    day = df[exit_col].dt.date
    daily_pnl = df.groupby(day)[pnl_col].sum().sort_index()

    if not daily_pnl.empty:
        print("Worst day:", daily_pnl.idxmin(), float(daily_pnl.min()))
        print("Best  day:", daily_pnl.idxmax(), float(daily_pnl.max()))

        # Create subplots for daily P&L, cumulative P&L, and drawdown
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

        daily_pnl.plot(ax=axes[0], title="Daily P&L (sum of closed roundtrips)")
        axes[0].set_ylabel("P&L (USD)")

        eq = daily_pnl.cumsum()
        rolling_max = eq.cummax()
        dd = eq - rolling_max
        eq.plot(ax=axes[1], title="Cumulative P&L (USD)")
        axes[1].set_ylabel("Cumulative P&L")
        dd.plot(ax=axes[2], title="Drawdown (P&L units)")
        axes[2].set_ylabel("Drawdown")
        axes[2].set_xlabel("Date")

        plt.tight_layout()
        save_plot(fig, "daily_pnl_equity_drawdown")
        plt.show()


        # Monthly & weekday aggregates. Convert index to date metadata for the pivot table.
        daily_df = daily_pnl.to_frame("pnl_usd")
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df["month"] = daily_df.index.to_period("M").astype(str)
        daily_df["weekday"] = daily_df.index.day_name()

        pivot = daily_df.pivot_table(
            values="pnl_usd",
            index="month",
            columns="weekday",
            aggfunc="sum",
            fill_value=0.0,
        )

        if pivot.empty:
            print("\nMonthly x Weekday P&L (sum): <empty>")
        else:
            print("\nMonthly x Weekday P&L (sum):")
            print(pivot)

            fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column

            # --- Top plot: Monthly P&L (sum)
            pivot.sum(axis=1).plot(
                kind="bar",
                ax=axes[0],
                title="Monthly P&L (sum)"
            )
            axes[0].set_xlabel("Month")
            axes[0].set_ylabel("P&L (USD)")
            axes[0].tick_params(axis='x', rotation=45)

            # --- Bottom plot: Weekday P&L (sum)
            wd_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            week_series = pivot.sum(axis=0).reindex([w for w in wd_order if w in pivot.columns])
            week_series.plot(
                kind="bar",
                ax=axes[1],
                title="Weekday P&L (sum)"
            )
            axes[1].set_xlabel("Weekday")
            axes[1].set_ylabel("P&L (USD)")
            axes[1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            save_plot(fig, "monthly_weekday_pnl_barcharts")
            plt.show()

            # Add heatmap of monthly and weekday P&L
            fig = plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".0f",
                cmap="viridis",
                cbar_kws={"label": "P&L (USD)"}
            )

            plt.title("Monthly vs. Weekday P&L Heatmap", fontsize=14, pad=15)
            plt.xlabel("Weekday", fontsize=12, labelpad=10)
            plt.ylabel("Month", fontsize=12, labelpad=10)

            # --- Fix overlapping labels ---
            plt.yticks(rotation=0)   # keep month labels horizontal
            plt.xticks(rotation=45, ha='right')  # tilt weekday labels for clarity
            plt.tight_layout()
            save_plot(fig, "monthly_weekday_pnl_heatmap")
            plt.show()

    else:
        print("Daily P&L aggregation is empty.")
else:
    print("Missing P&L or exit timestamp; skipping calendar & drawdown.")



# ------------- Holding Period Analytics -------------
# Assumes you already have:
# - roundtrips: DataFrame
# - pnl_col: str (name of P&L column)
# - ret_col: str (name of return % column)
# - print_section: function to print a section header
# Also assumes matplotlib.pyplot as plt, numpy as np, pandas as pd are imported.

print_section("Holding Period Analytics")

if "hold_days" in roundtrips.columns:
    # --- Prep & Histogram ---
    hd = pd.to_numeric(roundtrips["hold_days"], errors="coerce").clip(lower=0)
    ax = hd.hist(bins=30)
    ax.set_title("Holding Period (days)")
    ax.set_xlabel("days")
    ax.set_ylabel("frequency")
    save_plot(ax.get_figure(), "holding_period_days_histogram")
    plt.show()

    # --- Binning for summaries ---
    bins = [-0.1, 1, 2, 3, 5, 10, 30, np.inf]
    labels = ["<1","1-2","2-3","3-5","5-10","10-30",">30"]
    hp_bin = pd.cut(hd, bins=bins, labels=labels)

    out = pd.DataFrame(index=pd.Index(labels, name="hp_bin"))

    if 'pnl_col' in locals() and pnl_col in roundtrips.columns:
        s = pd.to_numeric(roundtrips[pnl_col], errors="coerce")
        out["pnl_mean"]   = s.groupby(hp_bin, observed=True).mean()
        out["pnl_median"] = s.groupby(hp_bin, observed=True).median()
        out["win_rate"]   = s.groupby(hp_bin, observed=True).apply(lambda x: (x > 0).mean())

    if 'ret_col' in locals() and ret_col in roundtrips.columns:
        r = pd.to_numeric(roundtrips[ret_col], errors="coerce")
        out["ret_mean"]   = r.groupby(hp_bin, observed=True).mean()
        out["ret_median"] = r.groupby(hp_bin, observed=True).median()

    print(out)

    # --- Plots: Scatter (top) + Boxplot (bottom) in ONE figure ---
    if 'ret_col' in locals() and ret_col in roundtrips.columns:
        # Prepare data for plotting
        x = hd.values
        y = pd.to_numeric(roundtrips[ret_col], errors="coerce").values
        mask = (~np.isnan(x)) & (~np.isnan(y))

        df_plot = pd.DataFrame({
            ret_col: pd.to_numeric(roundtrips[ret_col], errors="coerce"),
            "hold_days_bin": hp_bin
        }).dropna()

        # One figure with two rows
        fig, (ax_scatter, ax_box) = plt.subplots(
            nrows=2, ncols=1, figsize=(12, 10), constrained_layout=True
        )

        # Row 1: scatter with optional trend line
        ax_scatter.scatter(x[mask], y[mask], alpha=0.3)
        ax_scatter.set_title("Return % vs. Holding Days")
        ax_scatter.set_xlabel("hold_days")
        ax_scatter.set_ylabel(ret_col)

        if mask.sum() > 1:
            coef = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax_scatter.plot(xs, coef[0] * xs + coef[1], linewidth=2)

        # Row 2: boxplot (drawn on the same figure via ax=...)
        if not df_plot.empty:
            df_plot.boxplot(column=ret_col, by="hold_days_bin", ax=ax_box)
            ax_box.set_title("Return % by Holding Period Bin")
            ax_box.set_xlabel("Holding Period Bin")
            ax_box.set_ylabel(ret_col)
            fig.suptitle("")  # suppress pandas' default supertitle
            for tick in ax_box.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")
        else:
            ax_box.set_visible(False)

        save_plot(fig, "holding_period_return_relationship")
        plt.show()
    else:
        print("No data to plot for Return % vs. Holding Days.")
else:
    print("No 'hold_days' field available (and could not infer from timestamps).")


# ------------- Streaks -------------
print_section("Streaks")
if pnl_col in roundtrips.columns:
    s = (pd.to_numeric(roundtrips[pnl_col], errors="coerce") > 0).astype(int).dropna().values
    if s.size:
        streaks = []
        cur = s[0]; run = 1
        for v in s[1:]:
            if v == cur: run += 1
            else:
                streaks.append((cur, run))
                cur, run = v, 1
        streaks.append((cur, run))
        wins = [r for v,r in streaks if v==1]
        losses = [r for v,r in streaks if v==0]
        res = {
            "longest_win_streak": int(max(wins) if wins else 0),
            "longest_loss_streak": int(max(losses) if losses else 0),
            "avg_win_streak": float(np.mean(wins) if wins else 0.0),
            "avg_loss_streak": float(np.mean(losses) if losses else 0.0),
        }
        print(pd.Series(res))
    else:
        print("No P&L observations for streaks.")
else:
    print("No P&L column; skipping streaks.")


# ------------- Per-Symbol League Table -------------
print_section("Per-Symbol League Table")
if sym_col in roundtrips.columns:
    grp = roundtrips.groupby(sym_col)
    cols = {}
    if pnl_col in roundtrips.columns:
        cols["trades"] = grp[pnl_col].count()
        cols["win_rate"] = grp[pnl_col].apply(lambda s: (pd.to_numeric(s, errors="coerce")>0).mean())
        cols["total_pnl"] = grp[pnl_col].sum()
        cols["avg_pnl"] = grp[pnl_col].mean()
    if ret_col in roundtrips.columns:
        cols["ret_mean"] = grp[ret_col].mean()
        cols["ret_std"] = grp[ret_col].std(ddof=1)
        cols["ret_p5"] = grp[ret_col].quantile(0.05)
    if len(cols)>0:
        league = pd.concat(cols, axis=1)
        if "total_pnl" in league.columns:
            league = league.sort_values("total_pnl", ascending=False)
        print(league.head(25))
    else:
        print("No suitable columns to aggregate by symbol.")
else:
    print("No symbol/ticker column.")


# ------------- Bootstrap CIs for return -------------
print_section("Bootstrap CIs for Return%")
if ret_col in roundtrips.columns:
    r = pd.to_numeric(roundtrips[ret_col], errors="coerce").dropna().values
    if r.size >= 5:
        B = 5000
        rng = np.random.default_rng(42)
        means = np.empty(B); medians = np.empty(B)
        n = len(r)
        for i in range(B):
            sample = rng.choice(r, size=n, replace=True)
            means[i] = sample.mean()
            medians[i] = np.median(sample)
        ci_mean = (np.percentile(means, 2.5), np.percentile(means, 97.5))
        ci_median = (np.percentile(medians, 2.5), np.percentile(medians, 97.5))
        print("Mean return CI (95%):", ci_mean)
        print("Median return CI (95%):", ci_median)
    else:
        print("Not enough return observations for bootstrap.")
else:
    print("No return% column for bootstrap.")


# ------------- Core Performance -------------
print_section("Core Performance")
summary = {}
if pnl_col in roundtrips.columns:
    s = pd.to_numeric(roundtrips[pnl_col], errors="coerce").dropna()

    # Add histogram of P&L distribution
    ax = s.hist(bins=50, figsize=(10, 6))
    ax.set_title("Distribution of P&L (USD)")
    ax.set_xlabel("P&L (USD)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    save_plot(ax.get_figure(), "pnl_distribution_histogram")
    plt.show()

    summary.update({
        "trades": int(s.shape[0]),
        "total_pnl_usd": float(s.sum()),
        "mean_pnl_usd": float(s.mean()),
        "median_pnl_usd": float(s.median()),
        "win_rate": float((s > 0).mean())
    })
    gp = s[s>0].sum()
    gl = -s[s<0].sum()
    summary["profit_factor"] = float(gp/gl) if gl > 0 else np.nan
    avg_win = s[s>0].mean() if (s>0).any() else np.nan
    avg_loss = -s[s<0].mean() if (s<0).any() else np.nan
    summary["avg_win_usd"] = float(avg_win) if pd.notna(avg_win) else np.nan
    summary["avg_loss_usd"] = float(avg_loss) if pd.notna(avg_loss) else np.nan
    wr = (s>0).mean()
    summary["payoff_ratio"] = float(avg_win/avg_loss) if (pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss>0) else np.nan
    summary["expectancy_usd_per_trade"] = float(wr*avg_win - (1-wr)*avg_loss) if (pd.notna(avg_win) and pd.notna(avg_loss)) else np.nan
else:
    print("No P&L column found; many metrics will be skipped.")

if ret_col in roundtrips.columns:
    r = pd.to_numeric(roundtrips[ret_col], errors="coerce").dropna()
    summary.update({
        "ret_mean_pct": float(r.mean()),
        "ret_median_pct": float(r.median()),
        "ret_std_pct": float(r.std(ddof=1)),
        "ret_skew": float(r.skew()),
        "ret_kurtosis": float(r.kurt())
    })
    VaR95 = float(np.percentile(r, 5))
    ES95  = float(r[r <= VaR95].mean()) if (r <= VaR95).any() else np.nan
    summary["VaR_95_pct"] = VaR95
    summary["ES_95_pct"]  = ES95
else:
    print("No return% column found; tail metrics based on returns will be skipped.")

print(pd.Series(summary, name="value"))


# ------------- Per-Symbol League Table -------------
print_section("Per-Symbol League Table")
if sym_col in roundtrips.columns:
    grp = roundtrips.groupby(sym_col)
    cols = {}
    if pnl_col in roundtrips.columns:
        cols["trades"] = grp[pnl_col].count()
        cols["win_rate"] = grp[pnl_col].apply(lambda s: (pd.to_numeric(s, errors="coerce")>0).mean())
        cols["total_pnl"] = grp[pnl_col].sum()
        cols["avg_pnl"] = grp[pnl_col].mean()
    if ret_col in roundtrips.columns:
        cols["ret_mean"] = grp[ret_col].mean()
        cols["ret_std"] = grp[ret_col].std(ddof=1)
        cols["ret_p5"] = grp[ret_col].quantile(0.05)
    if len(cols)>0:
        league = pd.concat(cols, axis=1)
        if "total_pnl" in league.columns:
            league = league.sort_values("total_pnl", ascending=False)
            print(league.head(25))

            # Add bar plot of top 10 symbols by total P&L
            fig, ax = plt.subplots(figsize=(12, 6))
            league.head(10)["total_pnl"].plot(kind="bar", ax=ax)
            ax.set_title("Top 10 Symbols by Total P&L")
            ax.set_xlabel("Symbol")
            ax.set_ylabel("Total P&L (USD)")
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")
            plt.tight_layout()
            save_plot(fig, "top10_symbols_total_pnl")
            plt.show()

            # Add scatter plot of Average P&L vs. Win Rate
            if "avg_pnl" in league.columns and "win_rate" in league.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(league["win_rate"], league["avg_pnl"])
                ax.set_title("Average P&L vs. Win Rate per Symbol")
                ax.set_xlabel("Win Rate")
                ax.set_ylabel("Average P&L (USD)")
                ax.grid(True)
                # Add symbol labels for some points (optional, can be noisy)
                # for i, row in league.head(25).iterrows():
                #     ax.annotate(i, (row["win_rate"], row["avg_pnl"]), textcoords="offset points", xytext=(0,10), ha='center')
                save_plot(fig, "avg_pnl_vs_win_rate_scatter")
                plt.show()


    else:
        print("No suitable columns to aggregate by symbol.")
else:
    print("No symbol/ticker column.")


if ret_col in roundtrips.columns and r.size >= 5:
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column

    # --- Top: Mean Return Distribution ---
    axes[0].hist(means, bins=50, density=True, alpha=0.6, color='g')
    axes[0].axvline(ci_mean[0], color='r', linestyle='dashed', linewidth=1, label=f'95% CI Lower: {ci_mean[0]:.4f}')
    axes[0].axvline(ci_mean[1], color='r', linestyle='dashed', linewidth=1, label=f'95% CI Upper: {ci_mean[1]:.4f}')
    axes[0].set_title("Bootstrap Distribution of Mean Return (%)")
    axes[0].set_xlabel("Mean Return (%)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True)

    # --- Bottom: Median Return Distribution ---
    axes[1].hist(medians, bins=50, density=True, alpha=0.6, color='b')
    axes[1].axvline(ci_median[0], color='r', linestyle='dashed', linewidth=1, label=f'95% CI Lower: {ci_median[0]:.4f}')
    axes[1].axvline(ci_median[1], color='r', linestyle='dashed', linewidth=1, label=f'95% CI Upper: {ci_median[1]:.4f}')
    axes[1].set_title("Bootstrap Distribution of Median Return (%)")
    axes[1].set_xlabel("Median Return (%)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_plot(fig, "bootstrap_return_distributions")
    plt.show()


import io, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---------- Parameters (edit here) ----------
WIN_MIN_T1 = 0.65             # hit-rate threshold for Tier 1
WIN_MIN_T2 = 0.55             # hit-rate lower bound for Tier 2
MILD_TAIL_P5 = -4.0           # "mild tails" cutoff for 5th pct return (%)  (e.g., > -4%)
MIN_TRADES_FOR_TRUST = 20     # minimum N to scale size
BOOTSTRAP_B = 3000            # bootstraps for avg P&L CI (per symbol)
RANDOM_SEED = 42

# Suggested sizing multipliers (relative to base)
SIZE_MULT_T1  = 1.25
SIZE_MULT_T2  = 0.75
SIZE_MULT_T3  = 0.50
SIZE_MULT_OUT = 0.25          # Outlier-Hold (small N)

# Optional hard caps (e.g., fraction of portfolio risk)
CAP_T1 = 1.00
CAP_T2 = 0.60
CAP_T3 = 0.30
CAP_OUT= 0.20

# ---------- Load CSV (Colab upload or path) ----------
if in_colab() and not CSV_ARG_PROVIDED:
    try:
        from google.colab import files
        up = files.upload()
        assert len(up) > 0, "No file uploaded"
        fname = list(up.keys())[0]
        roundtrips = pd.read_csv(io.BytesIO(up[fname]))
    except Exception:
        # fallback to provided/default path if upload not used
        roundtrips = pd.read_csv(ROUNDTRIPS_CSV_PATH)
else:
    # fallback to provided/default path when not uploading
    roundtrips = pd.read_csv(ROUNDTRIPS_CSV_PATH)

# Parse timestamps if present (not required here)
for c in roundtrips.columns:
    if any(k in c.lower() for k in ["ts","time","date"]):
        try:
            roundtrips[c] = pd.to_datetime(roundtrips[c], errors="ignore")
        except Exception:
            pass

# Identify columns
def pick(df, cands):
    for c in df.columns:
        if c.lower() in cands:
            return c
    return None

SYM  = pick(roundtrips, {"symbol","ticker","asset"})
PNL  = pick(roundtrips, {"pnl","pnl_usd","profit","pl","p&l"})
RET  = pick(roundtrips, {"ret","ret_pct","return","return_pct","pct_return"})

assert SYM is not None, "Need a symbol/ticker column"
assert PNL is not None, "Need a P&L column (e.g., pnl_usd)"
# RET is optional but recommended for ret_p5 (tail risk)

roundtrips[PNL] = pd.to_numeric(roundtrips[PNL], errors="coerce")
if RET is not None:
    roundtrips[RET] = pd.to_numeric(roundtrips[RET], errors="coerce")

# ---------- Per-symbol metrics ----------
grp = roundtrips.groupby(SYM)

# Basic stats
trades = grp[PNL].count().rename("trades")
win_rate = grp[PNL].apply(lambda s: (s>0).mean()).rename("win_rate")
avg_pnl = grp[PNL].mean().rename("avg_pnl")
total_pnl = grp[PNL].sum().rename("total_pnl")

# Average win/loss for expectancy
avg_win = grp[PNL].apply(lambda s: s[s>0].mean() if (s>0).any() else np.nan).rename("avg_win")
avg_loss= grp[PNL].apply(lambda s: -s[s<0].mean() if (s<0).any() else np.nan).rename("avg_loss")
expectancy = (win_rate*avg_win - (1 - win_rate)*avg_loss).rename("expectancy_usd")

# Return stats (optional)
if RET is not None:
    ret_mean = grp[RET].mean().rename("ret_mean")
    ret_std  = grp[RET].std(ddof=1).rename("ret_std")
    ret_p5   = grp[RET].quantile(0.05).rename("ret_p5")  # tail risk
else:
    ret_mean = pd.Series(index=trades.index, dtype=float, name="ret_mean")
    ret_std  = pd.Series(index=trades.index, dtype=float, name="ret_std")
    ret_p5   = pd.Series(index=trades.index, dtype=float, name="ret_p5")

league = pd.concat([trades, win_rate, total_pnl, avg_pnl, expectancy, ret_mean, ret_std, ret_p5], axis=1)

# ---------- Confidence intervals ----------
# Wilson CI for win rate; bootstrap CI for avg_pnl (both per symbol)
from math import sqrt

def wilson_ci(p, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z*sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return (center - half, center + half)

rng = np.random.default_rng(RANDOM_SEED)

def bootstrap_mean_ci(x, B=BOOTSTRAP_B, alpha=0.05):
    x = np.asarray(pd.to_numeric(x, errors="coerce").dropna())
    if x.size < 2: return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha/2, 1 - alpha/2])
    return (float(lo), float(hi))

wr_lo, wr_hi, pnl_lo, pnl_hi = [], [], [], []
for sym, df in roundtrips.groupby(SYM):
    n = len(df)
    p = (df[PNL] > 0).mean()
    lo, hi = wilson_ci(p, n)
    wr_lo.append(lo); wr_hi.append(hi)
    lo2, hi2 = bootstrap_mean_ci(df[PNL])
    pnl_lo.append(lo2); pnl_hi.append(hi2)

league["win_ci_lo"] = wr_lo
league["win_ci_hi"] = wr_hi
league["avg_pnl_ci_lo"] = pnl_lo
league["avg_pnl_ci_hi"] = pnl_hi

# ---------- Tier assignment ----------
def assign_tier(row):
    n = row["trades"]
    wr = row["win_rate"]
    expct = row["expectancy_usd"]
    tail = row["ret_p5"]
    # small-N outliers (including 100% win) -> Outlier-Hold
    if n < MIN_TRADES_FOR_TRUST:
        return "Outlier-Hold"
    # Tier 3 if low win OR bad tail OR negative expectancy with bad tail
    if (wr < WIN_MIN_T2) or (pd.notna(tail) and tail < MILD_TAIL_P5) or (pd.notna(expct) and expct <= 0 and (pd.isna(tail) or tail < -3.5)):
        return "Tier 3"
    # Tier 1: high win, positive expectancy, mild tails
    if (wr >= WIN_MIN_T1) and (pd.notna(expct) and expct > 0) and (pd.isna(tail) or tail >= MILD_TAIL_P5):
        return "Tier 1"
    # Otherwise Tier 2
    return "Tier 2"

league["tier"] = league.apply(assign_tier, axis=1)

# Suggested sizing/caps
def size_mult(row):
    t = row["tier"]
    if t == "Tier 1": return SIZE_MULT_T1
    if t == "Tier 2": return SIZE_MULT_T2
    if t == "Tier 3": return SIZE_MULT_T3
    return SIZE_MULT_OUT

def size_cap(row):
    t = row["tier"]
    if t == "Tier 1": return CAP_T1
    if t == "Tier 2": return CAP_T2
    if t == "Tier 3": return CAP_T3
    return CAP_OUT

league["size_mult"] = league.apply(size_mult, axis=1)
league["size_cap"]  = league.apply(size_cap, axis=1)

# Order by tier and total P&L
tier_order = {"Tier 1":0, "Tier 2":1, "Tier 3":2, "Outlier-Hold":3}
league = league.sort_values(by=["tier", "total_pnl"], key=lambda s: s.map(lambda x: tier_order.get(x, 4)) if s.name=="tier" else s, ascending=[True, False])

print("\n=== Tier Summary (counts) ===")
print(league["tier"].value_counts())

print("\n=== Sample (top 25 by total P&L within tier) ===")
print(league.groupby("tier").head(25))


# ---------- Bubble plot: win rate vs avg P&L ----------
fig, ax = plt.subplots(figsize=(9, 6))
x = league["win_rate"]
y = league["avg_pnl"]
sizes = 50 + 5 * league["trades"]                  # bubble size ~ trade count
# Color by tail risk (ret_p5): red = worse tails
cmap = plt.cm.get_cmap("coolwarm")
# map ret_p5; if NaN, use mid color
ret_p5_norm = league["ret_p5"].fillna(0.0)
mn, mx = -8.0, 2.0
colors = cmap(np.clip((ret_p5_norm - mn) / (mx - mn), 0, 1))
ax.scatter(x, y, s=sizes, c=colors, alpha=0.8, edgecolor="k", linewidth=0.5)

# Threshold lines
ax.axvline(WIN_MIN_T1, color="green", linestyle="--", label=f"Tier1 win≥{WIN_MIN_T1:.2f}")
ax.axvline(WIN_MIN_T2, color="orange", linestyle="--", label=f"Tier2 win≥{WIN_MIN_T2:.2f}")
ax.axhline(0, color="gray", linewidth=1)

ax.set_title("Win Rate vs Avg P&L by Symbol\n(bubble=size=trades, color=tail risk ret_p5)")
ax.set_xlabel("Win Rate")
ax.set_ylabel("Average P&L (USD)")
ax.legend(loc="lower right")
fig.tight_layout()
save_plot(fig, "win_rate_vs_avg_pnl_bubble")
plt.show()

# Annotate a few key tickers (top by total P&L and worst tail)
annot_syms = list(league.sort_values("total_pnl", ascending=False).head(8).index)
annot_syms += list(league.sort_values("ret_p5").head(5).index)
annot_syms = list(dict.fromkeys(annot_syms))  # unique preserve order

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x, y, s=sizes, c=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
for sym in annot_syms:
    ax.annotate(sym, (league.loc[sym, "win_rate"], league.loc[sym, "avg_pnl"]),
                textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.axvline(WIN_MIN_T1, color="green", linestyle="--")
ax.axvline(WIN_MIN_T2, color="orange", linestyle="--")
ax.axhline(0, color="gray", linewidth=1)
ax.set_title("Annotated: key contributors & worst tails")
ax.set_xlabel("Win Rate")
ax.set_ylabel("Average P&L (USD)")
fig.tight_layout()
save_plot(fig, "win_rate_vs_avg_pnl_annotated")
plt.show()

# ---------- Export policy ----------
policy = league[[
    "trades","win_rate","win_ci_lo","win_ci_hi",
    "avg_pnl","avg_pnl_ci_lo","avg_pnl_ci_hi",
    "expectancy_usd","ret_mean","ret_std","ret_p5",
    "tier","size_mult","size_cap"
]]
policy.to_csv("symbol_tiers_policy.csv")
print("\nSaved: symbol_tiers_policy.csv")


# Create a single figure with a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# List of columns to plot and their titles
plot_cols = [('trades', 'Trades by Tier'),
             ('win_rate', 'Win Rate by Tier'),
             ('total_pnl', 'Total P&L by Tier'),
             ('avg_pnl', 'Average P&L by Tier')]

# Plot each violin plot in a separate subplot
for i, (col, title) in enumerate(plot_cols):
    sns.violinplot(data=league, x=col, y='tier', inner='stick', palette='Dark2', ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Tier')
sns.despine(top=True, right=True, bottom=True, left=True, ax=axes[i])

plt.tight_layout() # Adjust layout to prevent overlap
save_plot(fig, "tier_metric_violins")
plt.show()

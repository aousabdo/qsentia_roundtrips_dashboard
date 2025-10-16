# QSentia Roundtrips Dashboard

Interactive Streamlit dashboard for analysing roundtrip / trade-level performance with reusable analytics extracted from `BT_analysis/roundtrips_deepdive_v1.py`.

## Features
- File uploader with CSV/Parquet support, bundled sample dataset, schema validation, and caching.
- Sidebar controls for date ranges, symbols, strategies, runs, minimum trades, holding-period range, P&L clipping, optional winsorisation, and rolling-window selection.
- Six analytical tabs: overview KPIs, per-symbol league table, trade explorer with timeline, distribution plots, risk & drawdowns, and backtest run comparisons.
- Plotly-based interactive charts with PNG export, filtered table downloads, and Markdown report generation.
- Session-state persistence plus reset and apply filter actions.

## Project Structure
```
qsentia_roundtrips_dashboard/
├─ app.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ sample_roundtrips.csv
├─ qsentia/
│  ├─ __init__.py
│  ├─ io.py
│  ├─ compute.py
│  ├─ viz.py
│  └─ utils.py
└─ assets/
   └─ styles.css
```

## Installation & Run
```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Using the App
1. Launch with `streamlit run app.py`.
2. By default the dashboard loads `data/roundtrips.csv` (or `data/roundtrip.csv` if present); drop in your own file with either name or pick **Sample dataset** / **Upload file** from the sidebar.
3. Adjust sidebar filters and click **Apply filters**. Use **Reset filters** to revert to defaults.
4. Navigate tabs for KPIs, symbol league tables, trade explorer, distributions, risk metrics, and backtest runs.
5. Download filtered tables, PNG charts, or generate a Markdown report capturing current KPIs and filter state.

Expected canonical columns (matched case-insensitively):  
`timestamp_open`, `timestamp_close`, `symbol`, `side`, `qty`, `price_open`, `price_close`, `pnl`, `return`, `holding_period`, `strategy`, `run_id`.  
The loader auto-detects synonyms such as `pnl_usd`, `ret_pct`, `entry_ts`, `exit_ts`, etc., mirroring the logic in the legacy script.

## Mapping from `roundtrips_deepdive_v1.py`

| Legacy functionality | New module/function |
| --- | --- |
| `robust_load_roundtrips`, timestamp coercion | `qsentia.io.load_csv_or_parquet`, `qsentia.io.coerce_dtypes` |
| `ensure_hold_days` | `qsentia.compute.prepare_dataframe` |
| Core P&L & return summaries | `qsentia.compute.kpis` |
| Lorenz / Gini concentration | `qsentia.compute.pnl_concentration` |
| Daily equity, drawdown plots | `qsentia.compute.equity_curve`, `qsentia.compute.drawdowns`, `qsentia.viz.fig_equity_curve` / `fig_underwater` |
| Holding period histogram & bins | `qsentia.compute.holding_period_summary` |
| Per-symbol league table + CIs + tiering | `qsentia.compute.per_symbol_table` |
| Bubble/scatter visualisations | `qsentia.viz.fig_per_symbol_bar`, `fig_violin_returns`, `fig_box_returns` |
| Bootstrap distributions | Embedded in `qsentia.compute.per_symbol_table` (mean CI) and distribution tab |
| Run-level comparisons | `qsentia.compute.runs_summary`, `qsentia.viz.fig_runs_compare` |

## Notes
- Plot downloads require `kaleido`; a helpful hint is shown if the extension is unavailable.
- Risk metrics annualise daily grouped returns when a return column is present; otherwise Sharpe-like ratios fall back to P&L variability.
- Extend styling via `assets/styles.css`. Streamlit remembers sidebar state across reruns.

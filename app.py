from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import streamlit as st

from qsentia import compute, io as qio, utils, viz


st.set_page_config(
    page_title="QSentia Roundtrips Dashboard",
    page_icon="📈",
    layout="wide",
)


def _load_styles() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with css_path.open("r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset(
    source_type: str,
    file_bytes: Optional[bytes],
    file_name: Optional[str],
    default_path: Optional[str],
):
    if source_type == "sample":
        df = qio.load_sample()
    elif source_type == "default":
        if default_path is None:
            raise FileNotFoundError("Default dataset not available.")
        df = qio.load_default(default_path)
    elif source_type == "upload":
        if file_bytes is None or file_name is None:
            raise ValueError("No file provided for upload.")
        buffer = io.BytesIO(file_bytes)
        buffer.name = file_name
        df = qio.load_csv_or_parquet(buffer)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    is_valid, column_map, report = qio.validate_schema(df)
    df_prepared, column_map = compute.prepare_dataframe(df, column_map)
    return df_prepared, column_map, report, is_valid


def make_default_filters(df: pd.DataFrame, column_map: utils.ColumnMap) -> utils.FilterParams:
    start = None
    end = None
    if column_map.timestamp_close and not df[column_map.timestamp_close].isna().all():
        start = df[column_map.timestamp_close].min()
        end = df[column_map.timestamp_close].max()

    hold_range: Optional[tuple[Optional[float], Optional[float]]] = None
    if column_map.holding_period and column_map.holding_period in df.columns:
        series = pd.to_numeric(df[column_map.holding_period], errors="coerce")
        if not series.dropna().empty:
            hold_range = (float(series.min()), float(series.max()))

    pnl_clip: Optional[tuple[Optional[float], Optional[float]]] = None
    if column_map.pnl and column_map.pnl in df.columns:
        pnl_series = pd.to_numeric(df[column_map.pnl], errors="coerce")
        if not pnl_series.dropna().empty:
            pnl_clip = (float(pnl_series.min()), float(pnl_series.max()))

    return utils.FilterParams(
        start=start,
        end=end,
        symbols=None,
        strategies=None,
        runs=None,
        min_trades_per_symbol=0,
        hold_days_range=hold_range,
        pnl_clip=pnl_clip,
        winsor_pct=None,
    )


def render_kpi_cards(metrics: Dict[str, Optional[float]]) -> None:
    print(metrics)
    kpis = [
        ("Total Trades", metrics.get("total_trades"), lambda v: f"{int(v):,}" if v is not None else "—"),
        ("Win Rate", metrics.get("win_rate"), utils.fmt_pct),
        ("Total P&L", metrics.get("total_pnl"), utils.fmt_currency),
        ("Avg P&L / Trade", metrics.get("avg_pnl"), utils.fmt_currency),
        ("Annualised Return", metrics.get("ann_return"), utils.fmt_pct),
        ("Annualised Vol", metrics.get("ann_vol"), utils.fmt_pct),
        ("Sharpe", metrics.get("sharpe"), lambda v: f"{v:.2f}" if v is not None and pd.notna(v) else "—"),
        ("Max Drawdown", metrics.get("max_drawdown"), utils.fmt_currency),
        ("Avg Hold (days)", metrics.get("avg_hold_days"), lambda v: f"{v:.2f}" if v is not None and pd.notna(v) else "—"),
    ]
    cols = st.columns(3)
    for idx, (label, value, formatter) in enumerate(kpis):
        col = cols[idx % 3]
        display = formatter(value) if callable(formatter) else value
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <h3>{label}</h3>
                    <div class="value">{display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def download_plotly_figure(fig, label: str, key: str) -> None:
    if fig is None:
        return
    try:
        data = fig.to_image(format="png")
        st.download_button(
            label=label,
            data=data,
            file_name=f"{key}.png",
            mime="image/png",
            width="content",
        )
    except Exception:
        st.info("Install kaleido to enable PNG exports: pip install kaleido")


def filter_dataframe(df: pd.DataFrame, column_map: utils.ColumnMap, active_filters: utils.FilterParams) -> pd.DataFrame:
    return compute.apply_filters(df, active_filters, column_map)


def main() -> None:
    _load_styles()
    st.title("QSentia Roundtrips Dashboard")
    st.caption("Interactive performance analytics for roundtrip trades with multi-tab insights.")

    data_dir = Path(__file__).parent / "data"
    default_data_path: Optional[Path] = None
    for candidate in ("roundtrips.csv", "roundtrip.csv"):
        candidate_path = data_dir / candidate
        if candidate_path.exists():
            default_data_path = candidate_path.resolve()
            break

    with st.sidebar:
        st.header("Data")
        data_options = []
        if default_data_path is not None:
            data_options.append(("Default dataset", "default"))
        data_options.append(("Sample dataset", "sample"))
        data_options.append(("Upload file", "upload"))

        default_index = 0
        if default_data_path is None:
            default_index = 0  # sample will be first

        option_labels = [label for label, _ in data_options]
        label_to_key = {label: key for label, key in data_options}

        choice_label = st.radio("Source", option_labels, index=default_index)
        data_source = label_to_key[choice_label]

        uploaded = None
        if data_source == "upload":
            uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])

    if data_source == "upload" and uploaded is None:
        st.warning("Upload a CSV/Parquet file to continue.")
        st.stop()

    file_bytes = uploaded.getvalue() if uploaded else None
    file_name = uploaded.name if uploaded else None
    default_path_str = str(default_data_path) if default_data_path is not None else None

    try:
        df, column_map, schema_report, is_valid = load_dataset(
            data_source,
            file_bytes,
            file_name,
            default_path_str,
        )
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    if not is_valid:
        st.error("Schema validation failed. Please review the detected columns below.")
        st.dataframe(schema_report, width="stretch")
        st.stop()

    if "filters_state" not in st.session_state:
        st.session_state.filters_state = make_default_filters(df, column_map)
    if "start_capital" not in st.session_state:
        st.session_state.start_capital = compute.DEFAULT_START_CAPITAL

    with st.sidebar:
        st.header("Filters")
        st.subheader("Equity Settings")
        start_capital_input = st.number_input(
            "Start capital",
            min_value=0.0,
            value=float(st.session_state.start_capital),
            step=10_000.0,
            format="%.2f",
        )
        st.session_state.start_capital = start_capital_input
        start_capital = st.session_state.start_capital
        defaults = make_default_filters(df, column_map)
        current = st.session_state.filters_state

        with st.form("filters_form"):
            date_default = (current.start.date() if current.start else defaults.start.date() if defaults.start else pd.Timestamp.now().date(),
                            current.end.date() if current.end else defaults.end.date() if defaults.end else pd.Timestamp.now().date())
            if column_map.timestamp_close:
                date_range = st.date_input(
                    "Close date range",
                    value=date_default,
                    help="Trades closing outside this range are excluded.",
                )
            else:
                date_range = None

            symbols = sorted(df[column_map.symbol].dropna().unique()) if column_map.symbol else []
            selected_symbols = st.multiselect(
                "Symbols",
                options=symbols,
                default=current.symbols or symbols,
            ) if symbols else []

            strategies = sorted(df[column_map.strategy].dropna().unique()) if column_map.strategy else []
            selected_strategies = st.multiselect(
                "Strategies",
                options=strategies,
                default=current.strategies or strategies,
            ) if strategies else []

            runs = sorted(df[column_map.run_id].dropna().unique()) if column_map.run_id else []
            selected_runs = st.multiselect(
                "Runs",
                options=runs,
                default=current.runs or runs,
            ) if runs else []

            min_trades = st.number_input(
                "Min trades per symbol",
                min_value=0,
                value=current.min_trades_per_symbol or 0,
                step=1,
            )

            if column_map.holding_period and column_map.holding_period in df.columns:
                hold_series = pd.to_numeric(df[column_map.holding_period], errors="coerce").dropna()
                if not hold_series.empty:
                    hold_min = float(hold_series.min())
                    hold_max = float(hold_series.max())
                    hold_range = st.slider(
                        "Holding period (days)",
                        min_value=float(round(hold_min, 2)),
                        max_value=float(round(max(hold_max, hold_min + 0.01), 2)),
                        value=(
                            float(current.hold_days_range[0] if current.hold_days_range else hold_min),
                            float(current.hold_days_range[1] if current.hold_days_range else hold_max),
                        ),
                    )
                else:
                    hold_range = None
            else:
                hold_range = None

            pnl_range = None
            if column_map.pnl and column_map.pnl in df.columns:
                pnl_series = pd.to_numeric(df[column_map.pnl], errors="coerce").dropna()
                if not pnl_series.empty:
                    pnl_range = st.slider(
                        "P&L clip range",
                        min_value=float(pnl_series.min()),
                        max_value=float(pnl_series.max()),
                        value=(
                            float(current.pnl_clip[0] if current.pnl_clip else pnl_series.min()),
                            float(current.pnl_clip[1] if current.pnl_clip else pnl_series.max()),
                        ),
                    )

            winsor_enable = st.checkbox("Winsorize P&L (quantiles)", value=current.winsor_pct is not None)
            if winsor_enable:
                winsor_vals = st.slider(
                    "Winsor bounds",
                    min_value=0.0,
                    max_value=1.0,
                    value=(current.winsor_pct if current.winsor_pct else (0.01, 0.99)),
                    step=0.01,
                )
            else:
                winsor_vals = None

            rolling_window = st.slider(
                "Rolling window (days)",
                min_value=5,
                max_value=90,
                value=st.session_state.get("rolling_window", utils.DEFAULT_WINDOWS["rolling_window"]),
            )

            apply_filters = st.form_submit_button("Apply filters")

        reset_filters = st.button("Reset filters")

    if reset_filters:
        st.session_state.filters_state = defaults
        st.session_state.rolling_window = utils.DEFAULT_WINDOWS["rolling_window"]
        st.experimental_rerun()

    start_capital = st.session_state.start_capital

    if apply_filters:
        start_ts = pd.Timestamp(date_range[0]) if date_range and isinstance(date_range, Sequence) else None
        end_ts = pd.Timestamp(date_range[1]) if date_range and isinstance(date_range, Sequence) else None
        filters = utils.FilterParams(
            start=start_ts,
            end=end_ts,
            symbols=selected_symbols or None,
            strategies=selected_strategies or None,
            runs=selected_runs or None,
            min_trades_per_symbol=int(min_trades),
            hold_days_range=hold_range if hold_range else None,
            pnl_clip=pnl_range if pnl_range else None,
            winsor_pct=winsor_vals,
        )
        st.session_state.filters_state = filters
        st.session_state.rolling_window = rolling_window

    active_filters: utils.FilterParams = st.session_state.filters_state
    rolling_window = st.session_state.get("rolling_window", utils.DEFAULT_WINDOWS["rolling_window"])
    filtered_df = filter_dataframe(df, column_map, active_filters)

    filter_summary = utils.summarize_filters(active_filters)
    if filter_summary:
        summary_text = " | ".join(f"**{k}:** {v}" for k, v in filter_summary.items())
        st.info(f"Active filters → {summary_text}")

    metrics = compute.kpis(filtered_df, column_map, start_capital=start_capital)
    equity_df = compute.equity_curve(filtered_df, column_map, start_capital=start_capital)
    rolling = compute.rolling_metrics(filtered_df, column_map, window=rolling_window)
    drawdown_df = compute.drawdowns(filtered_df, column_map, start_capital=start_capital)
    league = compute.per_symbol_table(filtered_df, column_map)
    hold_summary = compute.holding_period_summary(filtered_df)
    timeline_df = compute.timeline_dataframe(filtered_df, column_map)
    runs_df = compute.runs_summary(filtered_df, column_map)
    calendar = compute.calendar_views(filtered_df, column_map)
    pnl_series = compute.pnl_distribution(filtered_df)
    holding_hist = compute.holding_period_histogram(filtered_df)
    holding_scatter = compute.holding_period_scatter(filtered_df)
    holding_box = compute.holding_period_box(filtered_df)
    streaks = compute.streak_statistics(filtered_df)
    bootstrap_data = compute.bootstrap_returns(filtered_df, column_map)
    tier_sample = compute.tier_summary_table(league)
    tier_counts = league["tier"].value_counts() if not league.empty and "tier" in league.columns else None

    tabs = st.tabs(
        [
            "Overview",
            "Per-Symbol League",
            "Roundtrips Explorer",
            "Distributions",
            "Risk & Drawdowns",
            "Backtest Runs",
        ]
    )

    with tabs[0]:
        render_kpi_cards(metrics)
        equity_fig = viz.fig_equity_curve(equity_df)
        # st.plotly_chart(equity_fig, width="stretch")
        equity_fig.update_layout(
            width=1000,
            height=600,
        )
        st.plotly_chart(equity_fig)
        download_plotly_figure(equity_fig, "Download equity chart (PNG)", "equity_curve")

        with st.expander("Assumptions"):
            st.markdown("Transaction costs or slippage are not explicitly modelled in the current calculations.")

        st.subheader("P&L Concentration")
        conc_cols = st.columns(3)
        conc_cols[0].metric("Top 5 trades share", utils.fmt_pct(metrics.get("top5_share")))
        conc_cols[1].metric("Top 1% trades share", utils.fmt_pct(metrics.get("top1pct_share")))
        conc_cols[2].metric("Gini(|P&L|)", f"{metrics.get('gini_abs'):.3f}" if metrics.get("gini_abs") is not None else "—")
        lorenz_fig = viz.fig_lorenz_curve(metrics.get("lorenz_curve"))
        lorenz_fig.update_layout(
            autosize=False,
            width=800,
            height=800,
        )
        st.plotly_chart(lorenz_fig, use_container_width=False)

        if streaks:
            st.subheader("Streaks")
            streak_cols = st.columns(4)
            streak_cols[0].metric("Longest win streak", streaks.get("longest_win_streak", "—"))
            streak_cols[1].metric("Longest loss streak", streaks.get("longest_loss_streak", "—"))
            streak_cols[2].metric("Avg win streak", f"{streaks.get('avg_win_streak', 0):.2f}")
            streak_cols[3].metric("Avg loss streak", f"{streaks.get('avg_loss_streak', 0):.2f}")

        if hold_summary is not None and not hold_summary.empty:
            st.subheader("Holding Period Summary")
            st.dataframe(
                hold_summary.style.format(
                    {"pnl_mean": "{:.2f}", "pnl_median": "{:.2f}", "win_rate": "{:.2%}"}
                ),
                width="stretch",
            )
            
        holding_hist_fig = viz.fig_hold_histogram(holding_hist)
        holding_hist_fig.update_layout(
            width=1000,
            height=600,
        )
        st.plotly_chart(holding_hist_fig, use_container_width=False)
        if holding_scatter:
            holding_scatter_fig = viz.fig_hold_scatter(holding_scatter["scatter"], holding_scatter.get("line"))
            holding_scatter_fig.update_layout(
                width=1000,
                height=800,
            )
            st.plotly_chart(
                holding_scatter_fig,
                use_container_width=False,
                # width="stretch",
            )
        if not holding_box.empty:
            holding_box_fig = viz.fig_hold_boxplot(holding_box)
            holding_box_fig.update_layout(
                width=1000,
                height=800,
            )
            st.plotly_chart(holding_box_fig, use_container_width=False)

        with st.expander("Data preview"):
            st.dataframe(filtered_df.head(50), width="stretch")

    with tabs[1]:
        st.subheader("Per-Symbol League Table")
        if league.empty:
            st.warning("No symbols after applying filters.")
        else:
            def _win_rate_style(val: float) -> str:
                if pd.isna(val):
                    return ""
                if val >= 0.6:
                    return "background-color: #d4edda; color: #155724;"
                if val < 0.4:
                    return "background-color: #f8d7da; color: #721c24;"
                return ""

            def _pnl_style(val: float) -> str:
                if pd.isna(val):
                    return ""
                if val > 0:
                    return "background-color: #d4edda; color: #155724;"
                if val < 0:
                    return "background-color: #f8d7da; color: #721c24;"
                return ""

            styled = (
                league.style.format(
                    {
                        "win_rate": "{:.2%}",
                        "total_pnl": "{:.2f}",
                        "avg_pnl": "{:.2f}",
                        "median_pnl": "{:.2f}",
                        "avg_win": "{:.2f}",
                        "avg_loss": "{:.2f}",
                        "expectancy_usd": "{:.2f}",
                        "ret_mean": "{:.4f}",
                        "ret_std": "{:.4f}",
                        "ret_p5": "{:.4f}",
                        "avg_hold_days": "{:.2f}",
                    }
                )
                .map(_win_rate_style, subset="win_rate")
                .map(_pnl_style, subset="total_pnl")
            )
            st.dataframe(styled, width="stretch")
            st.download_button(
                "Download league table (CSV)",
                league.to_csv().encode("utf-8"),
                file_name="per_symbol_league.csv",
                mime="text/csv",
                width="content",
            )
            st.plotly_chart(viz.fig_per_symbol_bar(league), width="stretch")
            # st.plotly_chart(viz.fig_per_symbol_scatter(league), width="stretch")
            per_symbol_scatter_fig = viz.fig_per_symbol_scatter(league)
            per_symbol_scatter_fig.update_layout(
                width=1000,
                height=800,
            )
            st.plotly_chart(per_symbol_scatter_fig, use_container_width=False)
            win_rate_bubble_fig = viz.fig_win_rate_bubble(league, annotate=False)
            win_rate_bubble_fig.update_layout(
                width=1000,
                height=900,
            )
            st.plotly_chart(win_rate_bubble_fig, use_container_width=False)
            win_rate_bubble_fig = viz.fig_win_rate_bubble(league, annotate=True)
            win_rate_bubble_fig.update_layout(
                width=1000,
                height=900,
            )
            st.plotly_chart(win_rate_bubble_fig, use_container_width=False)
            
            # st.plotly_chart(viz.fig_win_rate_bubble(league, annotate=True), width="stretch")
            st.plotly_chart(viz.fig_tier_violin(league), width="stretch")

            if tier_counts is not None and not tier_counts.empty:
                st.write("Tier counts:", tier_counts.to_frame("count"))

            if not tier_sample.empty:
                st.subheader("Top 25 Symbols per Tier")
                st.dataframe(tier_sample, width="stretch")

            policy_columns = [
                "trades",
                "win_rate",
                "win_ci_lo",
                "win_ci_hi",
                "avg_pnl",
                "avg_pnl_ci_lo",
                "avg_pnl_ci_hi",
                "expectancy_usd",
                "ret_mean",
                "ret_std",
                "ret_p5",
                "tier",
                "size_mult",
                "size_cap",
            ]
            policy_df = league.loc[:, [c for c in policy_columns if c in league.columns]]
            st.download_button(
                "Download tier policy CSV",
                policy_df.to_csv().encode("utf-8"),
                file_name="symbol_tiers_policy.csv",
                mime="text/csv",
                width="content",
            )

    with tabs[2]:
        st.subheader("Roundtrips Explorer")
        if filtered_df.empty:
            st.warning("No trades available for the current filters.")
        else:
            search_term = st.text_input("Search trades (symbol, strategy, run, side)")
            explorer_df = filtered_df.copy()
            if search_term:
                search_term_lower = search_term.lower()
                explorer_df = explorer_df[
                    explorer_df.apply(
                        lambda row: any(
                            search_term_lower in str(row[col]).lower()
                            for col in [column_map.symbol, column_map.strategy, column_map.run_id, column_map.side]
                            if col and col in explorer_df.columns
                        ),
                        axis=1,
                    )
                ]
            st.dataframe(explorer_df, width="stretch", height=420)
            st.download_button(
                "Download trades (CSV)",
                explorer_df.to_csv(index=False).encode("utf-8"),
                file_name="filtered_trades.csv",
                mime="text/csv",
                width="content",
            )

            if column_map.trade_id and column_map.trade_id in explorer_df.columns:
                trade_options = explorer_df[column_map.trade_id].astype(str).tolist()
                selected_trade = st.selectbox("Inspect trade", options=trade_options)
                trade_row = explorer_df[explorer_df[column_map.trade_id].astype(str) == selected_trade].head(1)
                if not trade_row.empty:
                    with st.expander("Trade details"):
                        st.write(trade_row.T)

            timeline_fig = viz.fig_timeline(timeline_df)
            st.plotly_chart(timeline_fig, width="stretch")
            download_plotly_figure(timeline_fig, "Download timeline (PNG)", "timeline")

    with tabs[3]:
        st.subheader("Distribution Views")
        log_scale = st.checkbox("Log scale", key="hist_log_scale")
        st.plotly_chart(viz.fig_pnl_histogram(pnl_series), width="stretch")
        st.plotly_chart(viz.fig_hist_returns(filtered_df, log_y=log_scale), width="stretch")
        # st.plotly_chart(
        #     viz.fig_violin_returns(
        #         filtered_df,
        #         by=column_map.symbol or "symbol",
        #         color=column_map.strategy if column_map.strategy in filtered_df.columns else None,
        #     ),
        #     width="stretch",
        # )
        box_fig = viz.fig_box_returns(
            filtered_df,
            by=column_map.symbol or "symbol",
            color=column_map.strategy if column_map.strategy in filtered_df.columns else None,
        )
        box_fig.update_layout(
            width=1200,
            height=1000,
        )
        st.plotly_chart(
            box_fig,
            use_container_width=False,
        )
        # st.plotly_chart(viz.fig_ecdf_returns(filtered_df), width="stretch")
        if bootstrap_data:
            bootstrap_fig = viz.fig_bootstrap_distributions(
                bootstrap_data.get("means"),
                bootstrap_data.get("medians"),
                bootstrap_data.get("ci_mean"),
                bootstrap_data.get("ci_median"),
            )
            bootstrap_fig.update_layout(
                width=1000,
                height=1200,
            )
            # st.plotly_chart(bootstrap_fig, use_container_width=False)
            st.plotly_chart(
                bootstrap_fig,
                use_container_width=False,
            )
            st.write(
                f"Mean return 95% CI: {bootstrap_data['ci_mean'][0]:.4f} → {bootstrap_data['ci_mean'][1]:.4f}"
            )
            st.write(
                f"Median return 95% CI: {bootstrap_data['ci_median'][0]:.4f} → {bootstrap_data['ci_median'][1]:.4f}"
            )

    with tabs[4]:
        st.subheader("Risk & Drawdowns")
        if calendar:
            best = calendar.get("best_day")
            worst = calendar.get("worst_day")
            info_cols = st.columns(2)
            if best:
                best_label = best[0].strftime("%Y-%m-%d") if hasattr(best[0], "strftime") else str(best[0])
                info_cols[0].markdown(f"**Best day:** {best_label} &nbsp; ({utils.fmt_currency(best[1])})")
            if worst:
                worst_label = worst[0].strftime("%Y-%m-%d") if hasattr(worst[0], "strftime") else str(worst[0])
                info_cols[1].markdown(f"**Worst day:** {worst_label} &nbsp; ({utils.fmt_currency(worst[1])})")
            # st.plotly_chart(viz.fig_daily_panels(calendar["daily"]), width="stretch")
            daily_panels_fig = viz.fig_daily_panels(calendar["daily"])
            daily_panels_fig.update_layout(
                width=1000,
                height=1200,
            )
            st.plotly_chart(daily_panels_fig, use_container_width=False)
            st.plotly_chart(viz.fig_monthly_pnl(calendar["monthly"]), width="stretch")
            st.plotly_chart(viz.fig_weekday_pnl(calendar["weekday"]), width="stretch")
            # st.plotly_chart(viz.fig_calendar_heatmap(calendar["pivot"]), width="stretch")
            calendar_heatmap_fig = viz.fig_calendar_heatmap(calendar["pivot"])
            calendar_heatmap_fig.update_layout(
                width=1200,
                height=1000,
            )
            st.plotly_chart(calendar_heatmap_fig, use_container_width=False)

        underwater_fig = viz.fig_underwater(drawdown_df)
        st.plotly_chart(underwater_fig, width="stretch")
        download_plotly_figure(underwater_fig, "Download underwater chart (PNG)", "underwater")
        st.plotly_chart(viz.fig_rolling_metrics(rolling), width="stretch")

    with tabs[5]:
        st.subheader("Backtest Runs")
        if runs_df.empty:
            st.info("No run identifiers detected.")
        else:
            st.dataframe(runs_df, width="stretch")
            st.plotly_chart(viz.fig_runs_compare(runs_df), width="stretch")
            sharpe_series = runs_df["sharpe_like"].dropna()
            if not sharpe_series.empty:
                st.metric("Mean Sharpe across runs", f"{sharpe_series.mean():.2f}")
                st.metric("Sharpe std dev", f"{sharpe_series.std(ddof=1):.2f}")
                gt_one = (sharpe_series > 1).mean()
                gt_two = (sharpe_series > 2).mean()
                st.write(f"Fraction of runs with Sharpe > 1: {gt_one:.0%}, Sharpe > 2: {gt_two:.0%}")

    st.divider()
    st.subheader("Reporting & Exports")
    st.download_button(
        "Download current schema report (CSV)",
        schema_report.to_csv(index=False).encode("utf-8"),
        file_name="schema_report.csv",
        mime="text/csv",
        width="content",
    )

    if st.button("Generate Markdown report"):
        report = compute.generate_report(filtered_df, column_map)
        st.download_button(
            "Download report.md",
            report.encode("utf-8"),
            file_name="report.md",
            mime="text/markdown",
            width="content",
        )


if __name__ == "__main__":
    main()

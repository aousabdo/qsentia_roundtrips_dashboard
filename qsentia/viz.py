from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fig_equity_curve(curve_df: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    if curve_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Equity curve unavailable",
            template="plotly_white",
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve_df["date"],
            y=curve_df["equity"],
            name="Equity",
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve_df["date"],
            y=curve_df["rolling_sharpe"],
            name="Rolling Sharpe",
            mode="lines",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            yaxis="y2",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Equity", zeroline=False),
        yaxis2=dict(
            title="Rolling Sharpe",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def fig_underwater(drawdown_df: pd.DataFrame) -> go.Figure:
    if drawdown_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Drawdown unavailable", template="plotly_white")
        return fig

    fig = go.Figure(
        go.Scatter(
            x=drawdown_df["date"],
            y=drawdown_df["drawdown"],
            mode="lines",
            name="Drawdown",
            line=dict(color="#d62728"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Underwater Curve",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Drawdown",
    )
    return fig


def fig_per_symbol_bar(table_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    if table_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No per-symbol data", template="plotly_white")
        return fig

    top = table_df.head(top_n).reset_index().rename(columns={"index": "symbol"})
    palette = {"Tier 1": "#1b9e77", "Tier 2": "#d95f02", "Tier 3": "#7570b3", "Outlier-Hold": "#e7298a"}

    fig = px.bar(
        top,
        x="symbol",
        y="total_pnl",
        color="tier" if "tier" in top.columns else None,
        color_discrete_map=palette if "tier" in top.columns else None,
        title=f"Top {top_n} Symbols by Total P&L",
        text="total_pnl",
    )

    has_tier = "tier" in top.columns
    if has_tier:
        fig.update_traces(
            customdata=top[["tier"]].values,
            hovertemplate="<b>%{x}</b><br>Tier: %{customdata[0]}<br>Total P&L: %{y:.1f}<extra></extra>",
        )
    else:
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Total P&L: %{y:.1f}<extra></extra>",
        )

    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_layout(template="plotly_white", xaxis_title="Symbol", yaxis_title="Total P&L", yaxis=dict(tickformat=".1f"))
    return fig





def fig_violin_returns(df: pd.DataFrame, by: str = "symbol", color: Optional[str] = None) -> go.Figure:
    if df.empty or "_ret" not in df.columns or df["_ret"].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No return data for violin plot", template="plotly_white")
        return fig
    fig = px.violin(
        df.dropna(subset=["_ret"]),
        x=by,
        y="_ret",
        color=color,
        box=True,
        points="all",
        hover_data=["_ret"],
        title=f"Return Distribution by {by.title()}",
    )
    fig.update_layout(template="plotly_white", yaxis_title="Return")
    return fig


def fig_box_returns(df: pd.DataFrame, by: str = "symbol", color: Optional[str] = None) -> go.Figure:
    if df.empty or "_ret" not in df.columns or df["_ret"].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No return data for box plot", template="plotly_white")
        return fig
    fig = px.box(
        df.dropna(subset=["_ret"]),
        x=by,
        y="_ret",
        color=color,
        points="suspectedoutliers",
        title=f"Return Box Plot by {by.title()}",
    )
    fig.update_layout(template="plotly_white", yaxis_title="Return")
    return fig


def fig_hist_returns(df: pd.DataFrame, log_y: bool = False) -> go.Figure:
    if df.empty or "_ret" not in df.columns or df["_ret"].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No return data for histogram", template="plotly_white")
        return fig
    fig = px.histogram(
        df.dropna(subset=["_ret"]),
        x="_ret",
        nbins=60,
        marginal="rug",
        opacity=0.85,
        title="Return Histogram",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Return", yaxis_title="Frequency")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def fig_ecdf_returns(df: pd.DataFrame) -> go.Figure:
    if df.empty or "_ret" not in df.columns or df["_ret"].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No return data for ECDF", template="plotly_white")
        return fig
    data = df["_ret"].dropna().sort_values()
    y = pd.Series(range(1, len(data) + 1), index=data.index) / len(data)
    fig = go.Figure(
        go.Scatter(
            x=data,
            y=y,
            mode="lines",
            name="ECDF",
            line=dict(color="#17becf"),
        )
    )
    fig.update_layout(
        title="Empirical CDF of Returns",
        template="plotly_white",
        xaxis_title="Return",
        yaxis_title="Cumulative probability",
    )
    return fig


def fig_lorenz_curve(lorenz: Optional[Iterable[float]]) -> go.Figure:
    if lorenz is None:
        fig = go.Figure()
        fig.update_layout(title="Lorenz curve unavailable", template="plotly_white")
        return fig
    lorenz = list(lorenz)
    xs = [i / (len(lorenz) - 1) for i in range(len(lorenz))]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=lorenz,
            mode="lines",
            name="Lorenz |P&L|",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Equality",
            line=dict(color="#aaaaaa", dash="dash"),
        )
    )
    fig.update_layout(
        title="Lorenz Curve of Absolute P&L",
        template="plotly_white",
        xaxis_title="Fraction of trades",
        yaxis_title="Fraction of total |P&L|",
        autosize=False,
        width=500,
        height=500,
    )
    return fig


def fig_daily_panels(daily_df: pd.DataFrame) -> go.Figure:
    if daily_df.empty:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.update_layout(title="Daily performance unavailable", template="plotly_white")
        return fig
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(
        go.Bar(
            x=daily_df["date"],
            y=daily_df["daily_pnl"],
            name="Daily P&L",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["equity"],
            name="Cumulative P&L",
            mode="lines",
            line=dict(color="#1f77b4"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["drawdown"],
            name="Drawdown",
            mode="lines",
            line=dict(color="#d62728"),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        template="plotly_white",
        title="Daily P&L, Cumulative P&L, and Drawdown",
        showlegend=False,
    )
    fig.update_xaxes(title="Date", row=3, col=1)
    fig.update_yaxes(title="Daily P&L", row=1, col=1)
    fig.update_yaxes(title="Cumulative P&L", row=2, col=1)
    fig.update_yaxes(title="Drawdown", row=3, col=1)
    return fig


def fig_monthly_pnl(monthly_series: pd.Series) -> go.Figure:
    if monthly_series.empty:
        fig = go.Figure()
        fig.update_layout(title="Monthly P&L unavailable", template="plotly_white")
        return fig
    df = monthly_series.reset_index()
    value_col = monthly_series.name if monthly_series.name else 0
    df = df.rename(columns={"index": "month", value_col: "pnl"})
    fig = px.bar(df, x="month", y="pnl", title="Monthly P&L (sum)")
    fig.update_layout(template="plotly_white", xaxis_title="Month", yaxis_title="P&L")
    return fig


def fig_weekday_pnl(weekday_series: pd.Series) -> go.Figure:
    if weekday_series.empty:
        fig = go.Figure()
        fig.update_layout(title="Weekday P&L unavailable", template="plotly_white")
        return fig
    df = weekday_series.reset_index()
    value_col = weekday_series.name if weekday_series.name else 0
    df = df.rename(columns={"index": "weekday", value_col: "pnl"})
    fig = px.bar(df, x="weekday", y="pnl", title="Weekday P&L (sum)")
    fig.update_layout(template="plotly_white", xaxis_title="Weekday", yaxis_title="P&L")
    return fig


def fig_calendar_heatmap(pivot: pd.DataFrame) -> go.Figure:
    if pivot.empty:
        fig = go.Figure()
        fig.update_layout(title="Calendar heatmap unavailable", template="plotly_white")
        return fig
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            colorbar=dict(title="P&L"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Monthly vs Weekday P&L",
        xaxis_title="Weekday",
        yaxis_title="Month",
    )
    return fig


def fig_hold_histogram(series: pd.Series) -> go.Figure:
    if series.empty:
        fig = go.Figure()
        fig.update_layout(title="Holding period histogram unavailable", template="plotly_white")
        return fig
    fig = px.histogram(series, nbins=30, title="Holding Period (days)")
    fig.update_layout(template="plotly_white", xaxis_title="Holding days", yaxis_title="Frequency")
    return fig


def fig_hold_scatter(scatter_df: pd.DataFrame, line: Optional[Tuple[float, float]]) -> go.Figure:
    if scatter_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Return % vs Holding Days unavailable", template="plotly_white")
        return fig
    fig = px.scatter(
        scatter_df,
        x="hold_days",
        y="return",
        opacity=0.3,
        title="Return % vs Holding Days",
    )
    if line and line[0] is not None and line[1] is not None:
        xs = pd.Series(scatter_df["hold_days"]).sort_values()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=line[0] * xs + line[1],
                mode="lines",
                name="Trend",
                line=dict(color="#d62728"),
            )
        )
    fig.update_layout(template="plotly_white", xaxis_title="Holding days", yaxis_title="Return")
    return fig


def fig_hold_boxplot(box_df: pd.DataFrame) -> go.Figure:
    if box_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Return % by Holding Period Bin unavailable", template="plotly_white")
        return fig
    fig = px.box(
        box_df,
        x="hold_bin",
        y="return",
        title="Return % by Holding Period Bin",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Holding period bin", yaxis_title="Return")
    return fig


def fig_pnl_histogram(series: pd.Series) -> go.Figure:
    if series.empty:
        fig = go.Figure()
        fig.update_layout(title="P&L distribution unavailable", template="plotly_white")
        return fig
    fig = px.histogram(
        series,
        nbins=50,
        title="Distribution of P&L (USD)",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="P&L (USD)",
        yaxis_title="Frequency",
    )
    return fig


def fig_bootstrap_distributions(
    means: Optional[Iterable[float]],
    medians: Optional[Iterable[float]],
    ci_mean: Optional[Tuple[float, float]],
    ci_median: Optional[Tuple[float, float]],
) -> go.Figure:
    means_list = list(means) if means is not None else []
    medians_list = list(medians) if medians is not None else []

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, subplot_titles=("Bootstrap Distribution of Mean Return (%)", "Bootstrap Distribution of Median Return (%)"))
    if means_list:
        fig.add_trace(
            go.Histogram(
                x=means_list,
                nbinsx=50,
                name="Mean bootstrap",
                marker_color="#2ca02c",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        if ci_mean:
            fig.add_vline(x=ci_mean[0], line_dash="dash", line_color="#d62728", row=1, col=1)
            fig.add_vline(x=ci_mean[1], line_dash="dash", line_color="#d62728", row=1, col=1)
    if medians_list:
        fig.add_trace(
            go.Histogram(
                x=medians_list,
                nbinsx=50,
                name="Median bootstrap",
                marker_color="#1f77b4",
                opacity=0.7,
            ),
            row=2,
            col=1,
        )
        if ci_median:
            fig.add_vline(x=ci_median[0], line_dash="dash", line_color="#d62728", row=2, col=1)
            fig.add_vline(x=ci_median[1], line_dash="dash", line_color="#d62728", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=700,
    )
    fig.update_xaxes(title="Mean return (%)", row=1, col=1)
    fig.update_xaxes(title="Median return (%)", row=2, col=1)
    fig.update_yaxes(title="Density", row=1, col=1)
    fig.update_yaxes(title="Density", row=2, col=1)
    return fig


def fig_timeline(timeline_df: pd.DataFrame) -> go.Figure:
    if timeline_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Timeline unavailable", template="plotly_white")
        return fig
    fig = px.timeline(
        timeline_df,
        x_start="start",
        x_end="end",
        y="symbol",
        color="side" if "side" in timeline_df.columns else "symbol",
        hover_data={
            "pnl": ":.2f",
            "hold_days": ":.2f",
            "start": "|%Y-%m-%d %H:%M",
            "end": "|%Y-%m-%d %H:%M",
        },
        title="Holding Period Timeline",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white")
    return fig


def fig_runs_compare(runs_df: pd.DataFrame) -> go.Figure:
    if runs_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No backtest runs detected", template="plotly_white")
        return fig
    fig = px.bar(
        runs_df.reset_index(),
        x=runs_df.index.name or "run_id",
        y="total_pnl",
        color="sharpe_like",
        hover_data=["trades", "avg_pnl", "win_rate"],
        title="Backtest Runs – Total P&L",
        color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Run", yaxis_title="Total P&L")
    return fig


def fig_rolling_metrics(rolling: dict) -> go.Figure:
    if not rolling:
        fig = go.Figure()
        fig.update_layout(title="Rolling metrics unavailable", template="plotly_white")
        return fig
    fig = go.Figure()
    if "rolling_mean" in rolling:
        fig.add_trace(
            go.Scatter(
                x=rolling["rolling_mean"].index,
                y=rolling["rolling_mean"],
                mode="lines",
                name="Rolling Mean P&L",
                line=dict(color="#2ca02c"),
            )
        )
    if "rolling_vol" in rolling:
        fig.add_trace(
            go.Scatter(
                x=rolling["rolling_vol"].index,
                y=rolling["rolling_vol"],
                mode="lines",
                name="Rolling Volatility",
                line=dict(color="#9467bd"),
            )
        )
    if "rolling_sharpe" in rolling:
        fig.add_trace(
            go.Scatter(
                x=rolling["rolling_sharpe"].index,
                y=rolling["rolling_sharpe"],
                mode="lines",
                name="Rolling Sharpe (scaled)",
                line=dict(color="#8c564b", dash="dash"),
            )
        )
    fig.update_layout(
        title="Rolling Metrics",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
    )
    return fig


def fig_per_symbol_scatter(league: pd.DataFrame) -> go.Figure:
    if league.empty:
        fig = go.Figure()
        fig.update_layout(title="Per-symbol scatter unavailable", template="plotly_white")
        return fig
    df = league.reset_index().rename(columns={"index": "symbol"})
    palette = {
        "Tier 1": "#1b9e77",
        "Tier 2": "#d95f02",
        "Tier 3": "#7570b3",
        "Outlier-Hold": "#e7298a",
    }
    fig = px.scatter(
        df,
        x="win_rate",
        y="avg_pnl",
        color="tier" if "tier" in df.columns else None,
        color_discrete_map=palette if "tier" in df.columns else None,
        hover_data=["symbol", "total_pnl", "trades"],
        title="Average P&L vs Win Rate per Symbol",
        size_max=60,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Win rate",
        yaxis_title="Average P&L (USD)",
    )
    return fig


def fig_win_rate_bubble(league: pd.DataFrame, annotate: bool = False) -> go.Figure:
    if league.empty:
        fig = go.Figure()
        fig.update_layout(title="Bubble chart unavailable", template="plotly_white")
        return fig
    df = league.reset_index().rename(columns={"index": "symbol"})
    # print(df)
    fig = px.scatter(
        df,
        x="win_rate",
        y="avg_pnl",
        size="trades",
        color="ret_p5",
        color_continuous_scale="RdBu_r",
        hover_data=["symbol", "total_pnl", "ret_p5", "tier"],
        title="Win Rate vs Avg P&L by Symbol",
        size_max=40,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Win rate",
        yaxis_title="Average P&L (USD)",
        coloraxis=dict(colorbar=dict(title="ret_p5"), cmin=df["ret_p5"].min(), cmax=df["ret_p5"].max(), cauto=True, cmid=0),
    )
    fig.add_vline(x=0.65, line_dash="dash", line_color="#2ca02c")
    fig.add_vline(x=0.55, line_dash="dash", line_color="#ff7f0e")
    fig.add_hline(y=0, line_color="#7f7f7f")
    if annotate:
        top_pnl = df.sort_values("total_pnl", ascending=False).head(8)
        worst_tail = df.sort_values("ret_p5").head(5)
        annotated = pd.concat([top_pnl, worst_tail]).drop_duplicates(subset="symbol")
        fig.add_trace(
            go.Scatter(
                x=annotated["win_rate"],
                y=annotated["avg_pnl"],
                mode="text",
                text=annotated["symbol"],
                textposition="top center",
                showlegend=False,
            )
        )
    return fig


def fig_tier_violin(league: pd.DataFrame) -> go.Figure:
    if league.empty or "tier" not in league.columns:
        fig = go.Figure()
        fig.update_layout(title="Tier violin unavailable", template="plotly_white")
        return fig

    metrics = [
        ("trades", "Trades by Tier"),
        ("win_rate", "Win Rate by Tier"),
        ("total_pnl", "Total P&L by Tier"),
        ("avg_pnl", "Average P&L by Tier"),
    ]
    tier_order = ["Tier 1", "Tier 2", "Tier 3", "Outlier-Hold"]
    color_cycle = px.colors.qualitative.Dark2
    palette = {tier: color_cycle[idx % len(color_cycle)] for idx, tier in enumerate(tier_order)}

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[title for _, title in metrics],
        shared_xaxes=False,
        shared_yaxes=False,
    )

    for idx, (metric, _) in enumerate(metrics, start=1):
        if metric not in league.columns:
            continue
        row = 1 if idx <= 2 else 2
        col_pos = 1 if idx in {1, 3} else 2

        for tier in tier_order:
            color = palette.get(tier)
            tier_values = league.loc[league["tier"] == tier, metric].dropna()
            if tier_values.empty or color is None:
                continue
            fig.add_trace(
                go.Violin(
                    x=tier_values,
                    y=[tier] * len(tier_values),
                    name=tier,
                    legendgroup=tier,
                    orientation="h",
                    scalegroup=metric,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.6,
                    meanline_visible=False,
                    spanmode="hard",
                    points="all",
                    pointpos=0.0,
                    jitter=0.0,
                    marker=dict(symbol="line-ns-open", size=8, line=dict(color=color, width=1)),
                    showlegend=(idx == 1),
                ),
                row=row,
                col=col_pos,
            )

        fig.update_xaxes(title_text=metric, row=row, col=col_pos)
        fig.update_yaxes(
            title_text="Tier",
            row=row,
            col=col_pos,
            categoryorder="array",
            categoryarray=tier_order,
        )

    fig.update_layout(
        template="plotly_white",
        height=850,
        title="Tier Metric Violins",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

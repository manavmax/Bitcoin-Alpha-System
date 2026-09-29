from __future__ import annotations

import math
import os
import sys
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Resolve the sibling `theme` module regardless of how the app is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import theme  # noqa: E402

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Bitcoin Intelligence — Model 8", layout="wide")

# =====================================================
# AUTO REFRESH — ONLY LIVE PRICE
# =====================================================
st_autorefresh(interval=1500, key="live_price_refresh")

# =====================================================
# PATHS
# =====================================================
DATA_PRICE = "data/raw/btc_price_daily.csv"
DATA_SIGNAL = "model8_meta_classifier/results/model8_final_signal.csv"
DATA_MODEL8 = "model8_meta_classifier/data/model8_dataset.csv"


def _inject_css() -> None:
    # Presentation only -- every token lives in theme.py / .streamlit/config.toml
    theme.inject_css(st)

# =====================================================
# LIVE BTC PRICE (1s TICK)
# =====================================================
def fetch_live_btc() -> float | None:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            timeout=2
        ).json()
        return float(r["price"])
    except Exception:
        return None

live_price = fetch_live_btc()

# =====================================================
# LOAD DAILY DATA (NORMALIZED DATES)
# =====================================================
@st.cache_data
def load_price():
    df = pd.read_csv(DATA_PRICE)
    # Support both legacy 'open_time' and 'date' columns
    if "date" in df.columns:
        ts_col = "date"
    else:
        ts_col = "open_time"
    df["date"] = pd.to_datetime(df[ts_col], utc=True).dt.date
    return df.sort_values("date")[["date", "open", "high", "low", "close", "volume"]]

@st.cache_data
def load_signal():
    df = pd.read_csv(DATA_SIGNAL)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    return df.sort_values("date")

@st.cache_data
def load_model8_dataset():
    df = pd.read_csv(DATA_MODEL8)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    return df.sort_values("date")

price_df = load_price()
signal_df = load_signal()

# =====================================================
# LATEST MODEL DECISION
# =====================================================
latest = signal_df.iloc[-1]
decision_date = latest["date"]
actionable_date = decision_date + timedelta(days=1)

_inject_css()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.markdown("## Controls")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold (τ)",
    0.50,
    0.90,
    0.70,
    0.01,
    help="Trades are only taken when confidence ≥ τ.",
)

lookback_days = st.sidebar.select_slider(
    "Chart lookback",
    options=[30, 90, 180, 365, 730, 1500],
    value=365,
)

show_all_signals = st.sidebar.toggle(
    "Show all model signals (ignore τ filter)",
    value=False,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Daily inference uses closed UTC candles. Live price is informational only."
)

# =====================================================
# HEADER (TOP BAR)
# =====================================================
top_l, top_r = st.columns([2.0, 1.0])
with top_l:
    st.markdown(
        """
        <div class="cq-topstrip">
          <span class="cq-brand">Bitcoin Intelligence — Model 8</span>
          <span class="cq-session">Regime-aware multi-factor system · research dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_r:
    if live_price:
        st.metric("BTC (live)", f"${live_price:,.2f}")

vol_label = "High vol" if float(latest["vol_regime"]) == 1.0 else "Low vol"
# macro_regime == 1  <=>  macro_signal < 0  (bearish / risk-off), which is the
# convention Model 8A was trained on (see build_model8A_dataset.py).
macro_label = "Risk-off" if float(latest["macro_regime"]) == 1.0 else "Risk-on"

hdr1, hdr2, hdr3, hdr4, hdr5 = st.columns([1.2, 1.0, 1.0, 1.0, 1.2])

# Presentation only: the decision card leads the row, and its direction is
# encoded on the card edge plus the value colour.
decision_card_class = "cq-card cq-card--decision"
decision_value_class = "cq-kpi-value"
if str(latest["final_signal"]) == "LONG":
    decision_card_class = "cq-card cq-card--long"
    decision_value_class = "cq-kpi-value cq-kpi-value--long"
elif str(latest["final_signal"]) == "SHORT":
    decision_card_class = "cq-card cq-card--short"
    decision_value_class = "cq-kpi-value cq-kpi-value--short"

with hdr1:
    st.markdown(
        f"""
        <div class="{decision_card_class}">
          <div class="cq-kpi-title">Model 8 decision</div>
          <div class="{decision_value_class}">{latest["final_signal"]}</div>
          <div class="cq-kpi-sub">Decision: {decision_date} · Actionable: {actionable_date}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hdr2:
    st.markdown(
        f"""
        <div class="cq-card">
          <div class="cq-kpi-title">Confidence</div>
          <div class="cq-kpi-value">{latest["confidence"]:.2f}</div>
          <div class="cq-kpi-sub">Threshold τ = {confidence_threshold:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hdr3:
    st.markdown(
        f"""
        <div class="cq-card">
          <div class="cq-kpi-title">Vol regime</div>
          <div class="cq-kpi-value">{vol_label}</div>
          <div class="cq-kpi-sub">vol_regime = {int(float(latest["vol_regime"]))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hdr4:
    st.markdown(
        f"""
        <div class="cq-card">
          <div class="cq-kpi-title">Macro regime</div>
          <div class="cq-kpi-value">{macro_label}</div>
          <div class="cq-kpi-sub">macro_regime = {int(float(latest["macro_regime"]))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hdr5:
    traded = (signal_df["final_signal"].isin(["LONG", "SHORT"]) & (signal_df["confidence"] >= confidence_threshold)).sum()
    coverage = traded / max(len(signal_df), 1)
    st.markdown(
        f"""
        <div class="cq-card">
          <div class="cq-kpi-title">Trade coverage (τ-filtered)</div>
          <div class="cq-kpi-value">{coverage*100:.1f}%</div>
          <div class="cq-kpi-sub">{int(traded)} trades · sample {len(signal_df)} days</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Effective signal after applying user-selected confidence filter
signal_df = signal_df.copy()
signal_df["effective_signal"] = signal_df.apply(
    lambda row: row["final_signal"]
    if (row["final_signal"] in ["LONG", "SHORT"]) and (row["confidence"] >= confidence_threshold)
    else "NO_TRADE",
    axis=1,
)

if show_all_signals:
    signal_df["effective_signal"] = signal_df["final_signal"].where(
        signal_df["final_signal"].isin(["LONG", "SHORT"]), "NO_TRADE"
    )

# =====================================================
# BTC PRICE + MODEL SIGNALS
# =====================================================
tabs = st.tabs(
    [
        "Overview",
        "Performance",
        "Regimes",
        "Signals",
        "Diagnostics",
        "Playground",
    ]
)

plot_df = price_df.merge(signal_df, on="date", how="left")
plot_df = plot_df.sort_values("date").reset_index(drop=True)

if lookback_days:
    plot_df = plot_df.tail(int(lookback_days)).reset_index(drop=True)

arrow_df = plot_df[plot_df["effective_signal"].isin(["LONG", "SHORT"])].copy()

def _make_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC",
            **theme.CANDLE,
        )
    )
    longs = df[df["effective_signal"] == "LONG"]
    shorts = df[df["effective_signal"] == "SHORT"]

    if not longs.empty:
        fig.add_trace(
            go.Scatter(
                x=longs["date"],
                y=longs["low"] * 0.996,
                mode="markers",
                marker=dict(theme.MARKER_LONG),
                name="LONG",
            )
        )
    if not shorts.empty:
        fig.add_trace(
            go.Scatter(
                x=shorts["date"],
                y=shorts["high"] * 1.004,
                mode="markers",
                marker=dict(theme.MARKER_SHORT),
                name="SHORT",
            )
        )

    fig.update_layout(
        **theme.chart_layout(
            height=theme.H_HERO,
            xaxis=dict(rangeslider=dict(visible=False)),
        )
    )
    return fig

bt_df = plot_df.copy()
bt_df["ret"] = bt_df["close"].pct_change().fillna(0.0)
bt_df["position"] = 0
bt_df.loc[bt_df["effective_signal"] == "LONG", "position"] = 1
bt_df.loc[bt_df["effective_signal"] == "SHORT", "position"] = -1
bt_df["strategy_ret"] = bt_df["position"].shift(1).fillna(0.0) * bt_df["ret"]
bt_df["equity"] = (1.0 + bt_df["strategy_ret"]).cumprod()
bt_df["bh_equity"] = (1.0 + bt_df["ret"]).cumprod()

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0

def _ann_sharpe(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((mu / sd) * math.sqrt(365.0))

def _rolling_sharpe(rets: pd.Series, window: int = 90) -> pd.Series:
    def _sh(x: pd.Series) -> float:
        if len(x) < 2:
            return float("nan")
        sd = x.std(ddof=1)
        if sd == 0:
            return float("nan")
        return float((x.mean() / sd) * math.sqrt(365.0))

    return rets.rolling(window, min_periods=max(10, window // 3)).apply(_sh, raw=False)

bt_df["roll_sharpe_90d"] = _rolling_sharpe(bt_df["strategy_ret"], window=90)

# Preload Model 8 dataset for decomposition/diagnostics
model8_df_full = load_model8_dataset()
model8_df_full = model8_df_full.sort_values("date").reset_index(drop=True)

# =====================================================
# REGIME HEATMAP
# =====================================================
with tabs[0]:
    left, right = st.columns([1.6, 1.0])
    with left:
        st.markdown("### Price & signals")
        st.plotly_chart(_make_price_chart(plot_df), width="stretch")
    with right:
        st.markdown("### Snapshot")
        last = bt_df.dropna(subset=["equity"]).iloc[-1]
        strat_cum = float(last["equity"] - 1.0)
        bh_cum = float(last["bh_equity"] - 1.0)
        sharpe = _ann_sharpe(bt_df["strategy_ret"])
        mdd = _max_drawdown(bt_df["equity"])
        win_rate = float((bt_df["strategy_ret"] > 0).mean()) if len(bt_df) else float("nan")
        trades = int((bt_df["effective_signal"].isin(["LONG", "SHORT"])).sum())

        k1, k2 = st.columns(2)
        k1.metric("Strategy (cum)", f"{strat_cum*100:.1f}%")
        k2.metric("Buy & hold (cum)", f"{bh_cum*100:.1f}%")
        k3, k4 = st.columns(2)
        k3.metric("Sharpe (ann.)", f"{sharpe:.2f}" if not math.isnan(sharpe) else "—")
        k4.metric("Max drawdown", f"{mdd*100:.1f}%")
        k5, k6 = st.columns(2)
        k5.metric("Win rate (daily)", f"{win_rate*100:.1f}%")
        k6.metric("Trades", f"{trades}")

        st.markdown("### Latest decision details")
        st.dataframe(
            signal_df.tail(15).sort_values("date", ascending=False),
            use_container_width=True,
            height=320,
        )

        st.markdown("### Model decomposition")
        diag_all = model8_df_full.merge(signal_df, on="date", how="left")
        selected_date = st.selectbox(
            "Inspect date",
            options=diag_all["date"].tolist()[::-1],
            index=0,
        )
        row = diag_all[diag_all["date"] == selected_date].iloc[0]
        st.caption(
            f"Final: {row.get('final_signal', '—')} · Effective: {row.get('effective_signal', '—')} · "
            f"confidence: {row.get('confidence', float('nan')):.2f} · τ={confidence_threshold:.2f}"
            if isinstance(row.get("confidence", None), (float, int))
            else f"Final: {row.get('final_signal', '—')} · Effective: {row.get('effective_signal', '—')} · τ={confidence_threshold:.2f}"
        )

        base_cols = [c for c in model8_df_full.columns if c.startswith("signal_") and not c.endswith("_missing")]
        if base_cols:
            base_view = pd.DataFrame(
                {
                    "model": base_cols,
                    "signal_value": [row.get(c, float('nan')) for c in base_cols],
                    "missing": [int(row.get(f"{c}_missing", 0)) for c in base_cols],
                }
            )
            st.dataframe(base_view, use_container_width=True, height=240)
        else:
            st.info("Base model signals not found in Model 8 dataset.")

with tabs[1]:
    st.markdown("### Performance")
    perf_fig = go.Figure()
    perf_fig.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["equity"], name="Strategy", line=dict(color=theme.COLORS["accent"], width=2)))
    perf_fig.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["bh_equity"], name="Buy & hold", line=dict(color=theme.COLORS["benchmark"], width=1.6)))
    perf_fig.update_layout(**theme.chart_layout(height=theme.H_MAIN))
    st.plotly_chart(perf_fig, width="stretch")

    rs_fig = go.Figure()
    rs_fig.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["roll_sharpe_90d"], name="Rolling Sharpe (90d)", line=dict(color=theme.COLORS["accent"], width=2)))
    rs_fig.update_layout(**theme.chart_layout(height=theme.H_SUB))
    st.plotly_chart(rs_fig, width="stretch")

with tabs[2]:
    st.markdown("### Regimes")
    regime_df = signal_df[["date", "vol_regime", "macro_regime"]].copy()
    if lookback_days:
        regime_df = regime_df.tail(int(lookback_days)).reset_index(drop=True)

    regime_fig = go.Figure()
    regime_fig.add_trace(
        go.Scatter(
            x=regime_df["date"],
            y=[1] * len(regime_df),
            mode="markers",
            marker=dict(size=7, color=regime_df["vol_regime"], colorscale=theme.SEQUENTIAL),
            name="Vol regime",
        )
    )
    regime_fig.add_trace(
        go.Scatter(
            x=regime_df["date"],
            y=[0] * len(regime_df),
            mode="markers",
            marker=dict(size=7, color=regime_df["macro_regime"], colorscale=theme.DIVERGING),
            name="Macro regime",
        )
    )
    regime_fig.update_layout(
        **theme.chart_layout(
            height=theme.H_MINI,
            yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Macro", "Vol"]),
            showlegend=True,
        )
    )
    st.plotly_chart(regime_fig, width="stretch")

# =====================================================
# TRADE LOG & EXPLANATORY PANEL
# =====================================================
with tabs[3]:
    st.markdown("### Signals")
    cdl, cdr = st.columns([1.0, 1.0])
    with cdl:
        st.download_button(
            "Download Model 8 signals (CSV)",
            data=signal_df.to_csv(index=False).encode("utf-8"),
            file_name="model8_final_signal.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cdr:
        st.download_button(
            "Download Model 8 dataset (CSV)",
            data=model8_df_full.to_csv(index=False).encode("utf-8"),
            file_name="model8_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.dataframe(
        signal_df.sort_values("date", ascending=False),
        use_container_width=True,
        height=520,
    )

with tabs[4]:
    st.markdown("### Diagnostics")

    diag = model8_df_full.merge(signal_df, on="date", how="left")
    if lookback_days:
        diag = diag.tail(int(lookback_days)).reset_index(drop=True)

    st.markdown("#### Missingness (base models)")
    miss_cols = [c for c in diag.columns if c.endswith("_missing")]
    if miss_cols:
        miss_fig = go.Figure()
        for idx, c in enumerate(miss_cols):
            miss_fig.add_trace(
                go.Scatter(
                    x=diag["date"],
                    y=diag[c],
                    mode="lines",
                    name=c.replace("_missing", ""),
                    line=dict(
                        width=1.5,
                        color=theme.CATEGORICAL[idx % len(theme.CATEGORICAL)],
                    ),
                )
            )
        miss_fig.update_layout(
            **theme.chart_layout(
                height=theme.H_SUB,
                yaxis=dict(title="missing flag (0/1)", range=[-0.05, 1.05]),
            )
        )
        st.plotly_chart(miss_fig, width="stretch")
    else:
        st.info("Missing-flag columns not found in Model 8 dataset.")

    st.markdown("#### Trade log (τ-filtered)")
    trades_df = bt_df[bt_df["effective_signal"].isin(["LONG", "SHORT"])].copy()
    trades_df["cum_pnl"] = (1.0 + trades_df["strategy_ret"]).cumprod() - 1.0
    st.dataframe(
        trades_df.sort_values("date", ascending=False)[
            ["date", "effective_signal", "confidence", "strategy_ret", "cum_pnl"]
        ],
        use_container_width=True,
        height=360,
    )

with tabs[5]:
    st.markdown("### Playground — explore base models vs price")

    pg_left, pg_right = st.columns([1.6, 1.0])

    with pg_left:
        base_cols = ["signal_1", "signal_2", "signal_3", "signal_4", "signal_6"]
        available = [c for c in base_cols if c in model8_df_full.columns]

        if not available:
            st.info("Base model signals not found in Model 8 dataset.")
        else:
            chosen = st.multiselect(
                "Base signals to overlay",
                options=available,
                default=["signal_1", "signal_2"],
            )
            smooth = st.slider(
                "Smoothing window (days)",
                min_value=1,
                max_value=30,
                value=5,
            )

            merged_pg = model8_df_full.merge(price_df, on="date", how="inner")
            if lookback_days:
                merged_pg = merged_pg.tail(int(lookback_days)).reset_index(drop=True)

            # build figure
            fig_pg = go.Figure()
            fig_pg.add_trace(
                go.Scatter(
                    x=merged_pg["date"],
                    y=merged_pg["close"],
                    name="BTC close",
                    line=dict(color=theme.COLORS["text"], width=2.0),
                    yaxis="y1",
                )
            )

            palette = theme.CATEGORICAL
            for idx, c in enumerate(chosen):
                s = merged_pg[c].astype(float)
                if smooth > 1:
                    s = s.rolling(smooth, min_periods=1).mean()
                # z-score so different scales can be compared visually
                z = (s - s.mean()) / (s.std(ddof=1) or 1.0)
                fig_pg.add_trace(
                    go.Scatter(
                        x=merged_pg["date"],
                        y=z,
                        name=c,
                        line=dict(color=palette[idx % len(palette)], width=1.6),
                        yaxis="y2",
                    )
                )

            fig_pg.update_layout(
                **theme.chart_layout(
                    height=theme.H_HERO,
                    xaxis=dict(domain=[0.0, 1.0]),
                    yaxis=dict(title="BTC close", side="left"),
                    yaxis2=dict(
                        title="z-scored signals",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        zeroline=False,
                        ticks="",
                        linecolor=theme.COLORS["axis"],
                    ),
                )
            )

            st.plotly_chart(fig_pg, width="stretch")

    with pg_right:
        st.markdown("#### Current base-signal snapshot")
        last_row = model8_df_full.iloc[-1]
        snap = {
            "signal": [],
            "value": [],
            "missing": [],
        }
        for c in ["signal_1", "signal_2", "signal_3", "signal_4", "signal_6"]:
            if c in model8_df_full.columns:
                snap["signal"].append(c)
                snap["value"].append(last_row.get(c, float("nan")))
                snap["missing"].append(int(last_row.get(f"{c}_missing", 0)))
        if snap["signal"]:
            st.dataframe(pd.DataFrame(snap), use_container_width=True, height=260)
        else:
            st.info("No base-model signal columns found.")

        st.markdown("#### Play ideas")
        st.markdown(
            "- Compare how each base model behaves in different regimes.\n"
            "- Increase smoothing to see slower structural trends.\n"
            "- Shorten lookback in the sidebar to focus on recent market conditions."
        )

st.caption(
    "Research system for educational use. Signals are computed on fully closed daily UTC candles."
)

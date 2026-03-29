import plotly.graph_objects as go
import pandas as pd


def btc_price_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="BTC Price"
    ))

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig


def signal_overlay(price_df, signal_df):
    """
    Robust overlay:
    - Safely aligns signal dates with price dates
    - Never crashes on missing timestamps
    """

    # Ensure datetime & UTC
    price_df = price_df.copy()
    signal_df = signal_df.copy()

    price_df["date"] = pd.to_datetime(price_df["date"], utc=True)
    signal_df["date"] = pd.to_datetime(signal_df["date"], utc=True)

    # Merge signals onto prices
    merged = price_df.merge(
        signal_df[["date", "final_signal"]],
        on="date",
        how="left"
    )

    fig = btc_price_chart(price_df)

    # LONG signals
    longs = merged[merged["final_signal"] == "LONG"]
    shorts = merged[merged["final_signal"] == "SHORT"]

    if not longs.empty:
        fig.add_trace(go.Scatter(
            x=longs["date"],
            y=longs["close"],
            mode="markers",
            marker=dict(color="lime", size=7, symbol="triangle-up"),
            name="LONG"
        ))

    if not shorts.empty:
        fig.add_trace(go.Scatter(
            x=shorts["date"],
            y=shorts["close"],
            mode="markers",
            marker=dict(color="red", size=7, symbol="triangle-down"),
            name="SHORT"
        ))

    return fig

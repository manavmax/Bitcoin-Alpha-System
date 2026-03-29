import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


SIGNAL_PATH = "model8_meta_classifier/results/model8_final_signal.csv"
PRICE_PATH = "data/raw/btc_price_daily.csv"
OUT_PDF = "daily_report.pdf"
OUT_PNG = "model8_meta_classifier/reports/equity_curve.png"


def load_data():
    sig = pd.read_csv(SIGNAL_PATH, parse_dates=["date"])

    price = pd.read_csv(PRICE_PATH)
    if "date" in price.columns:
        ts_col = "date"
    else:
        ts_col = "open_time"
    price["date"] = pd.to_datetime(price[ts_col], utc=True)
    price = price.sort_values("date")

    df = price[["date", "close"]].merge(
        sig, on="date", how="inner"
    ).sort_values("date")
    return df


def compute_backtest(df: pd.DataFrame, tau: float = 0.6) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change().fillna(0.0)

    df["effective_signal"] = np.where(
        (df["final_signal"].isin(["LONG", "SHORT"])) & (df["confidence"] >= tau),
        df["final_signal"],
        "NO_TRADE",
    )
    df["position"] = 0
    df.loc[df["effective_signal"] == "LONG", "position"] = 1
    df.loc[df["effective_signal"] == "SHORT", "position"] = -1

    df["strategy_ret"] = df["position"].shift(1).fillna(0.0) * df["ret"]
    df["bh_ret"] = df["ret"]

    df["strategy_equity"] = (1.0 + df["strategy_ret"]).cumprod()
    df["bh_equity"] = (1.0 + df["bh_ret"]).cumprod()
    return df


def annualized_sharpe(returns, freq=252):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * (freq ** 0.5)


def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return float(drawdown.min())


def plot_equity_curve(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(df["date"], df["strategy_equity"], label="Model 8 Strategy", color="#00aa88")
    plt.plot(df["date"], df["bh_equity"], label="Buy & Hold BTC", color="#6666aa", linestyle="--")
    plt.legend(loc="best")
    plt.title("Equity Curve — Model 8 vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df = load_data()
    bt_df = compute_backtest(df, tau=0.6)

    latest = bt_df.iloc[-1]

    strategy_total = bt_df["strategy_equity"].iloc[-1] - 1.0
    bh_total = bt_df["bh_equity"].iloc[-1] - 1.0

    strat_sharpe = annualized_sharpe(bt_df["strategy_ret"])
    bh_sharpe = annualized_sharpe(bt_df["bh_ret"])

    strat_dd = max_drawdown(bt_df["strategy_equity"])
    bh_dd = max_drawdown(bt_df["bh_equity"])

    trades = (bt_df["position"].shift(1) != 0).sum()
    wins = (bt_df["strategy_ret"] > 0).sum()
    win_rate = wins / max(trades, 1)
    coverage = (bt_df["position"] != 0).mean()

    plot_equity_curve(bt_df, OUT_PNG)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionHeader", parent=styles["Heading2"], spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, leading=11))

    doc = SimpleDocTemplate(OUT_PDF, pagesize=A4)
    story = []

    story.append(Paragraph("Bitcoin Market Intelligence — Model 8", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Executive Summary", styles["SectionHeader"]))
    story.append(
        Paragraph(
            f"Date: {latest['date'].date()}<br/>"
            f"Final Signal (T+1): <b>{latest['final_signal']}</b><br/>"
            f"Confidence: {latest['confidence']:.2f}<br/>"
            f"Volatility Regime: {int(latest['vol_regime'])}<br/>"
            f"Macro Regime: {int(latest['macro_regime'])}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Strategy Snapshot (τ = 0.60)", styles["SectionHeader"]))
    story.append(
        Paragraph(
            f"Model 8 Strategy Total Return: {strategy_total*100:.1f}%<br/>"
            f"Buy & Hold BTC Total Return: {bh_total*100:.1f}%<br/>"
            f"Strategy Sharpe: {strat_sharpe:.2f} &nbsp;&nbsp; | &nbsp;&nbsp; "
            f"Buy & Hold Sharpe: {bh_sharpe:.2f}<br/>"
            f"Max Drawdown (Strategy): {strat_dd*100:.1f}% &nbsp;&nbsp; | &nbsp;&nbsp; "
            f"Max Drawdown (Buy & Hold): {bh_dd*100:.1f}%<br/>"
            f"Win Rate: {win_rate*100:.1f}% &nbsp;&nbsp; | &nbsp;&nbsp; "
            f"Trade Coverage: {coverage*100:.1f}% (fraction of days with a position)",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Equity Curve", styles["SectionHeader"]))
    story.append(Image(OUT_PNG, width=15 * cm, height=7 * cm))
    story.append(Spacer(1, 0.4 * cm))

    recent = bt_df.tail(10).copy()
    recent["date_str"] = recent["date"].dt.strftime("%Y-%m-%d")
    rows = []
    for _, r in recent.iterrows():
        rows.append(
            f"{r['date_str']}: signal={r['final_signal']}, "
            f"eff={r['effective_signal']}, "
            f"conf={r['confidence']:.2f}, "
            f"ret={r['strategy_ret']*100:.2f}%"
        )

    story.append(Paragraph("Recent Signal & Trade History (last 10 days)", styles["SectionHeader"]))
    story.append(
        Paragraph("<br/>".join(rows), styles["Small"])
    )
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Methodology Note", styles["SectionHeader"]))
    story.append(
        Paragraph(
            "Model 8 is a regime-aware meta-ensemble that combines volatility (Model 2), "
            "price dynamics (Model 1), derivatives flow (Model 3), on-chain fundamentals "
            "(Model 4), and macro/liquidity (Model 6). A regime selector (8A) first filters "
            "for tradable environments based on volatility and macro regimes, and a "
            "directional classifier (8B) then chooses LONG / SHORT when confidence exceeds τ. "
            "Performance metrics are computed on daily close-to-close returns.",
            styles["Small"],
        )
    )

    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "This document is for research and informational purposes only. "
            "Not investment advice.",
            styles["Small"],
        )
    )

    doc.build(story)
    print("📄 PDF report generated with performance summary and equity curve")


if __name__ == "__main__":
    main()

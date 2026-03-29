import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    """
    Minimal .env loader (no external deps).
    - Supports lines like KEY=VALUE
    - Ignores comments and blank lines
    - Does not override already-set environment variables
    """
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def run(description: str, script_path: Path) -> None:
    """
    Run a Python script as a subprocess with logging.
    """
    cmd = [sys.executable, str(script_path)]
    print(f"\n▶ {description}")
    print(f"   $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_optional(description: str, script_path: Path, env_var: str | None = None) -> None:
    """
    Run a Python script, but treat failures or missing env vars as non-fatal.

    Useful for optional data sources (e.g. Coinalyze, premium APIs) where
    historical data may already exist and the system should still run.
    """
    if env_var and not os.getenv(env_var):
        print(
            f"\n⚠ Skipping optional step '{description}' "
            f"because environment variable {env_var} is not set."
        )
        return

    cmd = [sys.executable, str(script_path)]
    print(f"\n▶ {description} (optional)")
    print(f"   $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"\n⚠ Optional step failed but pipeline will continue: "
            f"{description} | exit code={exc.returncode}"
        )
        return


if __name__ == "__main__":
    print("========================================")
    print("   Regime-Aware BTC Daily Pipeline")
    print("========================================")

    load_dotenv(BASE_DIR / ".env")

    # ------------------------------------
    # 1. Core data ingestion (multi-source)
    # ------------------------------------
    run(
        "Global data fetch (price, volatility, derivatives, macro, on-chain with fallbacks)",
        BASE_DIR / "src" / "data_fetch" / "fetch_data.py",
    )

    run(
        "Sync latest BTC daily OHLCV into model-specific price files",
        BASE_DIR / "src" / "data_fetch" / "sync_btc_daily_to_models.py",
    )

    # Additional on-chain fundamentals for Model 4 (Blockchain.com)
    run(
        "On-chain fundamentals from Blockchain.com (Model 4 raw inputs)",
        BASE_DIR
        / "model_4_onchain_fundamentals"
        / "src"
        / "fetch_blockchain_com.py",
    )
    run_optional(
        "On-chain fundamentals from CoinMetrics (fallback for missing Blockchain.com days)",
        BASE_DIR / "model_4_onchain_fundamentals" / "src" / "fetch_coinmetrics_onchain.py",
        env_var="COINMETRICS_API_KEY",
    )

    # Derivatives / Coinalyze for Model 3 (optional: requires COINALYZE_API_KEY)
    run_optional(
        "Derivatives & flow data from Coinalyze (Model 3 raw inputs)",
        BASE_DIR / "model_3_derivatives_flow" / "src" / "fetch_coinalyze.py",
        env_var="COINALYZE_API_KEY",
    )
    run_optional(
        "Derivatives from CoinMetrics (fallback metrics where available)",
        BASE_DIR / "model_3_derivatives_flow" / "src" / "fetch_coinmetrics_derivatives.py",
        env_var="COINMETRICS_API_KEY",
    )
    run(
        "Derivatives long/short ratio from Binance (tail fallback)",
        BASE_DIR / "model_3_derivatives_flow" / "src" / "fetch_binance_long_short_ratio.py",
    )
    run_optional(
        "Aggregate Coinalyze to daily features (Model 3)",
        BASE_DIR
        / "model_3_derivatives_flow"
        / "src"
        / "aggregate_coinalyze_daily.py",
    )
    run_optional(
        "Prepare Model 3 train/test dataset from latest Coinalyze features",
        BASE_DIR / "model_3_derivatives_flow" / "src" / "prepare_model3_data.py",
        env_var="COINALYZE_API_KEY",
    )
    run_optional(
        "Train Model 3 TCN and write latest predictions",
        BASE_DIR / "model_3_derivatives_flow" / "src" / "train_model3_tcn.py",
        env_var="COINALYZE_API_KEY",
    )
    run_optional(
        "Train Model 3 CNN-LSTM and write latest predictions",
        BASE_DIR / "model_3_derivatives_flow" / "src" / "train_model3_cnn_lstm.py",
        env_var="COINALYZE_API_KEY",
    )

    # Macro & liquidity (FRED) for Model 6
    run(
        "Macro & liquidity factors from FRED (Model 6 raw inputs)",
        BASE_DIR / "model_6_macro_liquidity" / "src" / "fetch_fred_data.py",
    )
    run(
        "Build Model 6 macro feature table from raw FRED data",
        BASE_DIR / "model_6_macro_liquidity" / "src" / "prepare_model6_features.py",
    )

    # ------------------------------------
    # 2. Base model ensembles (1,2,3,4,6)
    #    Assumes models are already trained.
    # ------------------------------------
    run(
        "Build Model 1 daily features parquet from latest BTC OHLCV",
        BASE_DIR / "model_1_price_dynamics" / "src" / "prepare_data.py",
    )
    run(
        "Model 1 — Price dynamics ensemble (LSTM + TCN + N-BEATS)",
        BASE_DIR / "model_1_price_dynamics" / "src" / "model1_ensemble.py",
    )

    run(
        "Model 2 — GARCH volatility update",
        BASE_DIR / "model_2_volatility_risk" / "src" / "train_garch.py",
    )
    run(
        "Model 2 — compute volatility feature table",
        BASE_DIR / "model_2_volatility_risk" / "src" / "compute_volatility_features.py",
    )
    run(
        "Model 2 — LSTM volatility inference (no retraining)",
        BASE_DIR / "model_2_volatility_risk" / "src" / "infer_volatility_lstm.py",
    )
    run(
        "Model 2 — Volatility ensemble (GARCH + LSTM)",
        BASE_DIR / "model_2_volatility_risk" / "src" / "model2_ensemble.py",
    )

    run(
        "Model 4 — build on-chain feature table from raw metrics",
        BASE_DIR / "model_4_onchain_fundamentals" / "src" / "prepare_model4_features.py",
    )
    run(
        "Model 3 — Derivatives flow ensemble (TCN + CNN-LSTM)",
        BASE_DIR / "model_3_derivatives_flow" / "src" / "model3_ensemble.py",
    )

    run(
        "Model 4 — On-chain fundamentals ensemble (LSTM + Transformer)",
        BASE_DIR / "model_4_onchain_fundamentals" / "src" / "model4_ensemble.py",
    )

    run(
        "Model 6 — Macro & liquidity ensemble (LSTM + Tree)",
        BASE_DIR / "model_6_macro_liquidity" / "src" / "model6_ensemble.py",
    )

    # ------------------------------------
    # 3. Meta-ensemble (Model 8)
    # ------------------------------------
    run(
        "Build unified Model 8 dataset from all base-model signals",
        BASE_DIR
        / "model8_meta_classifier"
        / "src"
        / "build_dataset.py",
    )

    # Assumes Model 8A / 8B have already been trained and saved.
    run(
        "Run Model 8 (regime selector 8A + directional 8B) and save final signals",
        BASE_DIR
        / "model8_meta_classifier"
        / "src"
        / "model8_final.py",
    )

    # ------------------------------------
    # 4. Reporting (PDF + UI artifacts)
    # ------------------------------------
    run(
        "Generate daily PDF research report for latest signal",
        BASE_DIR
        / "model8_meta_classifier"
        / "reports"
        / "generate_daily_report.py",
    )

    print("\n✅ Daily regime-aware pipeline completed successfully.")
    print("   - Base models 1,2,3,4,6 updated")
    print("   - Model 8 final signals refreshed")
    print("   - PDF report generated")


import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run(cmd):
    print(f"▶ Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    print("📥 Fetching latest data")

    # 1. Ingestion
    run([sys.executable, str(BASE_DIR / "run_ingestion.py")])

    # 2. Model 8 inference
    run([
        sys.executable,
        str(BASE_DIR.parent / "src" / "model8_final.py")
    ])

    print("✅ Daily inference completed")

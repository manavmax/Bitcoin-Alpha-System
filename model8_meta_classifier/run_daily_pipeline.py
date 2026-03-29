# model8_meta_classifier/run_daily_pipeline.py

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

def run(file):
    subprocess.run([sys.executable, str(file)], check=True)

if __name__ == "__main__":

    print("STEP 1: Fetching data")
    run(BASE / "data_pipeline" / "run_ingestion.py")
    print("✅ Data ingestion complete")

    # ❌ STEP 2 REMOVED — base models already trained

    print("STEP 2: Running Model 8 inference")
    run(BASE / "model8_backend" / "run_model8.py")
    print("✅ Model 8 inference complete")

    print("DONE: Daily inference pipeline finished successfully")

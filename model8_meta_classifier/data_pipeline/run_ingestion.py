from fetch_btc import fetch_btc_daily
from fetch_macro import fetch_macro
import pandas as pd

btc = fetch_btc_daily()
macro = fetch_macro()

macro = macro.set_index("date").resample("1D").ffill().reset_index()

btc.to_csv("model8_meta_classifier/data/btc_price_daily.csv", index=False)
macro.to_csv("model8_meta_classifier/data/macro_features.csv", index=False)

print("✅ Data ingestion complete")

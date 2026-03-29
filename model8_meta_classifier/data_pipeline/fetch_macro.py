import pandas as pd
from fredapi import Fred
import os

FRED_KEY = os.getenv("9f51582cc2298729f11bc25e49986271")
fred = Fred(api_key=FRED_KEY)

SERIES = {
    "DGS10": "us_10y",
    "DGS2": "us_2y",
    "FEDFUNDS": "fed_funds"
}

def fetch_macro():
    dfs = []
    for code, name in SERIES.items():
        s = fred.get_series(code)
        df = s.to_frame(name)
        dfs.append(df)

    macro = pd.concat(dfs, axis=1)
    macro.index.name = "date"
    macro = macro.reset_index()
    macro["date"] = pd.to_datetime(macro["date"], utc=True)

    return macro.sort_values("date")

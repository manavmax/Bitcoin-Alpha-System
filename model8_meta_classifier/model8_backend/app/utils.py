import pandas as pd

def normalize_date(df):
    if "date" not in df.columns:
        raise RuntimeError("❌ No date column found")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    return df

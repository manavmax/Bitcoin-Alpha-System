import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============================
# CONFIG
# ============================
RAW_DUMP_DIR = Path("external_data/reddit_pushshift")
SAVE_DIR = Path("model_5_sentiment_narrative/raw/reddit")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = SAVE_DIR / "reddit_daily_posts.csv"

SUBREDDITS = {"bitcoin", "btc", "cryptocurrency"}
KEYWORDS = ("bitcoin", "btc")

# ============================
# PIPELINE
# ============================
print("📥 Loading Reddit dump files...")

daily = defaultdict(lambda: {
    "post_count": 0,
    "total_score": 0,
    "total_comments": 0,
    "texts": []
})

files = sorted(RAW_DUMP_DIR.glob("RS_*.csv"))
if not files:
    raise RuntimeError("❌ No Reddit dump files found")

for file in files:
    print(f"➡️ Processing {file.name}")
    df = pd.read_csv(file, usecols=[
        "created_utc", "subreddit", "title", "selftext", "score", "num_comments"
    ], low_memory=False)

    df["subreddit"] = df["subreddit"].str.lower()
    df = df[df["subreddit"].isin(SUBREDDITS)]

    text = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.lower()
    mask = text.str.contains("bitcoin") | text.str.contains("btc")
    df = df[mask]

    df["date"] = pd.to_datetime(df["created_utc"], unit="s").dt.date

    for _, r in df.iterrows():
        d = r["date"]
        daily[d]["post_count"] += 1
        daily[d]["total_score"] += r["score"]
        daily[d]["total_comments"] += r["num_comments"]
        daily[d]["texts"].append(r["title"])

# ============================
# FINAL DATAFRAME
# ============================
rows = []
for d, v in daily.items():
    if v["post_count"] == 0:
        continue

    rows.append({
        "date": d,
        "post_count": v["post_count"],
        "avg_score": v["total_score"] / v["post_count"],
        "avg_comments": v["total_comments"] / v["post_count"],
        "total_score": v["total_score"],
        "text_blob": " ".join(v["texts"])
    })

df_out = pd.DataFrame(rows).sort_values("date")
df_out.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Reddit daily data saved → {OUTPUT_FILE}")
print(f"📊 Days covered: {len(df_out)}")

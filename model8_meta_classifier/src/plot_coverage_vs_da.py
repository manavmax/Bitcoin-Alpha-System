# model8_meta_classifier/src/plot_coverage_vs_da.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "model8_meta_classifier/results/model8_regime_aware_final.csv"
)

thresholds = [0.55, 0.60, 0.65, 0.70]
results = []

for t in thresholds:
    f = df[df["confidence"] >= t]
    if len(f) == 0:
        continue
    da = (f["pred_class"] == f["true_label"]).mean()
    coverage = len(f) / len(df)
    results.append((t, coverage, da))

res = pd.DataFrame(results, columns=["threshold", "coverage", "DA"])

plt.figure(figsize=(7,5))
plt.plot(res["coverage"], res["DA"], marker="o")
for _, r in res.iterrows():
    plt.text(r.coverage, r.DA, f"τ={r.threshold}")

plt.xlabel("Coverage (fraction of days traded)")
plt.ylabel("Directional Accuracy")
plt.title("Model 8: Coverage vs Accuracy Trade-off")
plt.grid(True)
plt.tight_layout()
plt.savefig("model8_meta_classifier/results/coverage_vs_da.png")
plt.show()

import pandas as pd
import numpy as np

df = pd.read_csv("model8_meta_classifier/results/model8_predictions.csv")

results = []

for tau in [0.6, 0.65, 0.7]:
    gated = df[df["confidence"] >= tau]

    if len(gated) == 0:
        continue

    da = (np.sign(gated["future_ret"]) == (gated["pred_class"]*2-1)).mean()
    coverage = len(gated) / len(df)

    results.append({
        "threshold": tau,
        "coverage": coverage,
        "directional_accuracy": da
    })

res = pd.DataFrame(results)
res.to_csv(
    "model8_meta_classifier/results/model8_confidence_curve.csv",
    index=False
)

print("\n📊 CONFIDENCE GATING RESULTS\n")
print(res)

# model8_meta_classifier/src/plot_regime_heatmap.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    "model8_meta_classifier/results/model8_regime_aware_final.csv"
)

pivot = (
    df.groupby(["vol_regime", "macro_regime"])
      .apply(lambda x: (x["pred_class"] == x["true_label"]).mean())
      .unstack()
)

plt.figure(figsize=(6,4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0.5)
plt.title("Directional Accuracy by Market Regime")
plt.xlabel("Macro Regime")
plt.ylabel("Volatility Regime")
plt.tight_layout()
plt.savefig("model8_meta_classifier/results/regime_heatmap.png")
plt.show()

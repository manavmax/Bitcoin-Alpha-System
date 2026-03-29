# model_3/src/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class BTCSequenceDataset(Dataset):
    def __init__(self, seq_len=30):
        self.seq_len = seq_len

        base_dir = Path(__file__).resolve().parents[1]
        data_path = base_dir / "data_processed" / "features.parquet"

        df = pd.read_parquet(data_path)

        if len(df) <= seq_len:
            raise ValueError("Not enough data to create sequences")

        # TARGET: next-day return
        self.target = df["target_return"].values.astype(np.float32)

        # FEATURES
        features = df.drop(columns=[
            "timestamp",
            "close",
            "log_return",
            "target_return"
        ])

        features = features.select_dtypes(include=[np.number])

        scaler = StandardScaler()
        self.features = scaler.fit_transform(features).astype(np.float32)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx:idx + self.seq_len]
        y = self.target[idx + self.seq_len]

        return torch.from_numpy(X), torch.tensor(y)

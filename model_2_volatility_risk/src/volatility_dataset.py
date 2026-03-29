# model_2_volatility_risk/src/volatility_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np

class VolatilitySequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_len]
        y_t = self.y[idx+self.seq_len]
        return torch.tensor(X_seq), torch.tensor(y_t)

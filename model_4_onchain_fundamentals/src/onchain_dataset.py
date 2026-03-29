import torch
from torch.utils.data import Dataset

class OnchainSequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

        if len(self.X) <= self.seq_len:
            raise ValueError(
                f"❌ Not enough data: samples={len(self.X)}, seq_len={self.seq_len}"
            )

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)
        )

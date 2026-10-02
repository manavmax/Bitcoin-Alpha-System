# model_2_volatility_risk/src/volatility_lstm.py

import torch
import torch.nn as nn

class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # squeeze(-1) keeps the batch dimension for single-sample final batches.
        return self.fc(out[:, -1, :]).squeeze(-1)

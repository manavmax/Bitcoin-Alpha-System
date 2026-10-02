# model_3/src/model.py

import torch
import torch.nn as nn

class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        # squeeze(-1) (not squeeze()) keeps the batch dimension even when the
        # final batch has a single sample; squeeze() would collapse it to a 0-d
        # scalar and break `extend(pred.cpu().numpy())` in the ensemble loop.
        return self.fc(out).squeeze(-1)

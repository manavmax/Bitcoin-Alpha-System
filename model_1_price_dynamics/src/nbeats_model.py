# model_3/src/nbeats_model.py

import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, theta_size)
        )

    def forward(self, x):
        return self.fc(x)


class NBeats(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, input_size + 1, hidden_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.reshape(x.size(0), -1)

        forecast = 0
        residual = x

        for block in self.blocks:
            theta = block(residual)
            backcast = theta[:, :-1]
            forecast_step = theta[:, -1:]
            residual = residual - backcast
            forecast = forecast + forecast_step

        return forecast.squeeze()

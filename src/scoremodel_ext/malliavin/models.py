import torch
import torch.nn as nn


class TimeScoreMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, y):
        if t.ndim == 1:
            t = t[:, None]
        if y.ndim == 1:
            y = y[:, None]
        return self.net(torch.cat([t, y], dim=1))
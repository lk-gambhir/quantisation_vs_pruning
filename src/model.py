"""The single model compared in this study: a 784 -> 256 -> 128 -> 10 MLP."""
import torch.nn as nn

from src import config


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        first, second = config.hidden
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, 10),
        )

    def forward(self, x):
        return self.layers(self.flatten(x))


def countparams(model):
    return sum(p.numel() for p in model.parameters())

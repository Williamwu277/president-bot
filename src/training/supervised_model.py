from functools import cache
from pathlib import Path

import torch
from torch import nn

from .encoder import MOVE_COUNT, STATE_SIZE

HIDDEN_SIZE = 256
MODEL_PATH = Path("models/supervised_model.pt")


class SupervisedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, MOVE_COUNT),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@cache
def load_model(model_path: Path = MODEL_PATH) -> SupervisedModel:
    model = SupervisedModel()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model

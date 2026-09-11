from functools import cache
from pathlib import Path

import torch
from torch import nn

from .encoder import MOVE_COUNT, STATE_SIZE

HIDDEN_SIZE = 256
SUPERVISED_MODEL_PATH = Path("models/supervised_model.pt")
REINFORCEMENT_MODEL_PATH = Path("models/reinforcement_model.pt")


class PolicyValueModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(HIDDEN_SIZE, MOVE_COUNT)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)

        policy_logits = self.policy_head(hidden)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)

        return policy_logits, value


@cache
def load_playing_model(model_path: Path) -> PolicyValueModel:
    model = PolicyValueModel()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model

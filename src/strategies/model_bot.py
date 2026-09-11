from pathlib import Path

import torch

from ..president import Move, Player, PlayerView
from ..training.encoder import decode_move, encode_view
from ..training.model import (
    REINFORCEMENT_MODEL_PATH,
    load_playing_model,
)


class ModelBot(Player):
    """
    Strategy using trained model inference to find the best move.
    """

    def __init__(self, name: str, model_path: Path = REINFORCEMENT_MODEL_PATH):
        super().__init__(name)
        self.model = load_playing_model(model_path)

    def make_move(self, view: PlayerView) -> Move | None:
        features, legal_moves = encode_view(view)

        features = features.unsqueeze(0)
        legal_moves = legal_moves.unsqueeze(0)

        with torch.inference_mode():
            policy_logits, _ = self.model(features)

            masked_logits = policy_logits.masked_fill(
                ~legal_moves, torch.finfo(policy_logits.dtype).min
            )

            move_id = masked_logits.argmax(dim=1).item()

        return decode_move(move_id, list(view.possible_moves))

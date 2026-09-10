from pathlib import Path

import torch

from ..president import Move, Player, PlayerView
from ..training.encoder import decode_move, encode_view
from ..training.supervised_model import MODEL_PATH, load_model


class ModelBot(Player):
    """
    Strategy using trained model inference to find the best move.
    """

    def __init__(self, name: str, model_path: Path = MODEL_PATH):
        super().__init__(name)
        self.model = load_model(model_path)

    def make_move(self, view: PlayerView) -> Move | None:
        features, legal_moves = encode_view(view)

        features = features.unsqueeze(0)
        legal_moves = legal_moves.unsqueeze(0)

        with torch.inference_mode():
            logits = self.model(features)

            masked_logits = logits.masked_fill(
                ~legal_moves, torch.finfo(logits.dtype).min
            )

            move_id = masked_logits.argmax(dim=1).item()

        return decode_move(move_id, list(view.possible_moves))

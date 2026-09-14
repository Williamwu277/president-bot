from pathlib import Path

import torch

from ..president import Move, Player, PlayerView
from ..training.encoder import decode_move, encode_view
from ..training.model import PolicyValueModel, load_playing_model


def analyze_view(
    model: PolicyValueModel, view: PlayerView, limit: int
) -> tuple[list[tuple[Move | None, float]], float]:
    """
    Return ranked legal moves and the value head's state win probability.

    The value head predicts the outcome from the current state, before any of
    the recommended moves is made. Its [-1, 1] output is converted to [0, 1].
    """
    features, legal_moves = encode_view(view)
    legal_move_count = int(legal_moves.sum().item())
    if legal_move_count == 0:
        raise ValueError("No legal moves available")

    with torch.inference_mode():
        policy_logits, predicted_value = model(features.unsqueeze(0))
        masked_logits = policy_logits.masked_fill(
            ~legal_moves.unsqueeze(0), torch.finfo(policy_logits.dtype).min
        )
        probabilities = masked_logits.softmax(dim=1)
        top_probabilities, top_move_ids = probabilities.topk(
            min(limit, legal_move_count), dim=1
        )

    possible_moves = list(view.possible_moves)
    ranked_moves = [
        (decode_move(move_id.item(), possible_moves), probability.item())
        for move_id, probability in zip(top_move_ids[0], top_probabilities[0])
    ]
    win_probability = (predicted_value.item() + 1.0) / 2.0
    return ranked_moves, win_probability


class ModelBot(Player):
    """
    Strategy using trained model inference to find the best move.
    """

    def __init__(self, name: str, model_path: Path):
        super().__init__(name)
        self.model = load_playing_model(model_path)

    def make_move(self, view: PlayerView) -> Move | None:
        ranked_moves, _ = analyze_view(self.model, view, limit=1)
        return ranked_moves[0][0]

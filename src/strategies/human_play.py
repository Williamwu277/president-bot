from pathlib import Path

from ..president import Move, Player, PlayerView
from ..training.model import load_playing_model
from .model_bot import analyze_view

ADVISOR_MODEL_PATH = Path("final_models/reinforcement_model_supervised_base_30k.pt")


class HumanPlayer(Player):
    """
    Play by utilising human input.
    """

    def __init__(self, name: str, advisor: bool = False):
        super().__init__(name)
        self.advisor_model = load_playing_model(ADVISOR_MODEL_PATH) if advisor else None

    def make_move(self, view: PlayerView) -> Move | None:
        print(f"Your cards: {view.hand}")
        print(f"Cards remaining by player id: {view.cards_remaining}")
        print(f"Current move to beat: {view.current_move}")
        print("Possible moves:")
        for move_id, move in enumerate(view.possible_moves, start=1):
            print(f"{move_id}: {move}")

        if self.advisor_model is not None:
            recommendations, win_probability = analyze_view(
                self.advisor_model, view, limit=5
            )
            print("Model recommendations:")
            print(f"Value-head win probability: {win_probability:.2%}")
            for rank, (move, probability) in enumerate(recommendations, start=1):
                move_id = 0 if move is None else view.possible_moves.index(move) + 1
                label = "pass" if move is None else str(move)
                print(f"{rank}. [{move_id}] {label}: {probability:.2%}")

        chosen_move = int(input("Choose your move id (0 to pass): "))
        if chosen_move == 0:
            return None
        elif chosen_move > len(view.possible_moves) or chosen_move < 0:
            return self.make_move(view)
        return view.possible_moves[chosen_move - 1]

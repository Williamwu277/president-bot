from random import randint

from president import Move, Player, PlayerView


class RandomBot(Player):
    """
    Strategy choosing uniformly from the currently legal moves.
    """

    def make_move(self, view: PlayerView) -> Move | None:
        if not view.possible_moves:
            return None
        move_index = randint(0, len(view.possible_moves) - 1)
        return view.possible_moves[move_index]

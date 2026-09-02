"""
Ready-to-use player strategies for President.
"""

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


class MinimalCardBot(Player):
    """
    Strategy choosing 'smallest' valued legal response.
    """

    def make_move(self, view: PlayerView) -> Move | None:
        if not view.possible_moves:
            return None
        
        if view.current_move:
            # If responding to a move, choose the smallest response
            sorted_moves = sorted(
                view.possible_moves,
                key=lambda move: (len(move), max(move.cards[0].rank.value, move.cards[-1].rank.value))
            )
            return sorted_moves[0]
        else:
            # Otherwise, choose the smallest response that kills off the most cards
            sorted_moves = sorted(
                view.possible_moves,
                key=lambda move: (min(move.cards[0].rank.value, move.cards[-1].rank.value), -len(move))
            )
            return sorted_moves[0]


class HumanPlayer(Player):
    """
    Play by utilising human input.
    """

    def make_move(self, view: PlayerView) -> Move | None:
        print(f"Your cards: {view.hand}")
        print(f"Cards remaining by player id: {view.cards_remaining}")
        print(f"Current move to beat: {view.current_move}")
        print("Possible moves:")
        for move_id, move in enumerate(view.possible_moves, start=1):
            print(f"{move_id}: {move}")
        chosen_move = int(input("Choose your move id (0 to pass): "))
        if chosen_move == 0:
            return None
        elif chosen_move > len(view.possible_moves) or chosen_move < 0:
            return self.make_move(view)
        return view.possible_moves[chosen_move - 1]

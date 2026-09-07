from ..president import Move, Player, PlayerView


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

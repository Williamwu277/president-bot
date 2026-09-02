from president import Move, Player, PlayerView


class MinimalCardBot(Player):
    """
    Strategy choosing 'smallest' valued legal response.

    Note: Currently extremely scuffed with combination moves.
    E.g. It will play (3♡ 3♣ 2♠ 2♢ 2♡) without missing a beat
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

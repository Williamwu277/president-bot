from src.president import Card, Move, MoveType, Player, PlayerView, Rank, Suit


class ScriptedPlayer(Player):
    """
    Player that returns predetermined moves and records its views.
    """

    def __init__(self, name: str, *moves: Move | None):
        super().__init__(name)
        self.moves = iter(moves)
        self.views: list[PlayerView] = []

    def make_move(self, view: PlayerView) -> Move | None:
        self.views.append(view)
        return next(self.moves)


def make_cards(rank: Rank, count: int) -> list[Card]:
    """
    Make a group of same-rank cards with distinct suits.
    """
    return [Card(rank, suit) for suit in list(Suit)[:count]]


def make_sequence(start: Rank, count: int = 5) -> list[Card]:
    """
    Make consecutive cards starting at the given rank.
    """
    return [
        Card(Rank(rank_value), Suit.SPADES)
        for rank_value in range(start.value, start.value + count)
    ]


def get_move_keys(moves: list[Move], move_type: MoveType) -> set[tuple[Rank, ...]]:
    """
    Return the rank keys for moves of one type.
    """
    return {
        move.get_key()
        for move in moves
        if move.move_type is move_type
    }

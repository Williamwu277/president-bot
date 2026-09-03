from president import (
    Move, 
    Hand, 
    Player, 
    PlayerView, 
    MoveKey, 
    HandKey, 
    get_full_deck,
    FULL_DECK_SIZE
)
from random import randint


GameState = tuple[HandKey, HandKey, MoveKey | None]


def get_state_key(hand: Hand, other_hand: Hand, previous_move: Move | None) -> GameState:
    """
    Return a hashable representation of the current game-state
    """
    return (
        hand.get_key(),
        other_hand.get_key(),
        previous_move.get_key() if previous_move is not None else None
    )


def minimax_solver(
    memoized_results: dict[GameState, bool],
    hand: Hand, 
    other_hand: Hand, 
    previous_move: Move | None
) -> bool:
    """
    Given your current hand, the opponents hand and the most recent 
    move, return if you would win
    """
    if hand.is_empty():
        return True

    state = get_state_key(hand, other_hand, previous_move)

    if state in memoized_results:
        return memoized_results[state]
    
    possible_moves = hand.get_possible_moves(previous_move)

    if previous_move is not None:
        possible_moves.append(None)

    for move in possible_moves:
        hand.remove_cards(move)
        opponent_wins = minimax_solver(memoized_results, other_hand, hand, move)
        hand.add_cards(move)
        if not opponent_wins:
            memoized_results[state] = True
            return True

    memoized_results[state] = False
    return False


def is_move_winning(
    memoized_results: dict[GameState, bool],
    hand: Hand, 
    other_hand: Hand, 
    move: Move | None
) -> bool:
    """
    Determine if `move` is winning if `hand` plays it
    """
    hand.remove_cards(move)
    game_state = get_state_key(other_hand, hand, move)

    if game_state in memoized_results:
        hand.add_cards(move)
        return not memoized_results[game_state]
    
    result = minimax_solver(memoized_results, other_hand, hand, move)
    hand.add_cards(move)
    return not result


def find_winning_moves(
    memoized_results: dict[GameState, bool],
    hand: Hand, 
    other_hand: Hand, 
    previous_move: Move | None
) -> list[Move | None]:
    """
    Finds all winning moves for `hand`
    """
    possible_moves = hand.get_possible_moves(previous_move)

    if previous_move is not None:
        possible_moves.append(None)

    return list(
        filter(
            lambda move: is_move_winning(memoized_results, hand, other_hand, move), 
            possible_moves
        )
    )


class MinimaxBot(Player):
    """
    Optimally play.

    Note: This strategy only works with no hidden information.
    i.e. 2 players 26 cards each.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.other_hand: Hand = Hand(get_full_deck())
        self.memoized_results: dict[GameState, bool] = dict()

    def init_other_hand(self, hand: Hand):
        for card in hand.cards:
            self.other_hand.remove_cards(Move((card,)))

    def make_move(self, view: PlayerView) -> Move | None:
        hand = Hand(list(view.hand))

        if len(self.other_hand) == FULL_DECK_SIZE:
            self.init_other_hand(hand)

        self.other_hand.remove_cards(view.current_move)
        winning_moves = find_winning_moves(
            self.memoized_results,
            hand,
            self.other_hand,
            view.current_move
        )

        if len(winning_moves) > 0:
            move_index = randint(0, len(winning_moves) - 1)
            return winning_moves[move_index]

        if len(view.possible_moves) > 0:
            return view.possible_moves[0]

        return None


"""
hand = Hand([
    Card(Rank.THREE, Suit.SPADES),
    Card(Rank.FOUR, Suit.SPADES),
    Card(Rank.SIX, Suit.SPADES),
    Card(Rank.SEVEN, Suit.SPADES)
])

other_hand = Hand([
    Card(Rank.THREE, Suit.SPADES),
    Card(Rank.SIX, Suit.SPADES),
])

winning_moves = minimax_solver(hand, other_hand, None)
for winning_move in winning_moves:
    print(winning_move)
"""

from random import randint

from ..president import (
    Card,
    Hand,
    HandKey,
    Move,
    MoveKey,
    Player,
    PlayerView,
    get_full_deck,
)
from .minimal_card_bot import get_minimal_card_move

GameState = tuple[HandKey, HandKey, MoveKey | None]


def get_state_key(
    hand: Hand, other_hand: Hand, previous_move: Move | None
) -> GameState:
    """
    Return a hashable representation of the current game-state
    """
    return (
        hand.get_key(),
        other_hand.get_key(),
        previous_move.get_key() if previous_move is not None else None,
    )


def minimax_solver(
    memoized_results: dict[GameState, bool],
    hand: Hand,
    other_hand: Hand,
    previous_move: Move | None,
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
    move: Move | None,
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
    previous_move: Move | None,
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
            possible_moves,
        )
    )


class MinimaxBot(Player):
    """
    Optimally play a two-player game with perfect information.

    Without an explicit opponent hand, assume one complete deck is split
    between the two players and infer the opponent's cards from our own hand.
    """

    def __init__(self, name: str, known_opponent_cards: list[Card] | None = None):
        super().__init__(name)
        if known_opponent_cards is None:
            self.other_hand = Hand(get_full_deck())
            self._opponent_initialized = False
        else:
            self.other_hand = Hand(list(known_opponent_cards))
            self._opponent_initialized = True
        self.memoized_results: dict[GameState, bool] = {}

    def init_other_hand(self, hand: Hand):
        for card in hand.cards:
            self.other_hand.remove_cards(Move((card,)))

    def make_move(self, view: PlayerView) -> Move | None:
        hand = Hand(list(view.hand))

        if not self._opponent_initialized:
            self.init_other_hand(hand)
            self._opponent_initialized = True

        self.other_hand.remove_cards(view.current_move)
        winning_moves = find_winning_moves(
            self.memoized_results, hand, self.other_hand, view.current_move
        )

        if len(winning_moves) > 0:
            move_index = randint(0, len(winning_moves) - 1)
            return winning_moves[move_index]

        # If current trajectory points to a loss, make a best effort move
        # In case the opponent makes an exploitable mistake
        return get_minimal_card_move(view.possible_moves, view.current_move)

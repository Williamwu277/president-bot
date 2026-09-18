from collections import defaultdict
from functools import cache

from ..president import (
    Hand,
    Move,
    get_full_deck,
)

HandTuple = tuple[int, ...]
MoveTuple = tuple[int, ...]
MoveId = int

EMPTY_HAND: HandTuple = (0,) * 13
move_to_tuple: dict[Move, MoveTuple] = {}
moves_by_id: list[MoveTuple] = []
move_id_by_tuple: dict[MoveTuple, MoveId] = {}
move_is_beat_by: defaultdict[MoveId | None, list[MoveId]] = defaultdict(list)


def convert_move_to_tuple(move: Move | None) -> MoveTuple | None:
    if move is None:
        return None
    result = [0] * 13
    for card in move.cards:
        result[card.rank.value - 3] += 1
    return tuple(result)


def convert_hand_to_tuple(hand: Hand) -> HandTuple:
    result = [0] * 13
    for card in hand.cards:
        result[card.rank.value - 3] += 1
    return tuple(result)


def precompute_moves() -> None:
    if move_to_tuple:
        return

    full_hand = Hand(get_full_deck())
    all_moves = full_hand.get_all_moves()

    for move in all_moves:
        move_tuple = convert_move_to_tuple(move)
        assert move_tuple is not None

        move_id = len(moves_by_id)
        move_to_tuple[move] = move_tuple
        moves_by_id.append(move_tuple)
        move_id_by_tuple[move_tuple] = move_id

    for move_a in all_moves:
        move_a_id = move_id_by_tuple[move_to_tuple[move_a]]
        move_is_beat_by[None].append(move_a_id)
        for move_b in all_moves:
            if move_a.beats(move_b):
                move_b_id = move_id_by_tuple[move_to_tuple[move_b]]
                move_is_beat_by[move_b_id].append(move_a_id)

    for previous_move_id in move_is_beat_by:
        move_is_beat_by[previous_move_id].sort(
            key=lambda move_id: -sum(moves_by_id[move_id])
        )


def can_apply_move(hand: HandTuple, move: MoveTuple) -> bool:
    for i in range(13):
        if hand[i] < move[i]:
            return False
    return True


def apply_move(hand: HandTuple, move: MoveTuple) -> HandTuple:
    result = [0] * 13
    for i in range(13):
        result[i] = hand[i] - move[i]
    return tuple(result)


@cache
def get_legal_moves(
    hand: HandTuple, previous_move_id: MoveId | None
) -> tuple[MoveId, ...]:
    return tuple(
        move_id
        for move_id in move_is_beat_by[previous_move_id]
        if can_apply_move(hand, moves_by_id[move_id])
    )


def solve_game(
    starting_hand: Hand,
    starting_opponent: Hand,
) -> bool:
    @cache
    def solve(
        hand: HandTuple,
        opponent_hand: HandTuple,
        previous_move_id: MoveId | None,
    ) -> bool:
        if hand == EMPTY_HAND:
            return True

        for move_id in get_legal_moves(hand, previous_move_id):
            move = moves_by_id[move_id]
            result_hand = apply_move(hand, move)
            if not solve(opponent_hand, result_hand, move_id):
                return True

        return (previous_move_id is not None) and (not solve(opponent_hand, hand, None))

    return solve(
        convert_hand_to_tuple(starting_hand),
        convert_hand_to_tuple(starting_opponent),
        None,
    )


precompute_moves()

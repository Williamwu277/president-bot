import pytest

from src.president import Hand, Move, PlayerView, Rank, TurnRecord, get_full_deck
from src.training.encoder import (
    MOVE_COUNT,
    OPPONENT_SLOT_SIZE,
    decode_move,
    encode_move,
    encode_view,
    move_encoding_map,
)
from tests.helpers import make_cards


def make_view(
    player_count=4,
    *,
    turn_order=None,
    cards_remaining=None,
    current_move=None,
    current_trick=(),
    trick_history=(),
    possible_moves=(),
):
    return PlayerView(
        player_id=0,
        hand=tuple(make_cards(Rank.TWO, 1)),
        turn_order=turn_order or tuple(range(player_count)),
        cards_remaining=cards_remaining or (1,) * player_count,
        current_move=current_move,
        current_trick=current_trick,
        trick_history=trick_history,
        possible_moves=possible_moves,
    )


def test_action_ids():
    assert len(move_encoding_map) == len(set(move_encoding_map.values())) == 217
    assert encode_move(None) == 0


def test_move_round_trip():
    moves = Hand(get_full_deck()).get_all_moves()

    assert all(decode_move(encode_move(move), moves) == move for move in [None, *moves])


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_shapes(player_count):
    state = encode_view(make_view(player_count))

    assert state.features.shape == (950,)
    assert state.legal_moves.shape == (217,)


def test_player_slots():
    view = make_view(3, turn_order=(0, 2), cards_remaining=(1, 0, 1))
    state = encode_view(view)
    first_slot = MOVE_COUNT + 8 + (2 * len(Rank))
    missing_slot = first_slot + (2 * OPPONENT_SLOT_SIZE)

    assert state.features[first_slot : first_slot + 2].tolist() == [1, 0]
    assert state.features[missing_slot : missing_slot + 2].tolist() == [0, 0]


def test_current_trick():
    three = Move(tuple(make_cards(Rank.THREE, 1)))
    five = Move(tuple(make_cards(Rank.FIVE, 1)))
    trick = (
        TurnRecord(2, three),
        TurnRecord(3, None),
        TurnRecord(1, five),
        TurnRecord(2, None),
    )

    features = encode_view(make_view(current_move=five, current_trick=trick)).features

    assert features[MOVE_COUNT : MOVE_COUNT + 4].tolist() == [0, 1, 0, 0]
    assert features[MOVE_COUNT + 4 : MOVE_COUNT + 8].tolist() == [0, 0, 1, 0]


def test_trick_history():
    three = Move(tuple(make_cards(Rank.THREE, 1)))
    five = Move(tuple(make_cards(Rank.FIVE, 1)))
    old_trick = (TurnRecord(1, three), TurnRecord(2, None))
    current_trick = (TurnRecord(2, five), TurnRecord(1, None))
    features = encode_view(
        make_view(
            current_move=five,
            current_trick=current_trick,
            trick_history=(old_trick,),
        )
    ).features
    first_slot = MOVE_COUNT + 8 + (2 * len(Rank))
    second_slot = first_slot + OPPONENT_SLOT_SIZE
    played = 3
    passed = played + len(Rank)

    assert features[first_slot + played + list(Rank).index(Rank.THREE)] == 0.25
    assert features[first_slot + passed + encode_move(five)] == 0.25
    assert features[second_slot + played + list(Rank).index(Rank.FIVE)] == 0.25
    assert features[second_slot + passed + encode_move(three)] == 0.25


def test_teacher_action():
    three = Move(tuple(make_cards(Rank.THREE, 1)))
    hand = Hand(make_cards(Rank.FIVE, 2))
    moves = hand.get_possible_moves(three)
    legal = encode_view(
        make_view(current_move=three, possible_moves=tuple(moves))
    ).legal_moves

    assert all(legal[encode_move(move)] for move in [None, *moves])

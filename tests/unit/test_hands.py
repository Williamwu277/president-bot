import pytest

from src.president import Card, Hand, Move, MoveType, Rank, Suit
from tests.helpers import (
    get_move_keys,
    make_cards,
    make_sequence,
)


def test_key():
    hand = Hand(
        make_cards(Rank.THREE, 2) + make_cards(Rank.FIVE, 1) + make_cards(Rank.TWO, 1)
    )

    assert hand.get_key() == (2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)


def test_add_cards():
    five = Card(Rank.FIVE, Suit.SPADES)
    three = Card(Rank.THREE, Suit.DIAMONDS)
    hand = Hand([five])

    hand.add_cards(Move((three,)))

    assert hand.cards == [three, five]


def test_remove_cards():
    three_spades = Card(Rank.THREE, Suit.SPADES)
    three_diamonds = Card(Rank.THREE, Suit.DIAMONDS)
    hand = Hand([three_spades, three_diamonds])

    hand.remove_cards(Move((three_diamonds,)))

    assert hand.cards == [three_spades]


def test_get_all_same_rank_moves():
    moves = Hand(make_cards(Rank.FIVE, 4)).get_all_moves()

    assert {(move.move_type, move.get_key()) for move in moves} == {
        (MoveType.SINGLE, (Rank.FIVE,)),
        (MoveType.DOUBLE, (Rank.FIVE,) * 2),
        (MoveType.TRIPLE, (Rank.FIVE,) * 3),
        (MoveType.BOMB, (Rank.FIVE,) * 4),
    }


def test_get_all_straights():
    moves = Hand(make_sequence(Rank.THREE, 6)).get_all_moves()

    assert get_move_keys(moves, MoveType.STRAIGHT) == {
        (Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN),
        (Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT),
    }


@pytest.mark.parametrize(
    "ranks",
    [
        pytest.param(
            [Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.EIGHT],
            id="Gap in straight",
        ),
        pytest.param(
            [Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE, Rank.TWO],
            id="Straight cannot contain two",
        ),
    ],
)
def test_get_invalid_straights(ranks):
    hand = Hand([Card(rank, Suit.SPADES) for rank in ranks])

    assert get_move_keys(hand.get_all_moves(), MoveType.STRAIGHT) == set()


def test_get_all_full_house():
    hand = Hand(make_cards(Rank.FIVE, 3) + make_cards(Rank.SEVEN, 2))

    assert get_move_keys(hand.get_all_moves(), MoveType.FULL_HOUSE) == {
        (Rank.FIVE, Rank.FIVE, Rank.FIVE, Rank.SEVEN, Rank.SEVEN)
    }


def test_get_possible_starting_moves():
    hand = Hand(make_cards(Rank.FIVE, 2))

    assert hand.get_possible_moves(None) == hand.get_all_moves()


def test_get_possible_moves():
    hand = Hand(
        make_cards(Rank.THREE, 1) + make_cards(Rank.FIVE, 2) + make_cards(Rank.EIGHT, 4)
    )
    current_move = Move(tuple(make_cards(Rank.FOUR, 1)))

    possible_moves = hand.get_possible_moves(current_move)

    assert {(move.move_type, move.get_key()) for move in possible_moves} == {
        (MoveType.SINGLE, (Rank.FIVE,)),
        (MoveType.SINGLE, (Rank.EIGHT,)),
        (MoveType.BOMB, (Rank.EIGHT,) * 4),
    }

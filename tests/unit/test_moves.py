import pytest

from src.president import Card, Move, MoveType, Rank, Suit
from tests.helpers import make_cards, make_sequence

SINGLE_SM = make_cards(Rank.THREE, 1)
SINGLE_LG = make_cards(Rank.FIVE, 1)
DOUBLE_SM = make_cards(Rank.FOUR, 2)
DOUBLE_LG = make_cards(Rank.SIX, 2)
DOUBLE_XL = make_cards(Rank.ACE, 2)
TRIPLE_SM = make_cards(Rank.FIVE, 3)
TRIPLE_LG = make_cards(Rank.SEVEN, 3)
BOMB_SM = make_cards(Rank.SIX, 4)
BOMB_LG = make_cards(Rank.EIGHT, 4)


@pytest.mark.parametrize(
    ("cards", "expected_move_type"),
    [
        pytest.param([], None, id="Empty move"),
        pytest.param(SINGLE_SM, MoveType.SINGLE, id="Single"),
        pytest.param(DOUBLE_SM, MoveType.DOUBLE, id="Double"),
        pytest.param(TRIPLE_SM, MoveType.TRIPLE, id="Triple"),
        pytest.param(BOMB_SM, MoveType.BOMB, id="Bomb"),
        pytest.param(TRIPLE_SM + DOUBLE_SM, MoveType.FULL_HOUSE, id="Full house"),
        pytest.param(make_sequence(Rank.THREE), MoveType.STRAIGHT, id="Straight"),
        pytest.param(make_sequence(Rank.JACK), None, id="No straight to Rank.TWO"),
        pytest.param(
            [Card(Rank.THREE, Suit.SPADES), Card(Rank.FOUR, Suit.SPADES)],
            None,
            id="Two different cards",
        ),
        pytest.param([Card(Rank.THREE, Suit.SPADES)] * 5, None, id="Too many cards"),
        pytest.param(
            make_sequence(Rank.THREE)[:4] + [Card(Rank.NINE, Suit.SPADES)],
            None,
            id="Invalid straight",
        ),
        pytest.param(TRIPLE_LG + SINGLE_LG, None, id="Invalid full house"),
    ],
)
def test_move_validation(cards, expected_move_type):
    if expected_move_type is None:
        with pytest.raises(ValueError, match="Invalid Move"):
            Move(cards)
        return
    assert Move(cards).move_type is expected_move_type


@pytest.mark.parametrize(
    ("move", "other_move", "is_bigger"),
    [
        pytest.param(SINGLE_LG, SINGLE_SM, True, id="Larger single"),
        pytest.param(DOUBLE_LG, DOUBLE_SM, True, id="Larger double"),
        pytest.param(TRIPLE_LG, TRIPLE_SM, True, id="Larger triple"),
        pytest.param(BOMB_LG, BOMB_SM, False, id="Bombs flush the pile"),
        pytest.param(
            DOUBLE_SM + TRIPLE_LG,
            DOUBLE_LG + TRIPLE_SM,
            True,
            id="Larger full house (Larger primary, smaller secondary)",
        ),
        pytest.param(
            DOUBLE_LG + TRIPLE_SM,
            DOUBLE_SM + TRIPLE_SM,
            True,
            id="Larger full house (Equal primary, larger secondary)",
        ),
        pytest.param(
            make_sequence(Rank.FIVE),
            make_sequence(Rank.THREE),
            True,
            id="Larger straight",
        ),
        pytest.param(DOUBLE_SM, DOUBLE_LG, False, id="Smaller double"),
        pytest.param(
            DOUBLE_LG + TRIPLE_SM,
            DOUBLE_SM + TRIPLE_LG,
            False,
            id="Smaller full house (Smaller primary, larger secondary)",
        ),
        pytest.param(
            BOMB_SM, make_sequence(Rank.TEN), True, id="Bomb anything (but a bomb)"
        ),
        pytest.param(
            make_sequence(Rank.TEN), BOMB_SM, False, id="Non-bomb cannot beat bomb"
        ),
        pytest.param(
            TRIPLE_SM + DOUBLE_XL,
            TRIPLE_LG + DOUBLE_LG,
            False,
            id="Full house compares triple before pair",
        ),
        pytest.param(SINGLE_LG, DOUBLE_SM, False, id="Different non-bomb move types"),
    ],
)
def test_move_comparisons(move, other_move, is_bigger):
    assert Move(move).beats(Move(other_move)) == is_bigger


def test_move_sorts_cards_before_validating_straight():
    cards = list(reversed(make_sequence(Rank.THREE)))

    move = Move(cards)

    assert move.move_type is MoveType.STRAIGHT
    assert move.get_key() == (
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
    )

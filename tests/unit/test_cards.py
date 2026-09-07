import pytest

from src.president import Card, Rank, Suit


@pytest.mark.parametrize(
    ("card", "other_card", "is_bigger"),
    [
        pytest.param(
            Card(Rank.THREE, Suit.SPADES),
            Card(Rank.FOUR, Suit.SPADES),
            False,
            id="Smaller card",
        ),
        pytest.param(
            Card(Rank.FOUR, Suit.SPADES),
            Card(Rank.THREE, Suit.SPADES),
            True,
            id="Bigger card",
        ),
        pytest.param(
            Card(Rank.THREE, Suit.SPADES),
            Card(Rank.THREE, Suit.DIAMONDS),
            False,
            id="Equal rank, different suit",
        ),
        pytest.param(
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES),
            True,
            id="Ace is bigger than king",
        ),
        pytest.param(
            Card(Rank.TWO, Suit.SPADES),
            Card(Rank.ACE, Suit.SPADES),
            True,
            id="Two is bigger than ace",
        ),
    ],
)
def test_card_comparisons(card, other_card, is_bigger):
    """
    The general concern for cards is whether comparisons work
    and if ACE and TWO are the biggest cards.
    """
    assert card.beats(other_card) == is_bigger

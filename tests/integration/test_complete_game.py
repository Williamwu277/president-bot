from src.president import President, Rank
from src.strategies.minimal_card_bot import MinimalCardBot
from tests.helpers import make_cards


def test_complete_multiplayer_game():
    players = [
        MinimalCardBot("Player 0"),
        MinimalCardBot("Player 1"),
        MinimalCardBot("Player 2"),
    ]
    game = President(
        players,
        initial_hands=[
            make_cards(Rank.THREE, 1) + make_cards(Rank.SIX, 1),
            make_cards(Rank.FOUR, 1) + make_cards(Rank.SEVEN, 1),
            make_cards(Rank.FIVE, 1) + make_cards(Rank.EIGHT, 1),
        ],
    )

    standings = game.run()

    assert sorted(standings) == sorted(player.name for player in players)
    assert list(game.turn_order) == []
    assert all(
        game.players[player_id].hand.is_empty() for player_id in game.standings[:-1]
    )

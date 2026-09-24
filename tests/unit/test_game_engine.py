import pytest

from president_ai.president import Move, President, Rank, TurnRecord
from tests.helpers import ScriptedPlayer, make_cards


def test_minimum_player_count():
    with pytest.raises(ValueError, match="Not Enough Players"):
        President([ScriptedPlayer("Player 0")])


def test_minimum_initial_hands():
    players = [ScriptedPlayer("Player 0"), ScriptedPlayer("Player 1")]

    with pytest.raises(ValueError, match="Each Player Needs an Initial Hand"):
        President(players, initial_hands=[[]])


def test_minimum_card_count():
    players = [ScriptedPlayer("Player 0"), ScriptedPlayer("Player 1")]

    with pytest.raises(ValueError, match="Not Enough Cards"):
        President(players, cards_per_player=27)


def test_deals_cards_correctly():
    players = [ScriptedPlayer(f"Player {player_id}") for player_id in range(3)]
    game = President(players, cards_per_player=10)

    assert [len(player.hand) for player in game.players.values()] == [10, 10, 10]


def test_next_turn():
    three = make_cards(Rank.THREE, 1)[0]
    five = make_cards(Rank.FIVE, 1)[0]
    move = Move((three,))
    players = [
        ScriptedPlayer("Player 0", move),
        ScriptedPlayer("Player 1"),
    ]
    game = President(players, initial_hands=[[three, five], make_cards(Rank.FOUR, 1)])

    turn = game.next_turn()

    assert turn == TurnRecord(0, move)
    assert game.players[0].hand.cards == [five]
    assert list(game.turn_order) == [1, 0]
    assert game.current_move == move
    assert players[0].views[0].turn_order == (0, 1)
    assert players[0].views[0].cards_remaining == (2, 1)


def test_get_player_view_and_play_move():
    three = make_cards(Rank.THREE, 1)[0]
    five = make_cards(Rank.FIVE, 1)[0]
    move = Move((three,))
    players = [ScriptedPlayer("Player 0"), ScriptedPlayer("Player 1")]
    game = President(players, initial_hands=[[three, five], make_cards(Rank.FOUR, 1)])

    view = game.get_player_view()
    turn = game.play_move(move)

    assert view.player_id == 0
    assert view.hand == (three, five)
    assert move in view.possible_moves
    assert turn == TurnRecord(0, move)
    assert game.players[0].hand.cards == [five]
    assert list(game.turn_order) == [1, 0]


def test_play_move_removes_the_exact_selected_suit():
    three_diamonds, three_clubs = make_cards(Rank.THREE, 2)
    five = make_cards(Rank.FIVE, 1)[0]
    players = [ScriptedPlayer("Player 0"), ScriptedPlayer("Player 1")]
    game = President(
        players,
        initial_hands=[[three_diamonds, three_clubs, five], make_cards(Rank.FOUR, 1)],
    )

    game.play_move(Move((three_clubs,)))

    assert game.players[0].hand.cards == [three_diamonds, five]


def test_snapshot_restores_and_continues_game():
    three, six, ten = (
        make_cards(rank, 1)[0] for rank in (Rank.THREE, Rank.SIX, Rank.TEN)
    )
    four, seven, jack = (
        make_cards(rank, 1)[0] for rank in (Rank.FOUR, Rank.SEVEN, Rank.JACK)
    )
    five, eight, queen = (
        make_cards(rank, 1)[0] for rank in (Rank.FIVE, Rank.EIGHT, Rank.QUEEN)
    )
    players = [ScriptedPlayer(f"Player {player_id}") for player_id in range(3)]
    game = President(
        players,
        initial_hands=[
            [three, six, ten],
            [four, seven, jack],
            [five, eight, queen],
        ],
    )

    game.play_move(Move((three,)))
    game.play_move(None)
    game.play_move(None)
    game.play_move(Move((six,)))
    game.play_move(None)
    state = game.snapshot()

    restored_players = [ScriptedPlayer(f"Player {player_id}") for player_id in range(3)]
    restored = President.from_state(restored_players, state)

    assert restored.snapshot() == state

    move = Move((eight,))
    assert restored.play_move(move) == game.play_move(move)
    assert restored.snapshot() == game.snapshot()


def test_player_view_turn_order_and_card_counts():
    three = make_cards(Rank.THREE, 1)[0]
    four = make_cards(Rank.FOUR, 1)[0]
    seven = make_cards(Rank.SEVEN, 1)[0]
    players = [
        ScriptedPlayer("Player 0", Move((three,))),
        ScriptedPlayer("Player 1", Move((four,))),
        ScriptedPlayer("Player 2"),
    ]
    game = President(
        players,
        initial_hands=[[three], [four, seven], make_cards(Rank.FIVE, 1)],
    )

    game.next_turn()
    game.next_turn()

    view = players[1].views[0]
    assert view.turn_order == (1, 2)
    assert view.cards_remaining == (0, 2, 1)
    assert tuple(view.cards_remaining[player_id] for player_id in view.turn_order) == (
        2,
        1,
    )


def test_cannot_pass_first_move():
    players = [
        ScriptedPlayer("Player 0", None),
        ScriptedPlayer("Player 1"),
    ]
    game = President(
        players,
        initial_hands=[make_cards(Rank.THREE, 1), make_cards(Rank.FOUR, 1)],
    )

    with pytest.raises(ValueError, match="Cannot Pass When Starting a Trick"):
        game.next_turn()


def test_rejects_invalid_move():
    invalid_move = Move(tuple(make_cards(Rank.FOUR, 1)))
    players = [
        ScriptedPlayer("Player 0", invalid_move),
        ScriptedPlayer("Player 1"),
    ]
    game = President(
        players,
        initial_hands=[make_cards(Rank.THREE, 1), make_cards(Rank.FOUR, 1)],
    )

    with pytest.raises(ValueError, match="Invalid Move Played"):
        game.next_turn()


def test_winner():
    winning_move = Move(tuple(make_cards(Rank.THREE, 1)))
    players = [
        ScriptedPlayer("Player 0", winning_move),
        ScriptedPlayer("Player 1"),
    ]
    game = President(
        players,
        initial_hands=[list(winning_move.cards), make_cards(Rank.FOUR, 1)],
    )

    game.next_turn()

    assert game.standings == [0]
    assert list(game.turn_order) == [1]
    assert game.players[0].hand.is_empty()


def test_restart_after_everyone_passes():
    opening_move = Move(tuple(make_cards(Rank.THREE, 1)))
    players = [
        ScriptedPlayer("Player 0", opening_move),
        ScriptedPlayer("Player 1", None),
        ScriptedPlayer("Player 2", None),
    ]
    game = President(
        players,
        initial_hands=[
            list(opening_move.cards) + make_cards(Rank.TEN, 1),
            make_cards(Rank.FOUR, 1),
            make_cards(Rank.FIVE, 1),
        ],
    )

    game.next_turn()
    game.next_turn()
    assert game.current_move == opening_move
    game.next_turn()

    assert game.current_move is None
    assert game.current_trick == []
    assert game.passed_players == set()
    assert list(game.turn_order) == [0, 1, 2]
    assert game.trick_history == [
        (
            TurnRecord(0, opening_move),
            TurnRecord(1, None),
            TurnRecord(2, None),
        )
    ]


def test_game_standings():
    opening_move = Move(tuple(make_cards(Rank.THREE, 1)))
    winning_move = Move(tuple(make_cards(Rank.FOUR, 1)))
    players = [
        ScriptedPlayer("First Player", opening_move),
        ScriptedPlayer("Second Player", winning_move),
    ]
    game = President(
        players,
        initial_hands=[
            list(opening_move.cards) + make_cards(Rank.FIVE, 1),
            list(winning_move.cards),
        ],
    )

    assert game.run() == ["Second Player", "First Player"]

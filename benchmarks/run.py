import random
import time

from src.president import President, get_full_deck
from src.strategies.minimal_card_bot import MinimalCardBot
from src.strategies.model_bot import ModelBot
from src.strategies.random_bot import RandomBot

SEED = 67
random.seed(SEED)


"""
First question: Just how strong is advantage in playing first?

Methodology: At each hand size of 5, 7, 10, and 12, generate hand_1 and hand_2
and have the minimax 2-player solver solve it. Calculate the win rate of the player 
playing first with hand1 and then hand2.

Restrictions: Can only solve 2-player optimally with perfect information
and a smaller hand size.

For hand size 5: Original 63.3% WR (3.36s). Swapped hands 59.5% WR (3.43s).
For hand size 7: Original 60.1% WR (31.23s). Swapped hands 61.1% WR (30.55s).
For hand size 10: Original 59.0% WR (92.06s). Swapped hands 66.0% WR (87.83s).
For hand size 12: Original 60.0% WR (246.35s). Swapped hands 60.0% WR (295.1s).
"""
"""
MINIMAX_GAME_COUNTS = [1000, 1000, 100, 10]
MINIMAX_HAND_SIZES = [5, 7, 10, 12]


for hand_size, game_count in zip(MINIMAX_HAND_SIZES, MINIMAX_GAME_COUNTS):
    win_1, win_2 = 0, 0
    time_1, time_2 = 0, 0

    for trial_id in range(game_count):
        shuffled_deck = get_full_deck(shuffled=True)
        hand_1, hand_2 = (
            shuffled_deck[0:hand_size],
            shuffled_deck[hand_size : hand_size * 2],
        )

        start_time = time.perf_counter()
        result = minimax_solver({}, Hand(hand_1), Hand(hand_2), None)
        checkpoint = time.perf_counter()
        result_2 = minimax_solver({}, Hand(hand_2), Hand(hand_1), None)
        checkpoint_2 = time.perf_counter()

        if result:
            win_1 += 1
        if result_2:
            win_2 += 1

        time_1 += checkpoint - start_time
        time_2 += checkpoint_2 - checkpoint

    wr_1 = round(win_1 / game_count * 100, 2)
    wr_2 = round(win_2 / game_count * 100, 2)
    time_1 = round(time_1, 2)
    time_2 = round(time_2, 2)

    print(
        f"For hand size {hand_size}: Original {wr_1}% WR ({time_1}s). Swapped hands {wr_2}% WR ({time_2}s)."
    )
"""

"""
Second question: Given all the strategies, which one is the best?

Methodology: A round robin tournament where each bot strategy plays against every other
bot strategy for a number of games. To remain balanced, each generated pair of hands 
will be played normally, swapped, then with the other player going first on both.
The tournament will be played with 10 cards, 15 cards and then 20 cards each.

Restrictions: Can only calculate 2-player win-rates currently.

Over 1000 games with hand sizes [10, 15, 20] without question 1 running first on the seed:

RandomBot v.s. MinimalCardBot with 10 cards: 22.2% WR to 77.8% WR in 0.66s
RandomBot v.s. MinimalCardBot with 15 cards: 11.8% WR to 88.2% WR in 1.25s
RandomBot v.s. MinimalCardBot with 20 cards: 9.6% WR to 90.4% WR in 1.92s
RandomBot v.s. ModelBot with 10 cards: 20.7% WR to 79.3% WR in 3.42s
RandomBot v.s. ModelBot with 15 cards: 11.2% WR to 88.8% WR in 5.17s
RandomBot v.s. ModelBot with 20 cards: 7.4% WR to 92.6% WR in 6.54s
MinimalCardBot v.s. ModelBot with 10 cards: 50.2% WR to 49.8% WR in 3.18s
MinimalCardBot v.s. ModelBot with 15 cards: 50.1% WR to 49.9% WR in 4.67s
MinimalCardBot v.s. ModelBot with 20 cards: 50.0% WR to 50.0% WR in 6.09s
"""
TOURNAMENT_GAME_COUNT = 1000
TOURNAMENT_PLAYERS = [RandomBot, MinimalCardBot, ModelBot]
TOURNAMENT_HAND_SIZES = [10, 15, 20]


for i in range(len(TOURNAMENT_PLAYERS)):
    for j in range(i + 1, len(TOURNAMENT_PLAYERS)):
        for hand_size in TOURNAMENT_HAND_SIZES:
            player_1_wins = 0
            start_time = time.perf_counter()

            for game_id in range(TOURNAMENT_GAME_COUNT // 4):
                shuffled_deck = get_full_deck(shuffled=True)
                hand_a, hand_b = (
                    shuffled_deck[0:hand_size],
                    shuffled_deck[hand_size : hand_size * 2],
                )

                for player_1_id, player_2_id in [(i, j), (j, i)]:
                    for hand_1, hand_2 in [(hand_a, hand_b), (hand_b, hand_a)]:
                        players = [
                            TOURNAMENT_PLAYERS[player_1_id](f"{player_1_id}"),
                            TOURNAMENT_PLAYERS[player_2_id](f"{player_2_id}"),
                        ]
                        game = President(players, initial_hands=[hand_1, hand_2])
                        standings = game.run()

                        if standings[0] == f"{i}":
                            player_1_wins += 1

            time_elapsed = round(time.perf_counter() - start_time, 2)
            bot_1 = TOURNAMENT_PLAYERS[i].__name__
            bot_2 = TOURNAMENT_PLAYERS[j].__name__
            bot_1_wr = round(player_1_wins / TOURNAMENT_GAME_COUNT * 100, 2)
            bot_2_wr = 100 - bot_1_wr

            print(
                f"{bot_1} v.s. {bot_2} with {hand_size} cards: {bot_1_wr}% WR to {bot_2_wr}% WR in {time_elapsed}s"
            )

"""
President Bot benchmarking script

python -m benchmarks.run [command]

-WR for winrate benchmark
-T for pairwise tournament
"""

import random
import time
from sys import argv

from src.president import Hand, President, get_full_deck
from src.strategies.fast_minimax import solve_game
from src.strategies.registry import Strategy, create_player

SEED = 73
random.seed(SEED)


"""
First question: Just how strong is advantage in playing first?

Methodology: At each hand size of 5, 7, 10, and 12, generate hand_1 and hand_2
and have the minimax 2-player solver solve it. Calculate the win rate of the player 
playing first with hand1 and then hand2.

Restrictions: Can only solve 2-player optimally with perfect information
and a smaller hand size.

For hand size 5: Original 61.53% WR (1.78s). Swapped hands 61.92% WR (1.73s).
For hand size 7: Original 60.82% WR (11.4s). Swapped hands 60.68% WR (10.87s).
For hand size 10: Original 60.6% WR (21.2s). Swapped hands 62.7% WR (20.3s).
For hand size 12: Original 58.1% WR (225.45s). Swapped hands 64.1% WR (214.3s).
"""
MINIMAX_GAME_COUNTS = [10000, 10000, 1000, 1000]
MINIMAX_HAND_SIZES = [5, 7, 10, 12]


def run_winrate_benchmark():
    for hand_size, game_count in zip(MINIMAX_HAND_SIZES, MINIMAX_GAME_COUNTS):
        win_1, win_2 = 0, 0
        time_1, time_2 = 0, 0

        for _ in range(game_count):
            shuffled_deck = get_full_deck(shuffled=True)
            hand_1, hand_2 = (
                shuffled_deck[0:hand_size],
                shuffled_deck[hand_size : hand_size * 2],
            )

            start_time = time.perf_counter()
            result = solve_game(Hand(hand_1), Hand(hand_2))
            checkpoint = time.perf_counter()
            result_2 = solve_game(Hand(hand_2), Hand(hand_1))
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
Second question: Given all the strategies, which one is the best?

Methodology: A round robin tournament where each bot strategy plays against every other
bot strategy for a number of games. To remain balanced, each generated pair of hands 
will be played normally, swapped, then with the other player going first on both.

Restrictions: Can only calculate 2-player win-rates currently.

RandomBot v.s. MinimalCardBot with 5 cards: 38.7% WR to 61.3% WR in 0.29s
RandomBot v.s. MinimalCardBot with 10 cards: 22.6% WR to 77.4% WR in 0.87s
RandomBot v.s. MinimalCardBot with 15 cards: 10.85% WR to 89.15% WR in 1.66s
RandomBot v.s. MinimalCardBot with 20 cards: 7.6% WR to 92.4% WR in 2.8s
RandomBot v.s. MinimalCardBot with 25 cards: 6.35% WR to 93.65% WR in 3.82s
RandomBot v.s. Copycat-v1.0 with 5 cards: 37.25% WR to 62.75% WR in 2.0s
RandomBot v.s. Copycat-v1.0 with 10 cards: 19.95% WR to 80.05% WR in 3.97s
RandomBot v.s. Copycat-v1.0 with 15 cards: 12.6% WR to 87.4% WR in 5.81s
RandomBot v.s. Copycat-v1.0 with 20 cards: 7.9% WR to 92.1% WR in 7.39s
RandomBot v.s. Copycat-v1.0 with 25 cards: 6.45% WR to 93.55% WR in 10.19s
RandomBot v.s. Jester-v1.0 with 5 cards: 37.4% WR to 62.6% WR in 2.18s
RandomBot v.s. Jester-v1.0 with 10 cards: 21.45% WR to 78.55% WR in 4.53s
RandomBot v.s. Jester-v1.0 with 15 cards: 12.0% WR to 88.0% WR in 6.22s
RandomBot v.s. Jester-v1.0 with 20 cards: 7.65% WR to 92.35% WR in 8.56s
RandomBot v.s. Jester-v1.0 with 25 cards: 5.7% WR to 94.3% WR in 10.54s
RandomBot v.s. Jester-v1.1 with 5 cards: 37.4% WR to 62.6% WR in 2.33s
RandomBot v.s. Jester-v1.1 with 10 cards: 20.65% WR to 79.35% WR in 5.14s
RandomBot v.s. Jester-v1.1 with 15 cards: 11.1% WR to 88.9% WR in 7.46s
RandomBot v.s. Jester-v1.1 with 20 cards: 8.45% WR to 91.55% WR in 9.47s
RandomBot v.s. Jester-v1.1 with 25 cards: 6.45% WR to 93.55% WR in 11.52s
MinimalCardBot v.s. Copycat-v1.0 with 5 cards: 50.05% WR to 49.95% WR in 2.42s
MinimalCardBot v.s. Copycat-v1.0 with 10 cards: 50.0% WR to 50.0% WR in 5.03s
MinimalCardBot v.s. Copycat-v1.0 with 15 cards: 49.95% WR to 50.05% WR in 7.41s
MinimalCardBot v.s. Copycat-v1.0 with 20 cards: 49.8% WR to 50.2% WR in 9.43s
MinimalCardBot v.s. Copycat-v1.0 with 25 cards: 49.7% WR to 50.3% WR in 12.24s
MinimalCardBot v.s. Jester-v1.0 with 5 cards: 48.35% WR to 51.65% WR in 2.27s
MinimalCardBot v.s. Jester-v1.0 with 10 cards: 46.45% WR to 53.55% WR in 4.55s
MinimalCardBot v.s. Jester-v1.0 with 15 cards: 45.35% WR to 54.65% WR in 7.02s
MinimalCardBot v.s. Jester-v1.0 with 20 cards: 45.0% WR to 55.0% WR in 8.75s
MinimalCardBot v.s. Jester-v1.0 with 25 cards: 42.0% WR to 58.0% WR in 11.0s
MinimalCardBot v.s. Jester-v1.1 with 5 cards: 48.1% WR to 51.9% WR in 2.31s
MinimalCardBot v.s. Jester-v1.1 with 10 cards: 46.55% WR to 53.45% WR in 4.9s
MinimalCardBot v.s. Jester-v1.1 with 15 cards: 44.75% WR to 55.25% WR in 6.62s
MinimalCardBot v.s. Jester-v1.1 with 20 cards: 43.05% WR to 56.95% WR in 8.83s
MinimalCardBot v.s. Jester-v1.1 with 25 cards: 37.6% WR to 62.4% WR in 11.12s
Copycat-v1.0 v.s. Jester-v1.0 with 5 cards: 47.7% WR to 52.3% WR in 4.23s
Copycat-v1.0 v.s. Jester-v1.0 with 10 cards: 46.55% WR to 53.45% WR in 7.89s
Copycat-v1.0 v.s. Jester-v1.0 with 15 cards: 44.9% WR to 55.1% WR in 11.57s
Copycat-v1.0 v.s. Jester-v1.0 with 20 cards: 44.1% WR to 55.9% WR in 14.13s
Copycat-v1.0 v.s. Jester-v1.0 with 25 cards: 43.25% WR to 56.75% WR in 17.07s
Copycat-v1.0 v.s. Jester-v1.1 with 5 cards: 48.45% WR to 51.55% WR in 3.99s
Copycat-v1.0 v.s. Jester-v1.1 with 10 cards: 46.65% WR to 53.35% WR in 8.07s
Copycat-v1.0 v.s. Jester-v1.1 with 15 cards: 43.05% WR to 56.95% WR in 11.13s
Copycat-v1.0 v.s. Jester-v1.1 with 20 cards: 44.4% WR to 55.6% WR in 13.9s
Copycat-v1.0 v.s. Jester-v1.1 with 25 cards: 38.25% WR to 61.75% WR in 16.9s
Jester-v1.0 v.s. Jester-v1.1 with 5 cards: 50.3% WR to 49.7% WR in 4.03s
Jester-v1.0 v.s. Jester-v1.1 with 10 cards: 48.5% WR to 51.5% WR in 7.76s
Jester-v1.0 v.s. Jester-v1.1 with 15 cards: 49.3% WR to 50.7% WR in 10.56s
Jester-v1.0 v.s. Jester-v1.1 with 20 cards: 47.25% WR to 52.75% WR in 13.34s
Jester-v1.0 v.s. Jester-v1.1 with 25 cards: 47.8% WR to 52.2% WR in 16.02s
"""
TOURNAMENT_GAME_COUNT = 2000
TOURNAMENT_HAND_SIZES = [5, 10, 15, 20, 25]
TOURNAMENT_PLAYERS: list[Strategy] = [
    Strategy.RANDOM,
    Strategy.MINIMAL_CARD,
    Strategy.COPYCAT_V1_0,
    Strategy.JESTER_V1_0,
    Strategy.JESTER_V1_1,
]


def run_tournament_benchmark():
    for i in range(len(TOURNAMENT_PLAYERS)):
        for j in range(i + 1, len(TOURNAMENT_PLAYERS)):
            for hand_size in TOURNAMENT_HAND_SIZES:
                player_1_wins = 0
                start_time = time.perf_counter()

                for _ in range(TOURNAMENT_GAME_COUNT // 4):
                    shuffled_deck = get_full_deck(shuffled=True)
                    hand_a, hand_b = (
                        shuffled_deck[0:hand_size],
                        shuffled_deck[hand_size : hand_size * 2],
                    )

                    for player_1_id, player_2_id in [(i, j), (j, i)]:
                        for hand_1, hand_2 in [(hand_a, hand_b), (hand_b, hand_a)]:
                            players = [
                                create_player(
                                    TOURNAMENT_PLAYERS[player_1_id], f"{player_1_id}"
                                ),
                                create_player(
                                    TOURNAMENT_PLAYERS[player_2_id], f"{player_2_id}"
                                ),
                            ]
                            game = President(players, initial_hands=[hand_1, hand_2])
                            standings = game.run()

                            if standings[0] == f"{i}":
                                player_1_wins += 1

                time_elapsed = round(time.perf_counter() - start_time, 2)
                bot_1 = str(TOURNAMENT_PLAYERS[i])
                bot_2 = str(TOURNAMENT_PLAYERS[j])
                bot_1_wr = round(player_1_wins / TOURNAMENT_GAME_COUNT * 100, 2)
                bot_2_wr = 100 - bot_1_wr

                print(
                    f"{bot_1} v.s. {bot_2} with {hand_size} cards: {bot_1_wr}% WR to {bot_2_wr}% WR in {time_elapsed}s"
                )


if __name__ == "__main__":
    benchmark = argv[1].lower()
    match benchmark:
        case "-wr":
            run_winrate_benchmark()
        case "-t":
            run_tournament_benchmark()

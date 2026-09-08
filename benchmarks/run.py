import random
import time

from src.president import Hand, get_full_deck
from src.strategies.minimax import minimax_solver

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

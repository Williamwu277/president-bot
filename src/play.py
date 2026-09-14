import sys
from random import shuffle

from .president import FULL_DECK_SIZE, President
from .strategies.human_play import HumanPlayer
from .strategies.registry import Strategy, create_player

MAX_BOTS = 3

strategy_registry = [
    Strategy.RANDOM,
    Strategy.MINIMAL_CARD,
    Strategy.COPYCAT_V1_0,
    Strategy.JESTER_V1_0,
    Strategy.JESTER_V1_1,
]


def durable_input(prompt: str, start: int, end: int) -> int:
    while True:
        result = int(input(prompt))
        if start <= result <= end:
            return result
        else:
            print("Invalid input")


def play_game(debug: bool = False):
    bot_count = durable_input("Number of bots [1, 3]: ", 1, 3)

    print(f"We currently have a registry of {len(strategy_registry)} bot types:")
    for bot_id, bot_type in enumerate(strategy_registry, start=1):
        print(f"{bot_id}: {bot_type!s}")
    print("Please pick the id of the bots you want to play against one per line:")

    players = [HumanPlayer("Human Player", advisor=debug)]

    for i in range(1, bot_count + 1):
        bot_id = durable_input(f"Bot {i}: ", 1, len(strategy_registry))
        bot_type = strategy_registry[bot_id - 1]
        players.append(create_player(bot_type, f"{bot_type!s}-{i}"))

    shuffle(players)

    max_card_count = FULL_DECK_SIZE // len(players)
    card_count = durable_input(
        f"Number of cards per player [1, {max_card_count}]: ", 1, max_card_count
    )

    game = President(players=players, cards_per_player=card_count)

    standings = game.run()
    print("Final standings:")
    for rank, name in enumerate(standings, start=1):
        print(f"{rank}: {name}")


if __name__ == "__main__":
    play_game(debug="-D" in sys.argv[1:])

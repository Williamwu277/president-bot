from random import shuffle

from .president import FULL_DECK_SIZE, President
from .strategies.registry import Strategy, create_player

MAX_BOTS = 3

strategy_registry = [
    Strategy.RANDOM,
    Strategy.MINIMAL_CARD,
    Strategy.COPYCAT_V1_0,
    Strategy.JESTER_V1_0,
    Strategy.JESTER_V1_1,
]


def play_game():
    bot_count = int(input("Number of bots [1, 3]: "))

    if bot_count > MAX_BOTS or bot_count <= 0:
        raise ValueError("Invalid scenario")

    print(f"We currently have a registry of {len(strategy_registry)} bot types:")
    for bot_id, bot_type in enumerate(strategy_registry, start=1):
        print(f"{bot_id}: {bot_type!s}")
    print("Please pick the id of the bots you want to play against one per line:")

    players = [create_player(Strategy.HUMAN, "Human Player")]

    for i in range(1, bot_count + 1):
        bot_id = int(input(f"Bot {i}: "))

        if bot_id < 1 or bot_id > len(strategy_registry):
            raise ValueError("Invalid scenario")

        bot_type = strategy_registry[bot_id - 1]
        players.append(create_player(bot_type, f"{bot_type!s}-{i}"))

    shuffle(players)

    max_card_count = FULL_DECK_SIZE // len(players)
    card_count = int(input(f"Number of cards per player [1, {max_card_count}]: "))

    if not 1 <= card_count <= max_card_count:
        raise ValueError(f"Cards per player must be between 1 and {max_card_count}")

    game = President(players=players, cards_per_player=card_count)

    standings = game.run()
    print("Final standings:")
    for rank, name in enumerate(standings, start=1):
        print(f"{rank}: {name}")


if __name__ == "__main__":
    play_game()

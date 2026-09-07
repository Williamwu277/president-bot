from .president import President
from .strategies.human_play import HumanPlayer
from .strategies.random_bot import RandomBot
from .strategies.minimal_card_bot import MinimalCardBot
from .strategies.minimax import MinimaxBot
from random import shuffle


MAX_BOTS = 5

strategy_registry = [
    (RandomBot, "Random Bot"),
    (MinimalCardBot, "Minimal Card Bot"),
    (MinimaxBot, "Minimax Bot")
]


def play_game():
    bot_count = int(input("Number of bots [1, 5]: "))

    if bot_count > MAX_BOTS or bot_count <= 0:
        raise ValueError("Invalid scenario")

    print(f"We currently have a registry of {len(strategy_registry)} bot types:")
    for bot_id, (_, bot_name) in enumerate(strategy_registry, start = 1):
        print(f"{bot_id}: {bot_name}")
    print("Please pick the id of the bots you want to play against one per line:")

    players = [HumanPlayer("Human Player")]

    for i in range(1, bot_count + 1):
        bot_id = int(input(f"Bot {i}: "))

        if bot_id < 1 or bot_id > len(strategy_registry):
            raise ValueError("Invalid scenario")

        bot, name = strategy_registry[bot_id - 1]
        players.append(bot(f"{name}-{i}"))

    shuffle_order = input("Shuffle player order? [y/n]: ")
    if shuffle_order.lower() in ["y", "yes"]:
        shuffle(players)

    deck_count = int(input("Number of decks: "))
    card_count = int(input("Number of cards per player: "))

    game = President(
        players = players,
        deck_count = deck_count,
        cards_per_player = card_count
    )

    standings = game.run()
    print("Final standings:")
    for rank, name in enumerate(standings, start=1):
        print(f"{rank}: {name}")


if __name__ == "__main__":
    play_game()

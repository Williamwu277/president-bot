from president import President
from strategies import HumanPlayer, RandomBot, MinimalCardBot
from random import shuffle

players = [
    HumanPlayer("Human Player"),
    RandomBot("Random Bot"),
    MinimalCardBot("Minimal Card Bot")
]

shuffle(players)

game = President(
    players = players,
    cards_per_player = 13
)

standings = game.run()
print("Final standings:")
for rank, name in enumerate(standings, start=1):
    print(f"{rank}: {name}")

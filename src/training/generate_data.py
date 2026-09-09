from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import Any

import torch

from ..president import Move, Player, PlayerView, President
from ..strategies.minimal_card_bot import MinimalCardBot
from ..strategies.random_bot import RandomBot
from .encoder import MOVE_COUNT, SCHEMA_VERSION, STATE_SIZE, encode_move, encode_view


@dataclass(frozen=True)
class TrainingExample:
    features: torch.Tensor
    legal_moves: torch.Tensor
    target_move: int


class RecordingPlayer(Player):
    def __init__(self, name: str, player: Player, data: list[TrainingExample]):
        super().__init__(name)
        self.player = player
        self.data = data

    def make_move(self, view: PlayerView) -> Move | None:
        move = self.player.make_move(view)
        features, legal_moves = encode_view(view)

        self.data.append(TrainingExample(features, legal_moves, encode_move(move)))

        return move


PlayerArgs = tuple[type[Player], tuple[Any, ...]]


@dataclass(frozen=True)
class TrainingSetParameter:
    games: int
    cards_per_player: int
    teacher: PlayerArgs
    opponents: list[PlayerArgs]


def generate_training_set(
    games: int, cards_per_player: int, teacher: PlayerArgs, opponents: list[PlayerArgs]
) -> list[TrainingExample]:
    data: list[TrainingExample] = []

    for _ in range(games):
        teacher_type, teacher_args = teacher

        recording_teacher = RecordingPlayer(
            name="Teacher",
            player=teacher_type(*teacher_args),
            data=data,
        )

        opponent_players = [
            opponent_type(*opponent_args) for opponent_type, opponent_args in opponents
        ]

        players = [recording_teacher, *opponent_players]
        shuffle(players)

        game = President(
            players,
            cards_per_player=cards_per_player,
        )
        game.run()

    return data


def generate_data(training_sets: list[TrainingSetParameter]):
    data: list[TrainingExample] = []
    for training_set in training_sets:
        generated_data = generate_training_set(
            training_set.games,
            training_set.cards_per_player,
            training_set.teacher,
            training_set.opponents,
        )
        data.extend(generated_data)
    return data


def save_data(data: list[TrainingExample], path: Path) -> None:
    if not data:
        raise ValueError("Cannot save an empty dataset")

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "state_size": STATE_SIZE,
        "move_count": MOVE_COUNT,
        "features": torch.stack([example.features for example in data]),
        "legal_moves": torch.stack([example.legal_moves for example in data]),
        "target_moves": torch.tensor(
            [example.target_move for example in data],
            dtype=torch.long,
        ),
    }

    torch.save(payload, path)


def load_examples(path: Path):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Dataset encoder schema does not match current encoder")

    if payload["state_size"] != STATE_SIZE:
        raise ValueError("Dataset has an unexpected state size")

    if payload["move_count"] != MOVE_COUNT:
        raise ValueError("Dataset has an unexpected move count")

    return (
        payload["features"],
        payload["legal_moves"],
        payload["target_moves"],
    )


test_training_parameters: list[TrainingSetParameter] = [
    TrainingSetParameter(
        100, 26, (MinimalCardBot, ("Teacher",)), [(RandomBot, ("Opponent",))]
    ),
    TrainingSetParameter(
        100, 13, (MinimalCardBot, ("Teacher",)), [(RandomBot, ("Opponent",))]
    ),
    TrainingSetParameter(
        100, 26, (MinimalCardBot, ("Teacher",)), [(MinimalCardBot, ("Opponent",))]
    ),
    TrainingSetParameter(
        100, 13, (MinimalCardBot, ("Teacher",)), [(MinimalCardBot, ("Opponent",))]
    ),
]

if __name__ == "__main__":
    data = generate_data(test_training_parameters)
    print(f"Generated {len(data):,} training examples")
    save_data(data, Path("data/test_data.pt"))

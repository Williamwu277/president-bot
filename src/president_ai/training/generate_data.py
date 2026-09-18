from dataclasses import dataclass
from pathlib import Path
from random import shuffle

import torch

from ..president import Move, Player, PlayerView, President
from ..strategies.registry import Strategy, create_player
from .encoder import MOVE_COUNT, SCHEMA_VERSION, STATE_SIZE, encode_move, encode_view


@dataclass
class TrainingExample:
    features: torch.Tensor
    legal_moves: torch.Tensor
    target_move: int
    outcome: int | None = None


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


@dataclass(frozen=True)
class TrainingSetParameter:
    games: int
    cards_per_player: int
    teacher: Strategy
    opponents: list[Strategy]


def generate_training_set(
    games: int, cards_per_player: int, teacher: Strategy, opponents: list[Strategy]
) -> list[TrainingExample]:
    data: list[TrainingExample] = []

    for _ in range(games):
        game_data: list[TrainingExample] = []

        recording_teacher = RecordingPlayer(
            name="Teacher",
            player=create_player(teacher, ""),
            data=game_data,
        )

        opponent_players = [
            create_player(opponent_type, "") for opponent_type in opponents
        ]

        players = [recording_teacher, *opponent_players]
        shuffle(players)

        game = President(
            players,
            cards_per_player=cards_per_player,
        )
        winner_name = game.run()[0]

        result = -1
        if winner_name == recording_teacher.name:
            result = 1

        for training_example in game_data:
            training_example.outcome = result

        data.extend(game_data)

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
        "outcomes": torch.tensor(
            [example.outcome for example in data], dtype=torch.float32
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
        payload["outcomes"],
    )


training_parameters: list[TrainingSetParameter] = [
    TrainingSetParameter(1000, 26, Strategy.MINIMAL_CARD, [Strategy.RANDOM]),
    TrainingSetParameter(1000, 13, Strategy.MINIMAL_CARD, [Strategy.RANDOM]),
    TrainingSetParameter(1000, 26, Strategy.MINIMAL_CARD, [Strategy.MINIMAL_CARD]),
    TrainingSetParameter(1000, 13, Strategy.MINIMAL_CARD, [Strategy.MINIMAL_CARD]),
]


validation_parameters: list[TrainingSetParameter] = [
    TrainingSetParameter(200, 26, Strategy.MINIMAL_CARD, [Strategy.RANDOM]),
    TrainingSetParameter(200, 13, Strategy.MINIMAL_CARD, [Strategy.RANDOM]),
    TrainingSetParameter(200, 26, Strategy.MINIMAL_CARD, [Strategy.MINIMAL_CARD]),
    TrainingSetParameter(200, 13, Strategy.MINIMAL_CARD, [Strategy.MINIMAL_CARD]),
]


if __name__ == "__main__":
    training_data = generate_data(training_parameters)
    validation_data = generate_data(validation_parameters)
    print(f"Generated {len(training_data):,} training examples")
    print(f"Generated {len(validation_data):,} validation examples")
    save_data(training_data, Path("data/training_data.pt"))
    save_data(validation_data, Path("data/validation_data.pt"))

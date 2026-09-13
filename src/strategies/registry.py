from enum import Enum
from pathlib import Path

from ..president import Player
from .human_play import HumanPlayer
from .minimal_card_bot import MinimalCardBot
from .minimax import MinimaxBot
from .model_bot import ModelBot
from .random_bot import RandomBot


class Strategy(Enum):
    """
    A registry of all strategies
    """

    HUMAN = "HumanUser"
    RANDOM = "RandomBot"
    MINIMAL_CARD = "MinimalCardBot"
    MINIMAX = "MinimaxBot"
    COPYCAT_V1_0 = "Copycat-v1.0"
    JESTER_V1_0 = "Jester-v1.0"
    JESTER_V1_1 = "Jester-v1.1"

    def __str__(self) -> str:
        return self.value


MODEL_MAP = {
    Strategy.COPYCAT_V1_0: Path("final_models/supervised_model.pt"),
    Strategy.JESTER_V1_0: Path("final_models/reinforcement_model_10k.pt"),
    Strategy.JESTER_V1_1: Path(
        "final_models/reinforcement_model_supervised_base_30k.pt"
    ),
}


def create_player(strategy: Strategy, name: str) -> Player:
    match strategy:
        case Strategy.HUMAN:
            return HumanPlayer(name)
        case Strategy.RANDOM:
            return RandomBot(name)
        case Strategy.MINIMAL_CARD:
            return MinimalCardBot(name)
        case Strategy.MINIMAX:
            return MinimaxBot(name)
        case Strategy.COPYCAT_V1_0:
            return ModelBot(name, MODEL_MAP[strategy])
        case Strategy.JESTER_V1_0:
            return ModelBot(name, MODEL_MAP[strategy])
        case Strategy.JESTER_V1_1:
            return ModelBot(name, MODEL_MAP[strategy])
        case _:
            raise ValueError("No strategy matches")

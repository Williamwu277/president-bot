from dataclasses import dataclass
from pathlib import Path
from random import choice, random, shuffle

import torch
import torch.nn.functional as F

from ..president import Move, Player, PlayerView, President
from ..strategies.minimal_card_bot import MinimalCardBot
from ..strategies.model_bot import ModelBot
from ..strategies.random_bot import RandomBot
from .encoder import decode_move, encode_view
from .generate_data import PlayerArgs
from .model import SUPERVISED_MODEL_PATH, PolicyValueModel

BATCH_COUNT = 500
GAMES_PER_BATCH = 64
LEARNING_RATE = 1e-4
SELF_PLAY_FRACTION = 0.7
POLICY_TEMPERATURE = 1.25
REINFORCEMENT_MODEL_PATH = Path("models/reinforcement_model.pt")


@dataclass
class SelfPlayRecord:
    features: torch.Tensor
    legal_moves: torch.Tensor
    action: int
    reward: float | None = None


@dataclass(frozen=True)
class ReinforcementTrainingMetrics:
    combined_loss: float
    policy_loss: float
    value_loss: float
    entropy: float


def make_policy_distribution(
    policy_logits: torch.Tensor,
    legal_moves: torch.Tensor,
) -> torch.distributions.Categorical:
    scaled_logits = policy_logits / POLICY_TEMPERATURE
    masked_logits = scaled_logits.masked_fill(
        ~legal_moves,
        torch.finfo(scaled_logits.dtype).min,
    )
    return torch.distributions.Categorical(logits=masked_logits)


class SelfPlayRecorder(Player):
    def __init__(self, name: str, data: list[SelfPlayRecord], model: PolicyValueModel):
        super().__init__(name)
        self.data = data
        self.model = model

    def make_move(self, view: PlayerView) -> Move | None:
        features, legal_moves = encode_view(view)

        with torch.inference_mode():
            policy_logits, _ = self.model(features.unsqueeze(0))
            batched_legal_moves = legal_moves.unsqueeze(0)

            distribution = make_policy_distribution(
                policy_logits,
                batched_legal_moves,
            )
            move_id = distribution.sample().item()

        self.data.append(SelfPlayRecord(features, legal_moves, move_id))

        return decode_move(move_id, list(view.possible_moves))


def train_on_records(
    model: PolicyValueModel,
    records: list[SelfPlayRecord],
    optimizer: torch.optim.Optimizer,
) -> ReinforcementTrainingMetrics:
    model.train()
    device = next(model.parameters()).device

    features = torch.stack([record.features for record in records]).to(device)

    legal_moves = torch.stack([record.legal_moves for record in records]).to(device)

    actions = torch.tensor(
        [record.action for record in records], dtype=torch.long, device=device
    )
    rewards = torch.tensor(
        [record.reward for record in records], dtype=torch.float32, device=device
    )

    policy_logits, predicted_values = model(features)
    distribution = make_policy_distribution(policy_logits, legal_moves)

    chosen_log_probabilities = distribution.log_prob(actions)
    entropy = distribution.entropy().mean()

    advantages = rewards - predicted_values.detach()
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    policy_loss = -(chosen_log_probabilities * advantages).mean()
    value_loss = F.mse_loss(predicted_values, rewards)

    combined_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    optimizer.zero_grad()
    combined_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return ReinforcementTrainingMetrics(
        combined_loss=combined_loss.item(),
        policy_loss=policy_loss.item(),
        value_loss=value_loss.item(),
        entropy=entropy.item(),
    )


def train(
    batch_count: int,
    games_per_batch: int,
    opponent_types: list[PlayerArgs],
    hand_sizes: list[int],
    *,
    initial_model_path: Path,
    output_model_path: Path,
) -> None:
    if torch.backends.mps.is_available():
        training_device = torch.device("mps")
    elif torch.cuda.is_available():
        training_device = torch.device("cuda")
    else:
        training_device = torch.device("cpu")

    print(f"Training on {training_device}")

    training_model = PolicyValueModel().to(training_device)
    state_dict = torch.load(
        initial_model_path,
        map_location="cpu",
        weights_only=True,
    )
    training_model.load_state_dict(state_dict)
    print(f"Loaded initial policy from {initial_model_path}")

    inference_model = PolicyValueModel().cpu()
    inference_model.eval()
    inference_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(training_model.parameters(), lr=LEARNING_RATE)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    for batch_id in range(batch_count):
        batch_data: list[SelfPlayRecord] = []
        self_play_games = 0
        anchor_games = 0
        anchor_wins = 0

        inference_model.load_state_dict(
            {
                name: tensor.detach().cpu()
                for name, tensor in training_model.state_dict().items()
            }
        )
        inference_model.eval()

        for _ in range(games_per_batch):
            hand_size = choice(hand_sizes)
            is_self_play = random() < SELF_PLAY_FRACTION

            if is_self_play:
                recording_players = [
                    SelfPlayRecorder(name="Policy 1", data=[], model=inference_model),
                    SelfPlayRecorder(name="Policy 2", data=[], model=inference_model),
                ]
                players: list[Player] = list(recording_players)
                self_play_games += 1
            else:
                recording_player = SelfPlayRecorder(
                    name="Policy", data=[], model=inference_model
                )
                opponent_type, opponent_args = choice(opponent_types)
                opponent = opponent_type(*opponent_args)

                recording_players = [recording_player]
                players = [recording_player, opponent]
                anchor_games += 1

            shuffle(players)
            winner_name = President(
                players,
                cards_per_player=hand_size,
            ).run()[0]

            for recording_player in recording_players:
                reward = 1.0 if winner_name == recording_player.name else -1.0

                for record in recording_player.data:
                    record.reward = reward

                batch_data.extend(recording_player.data)

            if not is_self_play:
                anchor_wins += int(winner_name == recording_players[0].name)

        results = train_on_records(training_model, batch_data, optimizer)
        anchor_win_rate = f"{anchor_wins / anchor_games:.2%}" if anchor_games else "n/a"

        print(
            f"Batch {batch_id + 1:02d}/{batch_count:02d}\n"
            f"  rollout | self-play games: {self_play_games:,} | "
            f"anchor games: {anchor_games:,} | "
            f"anchor win rate: {anchor_win_rate} | "
            f"positions: {len(batch_data):,}\n"
            f"  train   | total loss: {results.combined_loss:.4f} | "
            f"policy loss: {results.policy_loss:.4f} | "
            f"value loss: {results.value_loss:.4f} | "
            f"entropy: {results.entropy:.4f}"
        )

        torch.save(training_model.state_dict(), output_model_path)


if __name__ == "__main__":
    train(
        batch_count=BATCH_COUNT,
        games_per_batch=GAMES_PER_BATCH,
        opponent_types=[
            (RandomBot, ("",)),
            (MinimalCardBot, ("",)),
            (ModelBot, ("", SUPERVISED_MODEL_PATH)),
        ],
        hand_sizes=[10, 15, 20, 26],
        initial_model_path=SUPERVISED_MODEL_PATH,
        output_model_path=REINFORCEMENT_MODEL_PATH,
    )

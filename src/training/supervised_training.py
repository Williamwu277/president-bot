from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .generate_data import load_examples
from .model import SUPERVISED_MODEL_PATH, PolicyValueModel

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCH_COUNT = 30
VALUE_LOSS_WEIGHT = 0.5
TRAINING_DATA_PATH = Path("data/training_data.pt")
VALIDATION_DATA_PATH = Path("data/validation_data.pt")


@dataclass(frozen=True)
class TrainingMetrics:
    combined_loss: float
    policy_loss: float
    value_loss: float
    policy_accuracy: float
    value_direction_accuracy: float


def make_data_loader(
    path: Path,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    features, legal_moves, target_moves, outcome = load_examples(path)

    dataset = TensorDataset(features, legal_moves, target_moves, outcome)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: PolicyValueModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> TrainingMetrics:
    model.train()

    total_combined_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_policy_correct = 0
    total_value_direction_correct = 0
    total_examples = 0

    for features, legal_moves, target_moves, outcomes in loader:
        features = features.to(device)
        legal_moves = legal_moves.to(device)
        target_moves = target_moves.to(device)
        outcomes = outcomes.to(device)

        policy_logits, predicted_values = model(features)

        masked_logits = policy_logits.masked_fill(
            ~legal_moves,
            torch.finfo(policy_logits.dtype).min,
        )

        policy_loss = nn.functional.cross_entropy(masked_logits, target_moves)
        value_loss = nn.functional.mse_loss(predicted_values, outcomes)
        combined_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_combined_loss += combined_loss.item() * batch_size
        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size
        total_policy_correct += (
            (masked_logits.argmax(dim=1) == target_moves).sum().item()
        )
        total_value_direction_correct += (
            ((predicted_values > 0) == (outcomes > 0)).sum().item()
        )
        total_examples += batch_size

    return TrainingMetrics(
        combined_loss=total_combined_loss / total_examples,
        policy_loss=total_policy_loss / total_examples,
        value_loss=total_value_loss / total_examples,
        policy_accuracy=total_policy_correct / total_examples,
        value_direction_accuracy=total_value_direction_correct / total_examples,
    )


def evaluate(
    model: PolicyValueModel,
    loader: DataLoader,
    device: torch.device,
) -> TrainingMetrics:
    model.eval()

    total_combined_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_policy_correct = 0
    total_value_direction_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, legal_moves, target_moves, outcomes in loader:
            features = features.to(device)
            legal_moves = legal_moves.to(device)
            target_moves = target_moves.to(device)
            outcomes = outcomes.to(device)

            policy_logits, predicted_values = model(features)

            masked_logits = policy_logits.masked_fill(
                ~legal_moves,
                torch.finfo(policy_logits.dtype).min,
            )

            policy_loss = nn.functional.cross_entropy(masked_logits, target_moves)
            value_loss = nn.functional.mse_loss(predicted_values, outcomes)
            combined_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            batch_size = features.shape[0]
            total_combined_loss += combined_loss.item() * batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            total_policy_correct += (
                (masked_logits.argmax(dim=1) == target_moves).sum().item()
            )
            total_value_direction_correct += (
                ((predicted_values > 0) == (outcomes > 0)).sum().item()
            )
            total_examples += batch_size

    return TrainingMetrics(
        combined_loss=total_combined_loss / total_examples,
        policy_loss=total_policy_loss / total_examples,
        value_loss=total_value_loss / total_examples,
        policy_accuracy=total_policy_correct / total_examples,
        value_direction_accuracy=total_value_direction_correct / total_examples,
    )


def train() -> None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Training on {device}")

    train_loader = make_data_loader(
        TRAINING_DATA_PATH, batch_size=BATCH_SIZE, shuffle=True
    )

    validation_loader = make_data_loader(
        VALIDATION_DATA_PATH, batch_size=BATCH_SIZE, shuffle=False
    )

    model = PolicyValueModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_validation_loss = float("inf")

    for epoch in range(EPOCH_COUNT):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1:02d}\n"
            f"  train      | total loss: {train_metrics.combined_loss:.4f} | "
            f"policy loss: {train_metrics.policy_loss:.4f} | "
            f"value loss: {train_metrics.value_loss:.4f} | "
            f"policy accuracy: {train_metrics.policy_accuracy:.2%} | "
            "value direction accuracy: "
            f"{train_metrics.value_direction_accuracy:.2%}\n"
            "  validation | "
            f"total loss: {validation_metrics.combined_loss:.4f} | "
            f"policy loss: {validation_metrics.policy_loss:.4f} | "
            f"value loss: {validation_metrics.value_loss:.4f} | "
            f"policy accuracy: {validation_metrics.policy_accuracy:.2%} | "
            "value direction accuracy: "
            f"{validation_metrics.value_direction_accuracy:.2%}"
        )

        if validation_metrics.combined_loss < best_validation_loss:
            best_validation_loss = validation_metrics.combined_loss
            SUPERVISED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SUPERVISED_MODEL_PATH)


if __name__ == "__main__":
    train()

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .generate_data import load_examples
from .supervised_model import MODEL_PATH, SupervisedModel

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCH_COUNT = 30
TRAINING_DATA_PATH = Path("data/test_train.pt")
VALIDATION_DATA_PATH = Path("data/test_validation.pt")


def make_data_loader(
    path: Path,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    features, legal_moves, target_moves = load_examples(path)

    dataset = TensorDataset(features, legal_moves, target_moves)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: SupervisedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features, legal_moves, target_moves in loader:
        features = features.to(device)
        legal_moves = legal_moves.to(device)
        target_moves = target_moves.to(device)

        logits = model(features)

        masked_logits = logits.masked_fill(~legal_moves, torch.finfo(logits.dtype).min)

        loss = nn.functional.cross_entropy(masked_logits, target_moves)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (masked_logits.argmax(dim=1) == target_moves).sum().item()
        total_examples += batch_size

    return (total_loss / total_examples, total_correct / total_examples)


def evaluate(
    model: SupervisedModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, legal_moves, target_moves in loader:
            features = features.to(device)
            legal_moves = legal_moves.to(device)
            target_moves = target_moves.to(device)

            logits = model(features)

            masked_logits = logits.masked_fill(
                ~legal_moves,
                torch.finfo(logits.dtype).min,
            )

            loss = nn.functional.cross_entropy(masked_logits, target_moves)

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (masked_logits.argmax(dim=1) == target_moves).sum().item()
            total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
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

    model = SupervisedModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_validation_loss = float("inf")

    for epoch in range(EPOCH_COUNT):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.2%} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    train()

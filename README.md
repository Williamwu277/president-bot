# President Bot

Ever since I was young, President—or at least some variation of it—has been one of my staple card games. I wanted to see whether I could discover any novel strategies by eventually training an ML model to play near-optimally.

## Table of Contents

- [Findings](#findings)
- [Rules](#rules)
- [Setup](#setup)
- [Operation](#operation)

## Findings

You can read the benchmarks in `benchmarks.run`, which uses seed `73`, to get an idea of the methodology. Here is the summary:

### First-player advantage

The minimax solver played each deal twice, swapping the hands while keeping the
same player first. With optimal play and perfect information, the first player
won approximately 61% of games.

| Cards per player | Deals tested | Games solved | First-player WR (original) | First-player WR (swapped) | Runtime (original) | Runtime (swapped) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 10,000 | 20,000 | 61.53% | 61.92% | 1.78s | 1.73s |
| 7 | 10,000 | 20,000 | 60.82% | 60.68% | 11.40s | 10.87s |
| 10 | 1,000 | 2,000 | 60.60% | 62.70% | 21.20s | 20.30s |
| 12 | 1,000 | 2,000 | 58.10% | 64.10% | 225.45s | 214.30s |

The larger-hand results use fewer deals because the exact minimax solver becomes
substantially slower as the number of cards increases.

### Models

* **Copycat-v1.0**: A model trained on the outputs of MinimalCardBot (which is a heuristic that mostly picks the smallest move each turn)
* **Jester-v1.0**: A model trained through self-play from random weights to beat the MinimalCardBot heuristic over 640,000 games
* **Jester-v1.1**: A model trained through self-play from the Copycat-v1.0 weights to beat both MinimalCardBot and Jester-v1.0 over 1,920,000 games

###  Pairwise Bot tournament

Each matchup at each hand size contains 2,000 two-player games. Every generated
pair of hands was played with both seating orders and with the hands swapped to
balance first-player and hand-quality advantages.

Each cell contains the **row model's win rate against the column model**, ordered
by **5 / 10 / 15 / 20 / 25 cards per player**.

| Y-axis ↓ / X-axis → | RandomBot | MinimalCardBot | Copycat-v1.0 | Jester-v1.0 | Jester-v1.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **RandomBot** | — | 38.70 / 22.60 / 10.85 / 7.60 / 6.35% | 37.25 / 19.95 / 12.60 / 7.90 / 6.45% | 37.40 / 21.45 / 12.00 / 7.65 / 5.70% | 37.40 / 20.65 / 11.10 / 8.45 / 6.45% |
| **MinimalCardBot** | 61.30 / 77.40 / 89.15 / 92.40 / 93.65% | — | 50.05 / 50.00 / 49.95 / 49.80 / 49.70% | 48.35 / 46.45 / 45.35 / 45.00 / 42.00% | 48.10 / 46.55 / 44.75 / 43.05 / 37.60% |
| **Copycat-v1.0** | 62.75 / 80.05 / 87.40 / 92.10 / 93.55% | 49.95 / 50.00 / 50.05 / 50.20 / 50.30% | — | 47.70 / 46.55 / 44.90 / 44.10 / 43.25% | 48.45 / 46.65 / 43.05 / 44.40 / 38.25% |
| **Jester-v1.0** | 62.60 / 78.55 / 88.00 / 92.35 / 94.30% | 51.65 / 53.55 / 54.65 / 55.00 / 58.00% | 52.30 / 53.45 / 55.10 / 55.90 / 56.75% | — | 50.30 / 48.50 / 49.30 / 47.25 / 47.80% |
| **Jester-v1.1** | 62.60 / 79.35 / 88.90 / 91.55 / 93.55% | 51.90 / 53.45 / 55.25 / 56.95 / 62.40% | 51.55 / 53.35 / 56.95 / 55.60 / 61.75% | 49.70 / 51.50 / 50.70 / 52.75 / 52.20% | — |

Copycat-v1.0 performs approximately as well as its MinimalCardBot teacher. Both
Jester versions increasingly outperform Copycat-v1.0 and MinimalCardBot as hand
size grows, while Jester-v1.1 holds a small advantage over Jester-v1.0 in most
hand sizes.

### Generalizing to Four Players

Interestingly, all the Jester models can play decently well in 4 players even though they were exclusively trained on 2 player games.

## Rules

This experiment is constructed under a specific variation of President that I've probably some-what made up. I've also added certain restrictions to make things easier on myself. Here are the rules:

* 1 deck of 52 cards without the Jokers
* 2-4 players

Game-play:

1. Deal the same number of cards to each player
2. A random player goes first
3. On their turn, they can put down a \**move*
4. Each successive player must put down a move of a greater \*\**denomination*. They can also pass
5. The standings are determined by the order in which players empty their hand

\* These are the possible types of *moves*

* 1 of a kind
* 2 of a kind
* 3 of a kind
* 4 of a kind is a bomb, which automatically flushes the current play
* A full house: 3 of a kind and 2 of a kind
* A straight: 5 consecutive cards (but no 2 allowed)
* pass

\*\* Moves are greater if they are the same type of move, but the card value is larger:

* In sorted order: 3, 4, ..., 10, J, Q, K, A, 2
* Suits do not matter
* Nothing beats a bomb
* Full houses first compare the triple's rank, then the pair's rank

## Setup

1. Setup the virtual environment
2. Install the requirements
3. Set up `pre-commit`

```bash
python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt
pre-commit install
pre-commit run --all-files
```

## Operation

Run from repository root.

To play with the bots on CLI:

```bash
python -m src.play
```

To run the benchmarks:

```bash
python -m benchmarks.run [command]
```

* `-WR` for the winrate benchmark
* `-T` for the pairwise tournament

To train a model: 

1. Go to `src.training.generate_data` and modify the training set parameters list to generate training and validation data. 
2. Then run the training script in `src.training.supervised_training` with the correct file names in the script.

```bash
python -m src.training.generate_data
python -m src.training.supervised_training
```

3. For reinforcement learning, modify the `src.training.reinforcement_learning` script with the base model and output path. Then call the command

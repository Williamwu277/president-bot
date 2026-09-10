# President Bot

Ever since I was young, President—or at least some variation of it—has been one of my staple card games. I wanted to see whether I could discover any novel strategies by eventually training an ML model to play near-optimally.

## Table of Contents

- [Findings](#findings)
- [Rules](#rules)
- [Setup](#setup)
- [Operation](#operation)

## Findings

TBD—no novel model trained just yet ... So far I've only created a model that imitates the `minimal_card_bot`. You can read the benchmarks in `benchmarks.run`, which uses seed `67`, to get an idea of the methodology but here is the summary:

### First-player advantage

The minimax solver played each deal twice, swapping the hands while keeping the
same player first. With optimal play and perfect information, the first player
won approximately 60% of games.

| Cards per player | Deals tested | Games solved | First-player WR (original) | First-player WR (swapped) |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 1,000 | 2,000 | 63.3% | 59.5% |
| 7 | 1,000 | 2,000 | 60.1% | 61.1% |
| 10 | 100 | 200 | 59.0% | 66.0% |
| 12 | 10 | 20 | 60.0% | 60.0% |

The larger-hand results use fewer deals because the exact minimax solver becomes
substantially slower as the number of cards increases.

### Bot tournament

Each row contains 10,000 two-player games. Every generated pair of hands was
played with both seating orders and with the hands swapped to balance first-player
and hand-quality advantages.

| Matchup | Cards per player | First bot win rate | Second bot win rate | Runtime |
| --- | ---: | ---: | ---: | ---: |
| RandomBot vs. MinimalCardBot | 10 | 21.42% | 78.58% | 4.44s |
| RandomBot vs. MinimalCardBot | 15 | 12.84% | 87.16% | 8.30s |
| RandomBot vs. MinimalCardBot | 20 | 7.99% | 92.01% | 12.90s |
| RandomBot vs. ModelBot | 10 | 22.52% | 77.48% | 20.57s |
| RandomBot vs. ModelBot | 15 | 12.28% | 87.72% | 32.76s |
| RandomBot vs. ModelBot | 20 | 8.63% | 91.37% | 47.93s |
| MinimalCardBot vs. ModelBot | 10 | 50.56% | 49.44% | 24.29s |
| MinimalCardBot vs. ModelBot | 15 | 50.68% | 49.32% | 35.99s |
| MinimalCardBot vs. ModelBot | 20 | 51.08% | 48.92% | 46.77s |

The imitation-trained ModelBot performs approximately as well as its
MinimalCardBot teacher. Both substantially outperform RandomBot, especially as
the number of cards increases.

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
python -m benchmarks.run
```

To train a model: 

1. Go to `src.training.generate_data` and modify the training set parameters list to generate training and validation data. 
2. Then run the training script in `src.training.supervised_training` with the correct file names in the script.

```bash
python -m src.training.generate_data
python -m src.training.supervised_training
```
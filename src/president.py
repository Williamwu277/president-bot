from enum import Enum, auto
from random import shuffle
from dataclasses import dataclass
from collections import Counter, defaultdict


class Rank(Enum):
    THREE = 3
    FOUR = auto()
    FIVE = auto()
    SIX = auto()
    SEVEN = auto()
    EIGHT = auto()
    NINE = auto()
    TEN = auto()
    JACK = auto()
    QUEEN = auto()
    KING = auto()
    ACE = auto()
    TWO = auto()

    def __str__(self) -> str:
        match self:
            case Rank.ACE: return "A"
            case Rank.TWO: return "2"
            case Rank.JACK: return "J"
            case Rank.QUEEN: return "Q"
            case Rank.KING: return "K"
            case _: return str(self.value)


class Suit(Enum):
    DIAMONDS = "♢"
    CLUBS = "♣"
    HEARTS = "♡"
    SPADES = "♠"

    def __str__(self) -> str:
        return self.value


class MoveType(Enum):
    SINGLE = auto()
    DOUBLE = auto()
    TRIPLE = auto()
    FULL_HOUSE = auto()
    STRAIGHT = auto()
    BOMB = auto()


@dataclass
class Card:
    rank: Rank
    suit: Suit

    def __repr__(self) -> str:
        return f"{str(self.rank)}{str(self.suit)}"

    def beats(self, other: "Card") -> bool:
        return self.rank.value > other.rank.value


@dataclass(init=False)
class Move:
    cards: tuple[Card, ...]
    move_type: MoveType

    def __init__(self, cards: tuple[Card]):
        self.cards = sorted(cards, key=lambda card: card.rank.value)

        # Move Validation
        rank_counts = Counter(card.rank for card in self.cards)
        counts = sorted(rank_counts.values())

        match counts:
            case [1]:
                self.move_type = MoveType.SINGLE
            case [2]:
                self.move_type = MoveType.DOUBLE
            case [3]:
                self.move_type = MoveType.TRIPLE
            case [4]:
                self.move_type = MoveType.BOMB
            case [2, 3]:
                self.move_type = MoveType.FULL_HOUSE
            case [1, 1, 1, 1, 1]:
                if Rank.TWO in rank_counts:
                    raise ValueError("Invalid Move")

                self.move_type = MoveType.STRAIGHT

    def __repr__(self) -> str:
        return f"({' '.join(map(str, self.cards))})"

    def beats(self, other: "Move") -> bool:
        if self.move_type is not other.move_type:
            # One must be a bomb
            if MoveType.BOMB not in [self.move_type, other.move_type]:
                return False
            return self.move_type is MoveType.BOMB
        
        elif self.move_type is MoveType.BOMB and other.move_type is MoveType.BOMB:
            return False
        
        elif self.move_type is MoveType.FULL_HOUSE:
            double, triple = self.cards[0], self.cards[-1] 
            other_double, other_triple = other.cards[0], other.cards[-1]

            if self.cards[2] == self.cards[0]:
                double, triple = triple, double
            if other.cards[2] == other.cards[0]:
                other_double, other_triple = other_triple, other_double

            if triple == other_triple:
                return double.beats(other_double)

            return triple.beats(other_triple)
        
        return self.cards[0].beats(other.cards[0])


class Hand:

    def __init__(self, cards: list[Card]):
        self.cards = sorted(cards, key=lambda card: card.rank.value)

    def __str__(self) -> str:
        return ' '.join(map(str, self.cards))

    def get_all_moves(self) -> list[Move]:
        card_ranks: defaultdict[Rank, list[Card]] = defaultdict(list)

        # It doesn't matter which card is picked. We don't need to differentiate between suits
        for card in self.cards:
            card_ranks[card.rank].append(card)

        moves = []

        # Single, double, triple, bomb
        for rank in card_ranks:
            for i in range(1, len(card_ranks[rank]) + 1):
                moves.append(Move(tuple(card_ranks[rank][0:i])))

        # Straight
        straight = []
        for rank in card_ranks:
            if rank is Rank.TWO:
                break

            straight.append(card_ranks[rank][0])
            if len(straight) > 5:
                straight.pop(0)

            if len(straight) == 5 and (straight[-1].rank.value - straight[0].rank.value) == 4:
                moves.append(Move(tuple(straight)))

        # Full house
        for first_rank in card_ranks:
            for second_rank in card_ranks:
                if first_rank is not second_rank and len(card_ranks[first_rank]) >= 3 and len(card_ranks[second_rank]) >= 2:
                    moves.append(Move(tuple(card_ranks[first_rank][0:3] + card_ranks[second_rank][0:2])))

        return moves

    def get_possible_moves(self, current_move: Move) -> list[Move]:
        all_moves = self.get_all_moves()
        return list(filter(lambda move: move.beats(current_move), all_moves))


class President:

    def __init__(self):
        pass


def get_full_deck(shuffled: bool = False) -> list[Card]:
    deck = []

    for rank in Rank:
        for suit in Suit:
            deck.append(Card(rank, suit))

    if shuffled:
        shuffle(deck)

    return deck


deck = get_full_deck(shuffled=True)
hand = Hand(deck[0:len(deck)//4])
print(hand.get_possible_moves(
    Move((
        Card(Rank.EIGHT, Suit.SPADES),
    ))
))

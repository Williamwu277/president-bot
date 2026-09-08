"""
Core models and game engine for President.
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from random import shuffle

FULL_DECK_SIZE = 52


class Rank(Enum):
    """
    Card ranks ordered from lowest to highest.
    """

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
            case Rank.ACE:
                return "A"
            case Rank.TWO:
                return "2"
            case Rank.JACK:
                return "J"
            case Rank.QUEEN:
                return "Q"
            case Rank.KING:
                return "K"
            case _:
                return str(self.value)


class Suit(Enum):
    """
    The four standard card suits.
    """

    DIAMONDS = "♢"
    CLUBS = "♣"
    HEARTS = "♡"
    SPADES = "♠"

    def __str__(self) -> str:
        return self.value


class MoveType(Enum):
    """
    Valid categories of plays.
    """

    SINGLE = auto()
    DOUBLE = auto()
    TRIPLE = auto()
    FULL_HOUSE = auto()
    STRAIGHT = auto()
    BOMB = auto()


MoveKey = tuple[Rank, ...]
HandKey = tuple[int, ...]


CARD_SORT_ORDER = lambda card: card.rank.value


@dataclass(frozen=True)
class Card:
    """
    An immutable playing card.
    """

    rank: Rank
    suit: Suit

    def __repr__(self) -> str:
        return f"{self.rank!s}{self.suit!s}"

    def beats(self, other: "Card") -> bool:
        return self.rank.value > other.rank.value


@dataclass(frozen=True)
class Move:
    """
    An immutable, validated collection of cards played together.
    """

    cards: tuple[Card, ...]
    move_type: MoveType = field(init=False)

    def __post_init__(self):
        cards = tuple(sorted(self.cards, key=CARD_SORT_ORDER))

        # Move Validation
        rank_counts = Counter(card.rank for card in self.cards)
        counts = sorted(rank_counts.values())
        move_type = None

        match counts:
            case [1]:
                move_type = MoveType.SINGLE
            case [2]:
                move_type = MoveType.DOUBLE
            case [3]:
                move_type = MoveType.TRIPLE
            case [4]:
                move_type = MoveType.BOMB
            case [2, 3]:
                move_type = MoveType.FULL_HOUSE
            case [1, 1, 1, 1, 1]:
                if (
                    Rank.TWO in rank_counts
                    or (cards[-1].rank.value - cards[0].rank.value) != 4
                ):
                    raise ValueError("Invalid Move")

                move_type = MoveType.STRAIGHT
            case _:
                raise ValueError("Invalid Move")

        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "move_type", move_type)

    def __repr__(self) -> str:
        return f"({' '.join(map(str, self.cards))})"

    def __len__(self) -> int:
        return len(self.cards)

    def get_key(self) -> MoveKey:
        return tuple(card.rank for card in self.cards)

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

            if self.cards[2].rank == self.cards[0].rank:
                double, triple = triple, double
            if other.cards[2].rank == other.cards[0].rank:
                other_double, other_triple = other_triple, other_double

            if triple.rank == other_triple.rank:
                return double.beats(other_double)

            return triple.beats(other_triple)

        return self.cards[0].beats(other.cards[0])


class Hand:
    """
    A player's mutable collection of cards.
    """

    def __init__(self, cards: list[Card]):
        self.cards: list[Card] = sorted(cards, key=CARD_SORT_ORDER)

    def __str__(self) -> str:
        return " ".join(map(str, self.cards))

    def __len__(self) -> int:
        return len(self.cards)

    def is_empty(self) -> bool:
        return len(self.cards) == 0

    def get_key(self) -> HandKey:
        counts = Counter(card.rank for card in self.cards)
        return tuple(counts[rank] for rank in Rank)

    def add_cards(self, move: Move | None):
        if move is None:
            return
        self.cards.extend(move.cards)
        self.cards.sort(key=CARD_SORT_ORDER)

    def remove_cards(self, move: Move | None):
        if move is None:
            return
        for card in move.cards:
            self.cards.remove(card)

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

            if (
                len(straight) == 5
                and (straight[-1].rank.value - straight[0].rank.value) == 4
            ):
                moves.append(Move(tuple(straight)))

        # Full house
        for first_rank in card_ranks:
            for second_rank in card_ranks:
                if (
                    first_rank is not second_rank
                    and len(card_ranks[first_rank]) >= 3
                    and len(card_ranks[second_rank]) >= 2
                ):
                    moves.append(
                        Move(
                            tuple(
                                card_ranks[first_rank][0:3]
                                + card_ranks[second_rank][0:2]
                            )
                        )
                    )

        return moves

    def get_possible_moves(self, current_move: Move | None) -> list[Move]:
        all_moves = self.get_all_moves()
        if current_move:
            return list(filter(lambda move: move.beats(current_move), all_moves))
        return all_moves


PlayerId = int


@dataclass(frozen=True)
class TurnRecord:
    """
    A record of a turn made by one player.
    """

    player_id: PlayerId
    move: Move | None


@dataclass(frozen=True)
class PlayerView:
    """
    An immutable snapshot provided to a player for one turn.
    """

    player_id: PlayerId
    hand: tuple[Card, ...]
    cards_remaining: tuple[int, ...]
    current_move: Move | None
    current_trick: tuple[TurnRecord, ...]
    trick_history: tuple[tuple[TurnRecord, ...], ...]
    possible_moves: tuple[Move, ...]


class Player(ABC):
    """
    Base class for human, AI, and other player strategies.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def make_move(self, view: PlayerView) -> Move | None:
        """
        Return a legal move, or None to pass.
        """


@dataclass
class PlayerState:
    """Mutable engine state associated with one player."""

    player_id: PlayerId
    controller: Player
    hand: Hand


class President:
    """
    Run a game of President and coordinate player turns.

    When ``initial_hands`` is provided, use those hands as a preset deal and
    bypass deck creation, shuffling, and automatic dealing.
    """

    def __init__(
        self,
        players: list[Player],
        deck_count: int = 1,
        cards_per_player: int | None = None,
        initial_hands: list[list[Card]] | None = None,
    ):
        player_count = len(players)
        self.players: dict[PlayerId, PlayerState] = {}
        self.turn_order: deque[PlayerId] = deque(range(player_count))
        self.standings: list[PlayerId] = []
        self.trick_history: list[tuple[TurnRecord, ...]] = []
        self.current_trick: list[TurnRecord] = []
        self.current_move: Move | None = None
        self.last_player_to_play: PlayerId | None = None
        self.passed_players: set[PlayerId] = set()

        if player_count <= 1:
            raise ValueError("Not Enough Players")

        if initial_hands is not None:
            if len(initial_hands) != player_count:
                raise ValueError("Each Player Needs an Initial Hand")
            hands = [Hand(cards) for cards in initial_hands]
        else:
            cards = get_full_deck() * deck_count
            shuffle(cards)

            if cards_per_player and len(cards) < player_count * cards_per_player:
                raise ValueError("Not Enough Cards")

            if not cards_per_player:
                cards_per_player = len(cards) // player_count

            hands = [
                Hand(
                    cards[
                        player_id * cards_per_player : (player_id + 1)
                        * cards_per_player
                    ]
                )
                for player_id in range(player_count)
            ]

        for player_id, (controller, hand) in enumerate(zip(players, hands)):
            self.players[player_id] = PlayerState(player_id, controller, hand)

    def next_turn(self) -> TurnRecord:
        """
        Simulate one turn.
        """

        player_id = self.turn_order[0]
        player = self.players[player_id]
        possible_moves = tuple(player.hand.get_possible_moves(self.current_move))

        view = PlayerView(
            player_id=player_id,
            hand=tuple(player.hand.cards),
            cards_remaining=tuple(
                len(self.players[current_player_id].hand.cards)
                for current_player_id in range(len(self.players))
            ),
            current_move=self.current_move,
            current_trick=tuple(self.current_trick),
            trick_history=tuple(self.trick_history),
            possible_moves=possible_moves,
        )

        move = player.controller.make_move(view)

        # Move validation
        if move is None:
            if self.current_move is None:
                raise ValueError("Cannot Pass When Starting a Trick")
            self.passed_players.add(player_id)
        elif move in possible_moves:
            player.hand.remove_cards(move)
            self.current_move = move
            self.last_player_to_play = player_id
            self.passed_players.clear()
        else:
            raise ValueError("Invalid Move Played")

        turn_record = TurnRecord(player_id, move)
        self.current_trick.append(turn_record)

        # Check for winner
        if not player.hand.cards:
            self.standings.append(player_id)
            self.turn_order.popleft()
        else:
            self.turn_order.rotate(-1)

        # Finish the trick once every active player except the last player to
        # play has passed. If that player went out, every active player passes.
        players_who_must_pass = set(self.turn_order)
        if self.last_player_to_play is not None:
            players_who_must_pass.discard(self.last_player_to_play)

        if (
            self.current_move is not None
            and players_who_must_pass <= self.passed_players
        ):
            self.trick_history.append(tuple(self.current_trick))
            self.current_trick.clear()
            self.current_move = None
            self.last_player_to_play = None
            self.passed_players.clear()

        return turn_record

    def run(self) -> list[str]:
        """
        Play until the game ends and return standings.
        """

        while len(self.turn_order) > 1:
            self.next_turn()

        self.standings.append(self.turn_order.popleft())
        return [self.players[player_id].controller.name for player_id in self.standings]


def get_full_deck(shuffled: bool = False) -> list[Card]:
    """
    Return one standard 52-card deck without Jokers.
    """

    deck = []

    for rank in Rank:
        for suit in Suit:
            deck.append(Card(rank, suit))

    if shuffled:
        shuffle(deck)

    return deck

from enum import Enum, auto
from random import shuffle, randint
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict, deque
from abc import ABC, abstractmethod


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


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __repr__(self) -> str:
        return f"{str(self.rank)}{str(self.suit)}"

    def beats(self, other: "Card") -> bool:
        return self.rank.value > other.rank.value


@dataclass(frozen=True)
class Move:
    cards: tuple[Card, ...]
    move_type: MoveType = field(init=False)

    def __post_init__(self):
        cards = tuple(sorted(self.cards, key=lambda card: card.rank.value))

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
                if Rank.TWO in rank_counts:
                    raise ValueError("Invalid Move")

                move_type = MoveType.STRAIGHT
            case _:
                raise ValueError("Invalid Move")

        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "move_type", move_type)

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
        self.cards: list[Card] = sorted(cards, key=lambda card: card.rank.value)

    def __str__(self) -> str:
        return ' '.join(map(str, self.cards))

    def add_cards(self, move: Move):
        self.cards.extend(move.cards)

    def remove_cards(self, move: Move):
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

            if len(straight) == 5 and (straight[-1].rank.value - straight[0].rank.value) == 4:
                moves.append(Move(tuple(straight)))

        # Full house
        for first_rank in card_ranks:
            for second_rank in card_ranks:
                if first_rank is not second_rank and len(card_ranks[first_rank]) >= 3 and len(card_ranks[second_rank]) >= 2:
                    moves.append(Move(tuple(card_ranks[first_rank][0:3] + card_ranks[second_rank][0:2])))

        return moves

    def get_possible_moves(self, current_move: Move | None) -> list[Move]:
        all_moves = self.get_all_moves()
        if current_move:
            return list(filter(lambda move: move.beats(current_move), all_moves))
        return all_moves


PlayerId = int


@dataclass(frozen=True)
class TurnRecord:
    player_id: PlayerId
    move: Move | None


@dataclass(frozen=True)
class PlayerView:
    player_id: PlayerId
    hand: tuple[Card, ...]
    current_move: Move | None
    current_trick: tuple[TurnRecord, ...]
    trick_history: tuple[tuple[TurnRecord, ...], ...]
    possible_moves: tuple[Move, ...]


class Player(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def make_move(self, view: PlayerView) -> Move | None:
        pass


@dataclass
class PlayerState:
    player_id: PlayerId
    controller: Player
    hand: Hand


class President:

    def __init__(
        self,
        players: list[Player],
        deck_count: int = 1,
        cards_per_player: Optional[int] = None
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

        cards = get_full_deck() * deck_count
        shuffle(cards)

        if cards_per_player and len(cards) < player_count * cards_per_player:
            raise ValueError("Not Enough Cards")

        if not cards_per_player:
            cards_per_player = len(cards) // player_count

        for player_id, controller in enumerate(players):
            hand = Hand(cards[player_id * cards_per_player : (player_id + 1) * cards_per_player])
            self.players[player_id] = PlayerState(player_id, controller, hand)
            # print(f"{controller.name} hand: {hand}")

        # print(f"Starting new game with {player_count} players, each with {cards_per_player} cards")

    def next_turn(self) -> TurnRecord:
        player_id = self.turn_order[0]
        player = self.players[player_id]
        possible_moves = tuple(player.hand.get_possible_moves(self.current_move))

        view = PlayerView(
            player_id=player_id,
            hand=tuple(player.hand.cards),
            current_move=self.current_move,
            current_trick=tuple(self.current_trick),
            trick_history=tuple(self.trick_history),
            possible_moves=possible_moves
        )

        # print(f"{player.controller.name}'s turn")
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

        # print(move if move is not None else "Pass")
        turn_record = TurnRecord(player_id, move)
        self.current_trick.append(turn_record)

        # Check for winner
        if not player.hand.cards:
            # print(f"{player.controller.name} wins!")
            self.standings.append(player_id)
            self.turn_order.popleft()
        else:
            self.turn_order.rotate(-1)

        # Finish the trick once every active player except the last player to
        # play has passed. If that player went out, every active player passes.
        players_who_must_pass = set(self.turn_order)
        if self.last_player_to_play is not None:
            players_who_must_pass.discard(self.last_player_to_play)

        if self.current_move is not None and players_who_must_pass <= self.passed_players:
            self.trick_history.append(tuple(self.current_trick))
            self.current_trick.clear()
            self.current_move = None
            self.last_player_to_play = None
            self.passed_players.clear()
            # print("All players have passed. Starting new trick")

        return turn_record

    def run(self) -> list[str]:
        while len(self.turn_order) > 1:
            self.next_turn()

        self.standings.append(self.turn_order.popleft())

        # print("Player standings:")
        #for place, player_id in enumerate(self.standings, start=1):
            # print(f"{place}: {self.players[player_id].controller.name}")

        return [self.players[player_id].controller.name for player_id in self.standings]


class RandomPlayer(Player):

    def make_move(self, view: PlayerView) -> Move | None:
        if not view.possible_moves:
            return None
        move_index = randint(0, len(view.possible_moves) - 1)
        return view.possible_moves[move_index]


def get_full_deck(shuffled: bool = False) -> list[Card]:
    deck = []

    for rank in Rank:
        for suit in Suit:
            deck.append(Card(rank, suit))

    if shuffled:
        shuffle(deck)

    return deck


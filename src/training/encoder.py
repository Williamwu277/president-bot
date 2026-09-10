from collections import defaultdict

import torch

from src.president import (
    FULL_DECK_SIZE,
    Card,
    Hand,
    Move,
    MoveKey,
    PlayerId,
    PlayerView,
    Rank,
    TurnRecord,
    get_full_deck,
)

MOVE_COUNT = 217
SCHEMA_VERSION = 2
MAX_PLAYERS = 4
MIN_PLAYERS = 2
RANK_COUNT = len(Rank)
OPPONENT_SLOT_SIZE = 3 + RANK_COUNT + MOVE_COUNT
CURRENT_TRICK_STATE_SIZE = 2 * MAX_PLAYERS
STATE_SIZE = (
    MOVE_COUNT
    + CURRENT_TRICK_STATE_SIZE
    + (2 * RANK_COUNT)
    + ((MAX_PLAYERS - 1) * OPPONENT_SLOT_SIZE)
)


def get_move_encoding_map() -> dict[MoveKey | None, int]:
    """
    Get the map of a move to stable numerical ID.
    """
    move_encoding_map = {None: 0}
    deck = get_full_deck()
    hand = Hand(deck)
    moves = hand.get_all_moves()
    if MOVE_COUNT != len(moves) + 1:
        raise RuntimeError(
            f"Expected {MOVE_COUNT - 1} playable moves, found {len(moves)}"
        )
    for move_id, move in enumerate(moves, start=1):
        move_encoding_map[move.get_key()] = move_id
    return move_encoding_map


move_encoding_map = get_move_encoding_map()


def find_pass_map(
    trick_history: list[tuple[TurnRecord, ...]],
) -> defaultdict[PlayerId, list[Move]]:
    """
    Find all the moves that each player has passed on.
    """
    pass_map = defaultdict(list)
    for trick in trick_history:
        previous_move = None
        for record in trick:
            if previous_move is not None and record.move is None:
                pass_map[record.player_id].append(previous_move)
            elif record.move is not None:
                previous_move = record.move
    return pass_map


def find_played_card_map(
    trick_history: list[tuple[TurnRecord, ...]],
) -> defaultdict[PlayerId, list[Card]]:
    """
    Find all cards played by each player.
    """
    played_card_map = defaultdict(list)
    for trick in trick_history:
        for record in trick:
            if record.move is not None:
                played_card_map[record.player_id].extend(record.move.cards)
    return played_card_map


def find_current_move_owner_and_passed_players(
    current_trick: list[TurnRecord],
) -> tuple[PlayerId | None, set[PlayerId]]:
    """
    Find who made the current move and who has passed since it was made.
    """
    current_move_owner = None
    passed_players = set()
    for record in current_trick:
        if record.move is None:
            if current_move_owner is not None:
                passed_players.add(record.player_id)
        else:
            current_move_owner = record.player_id
            passed_players.clear()
    return current_move_owner, passed_players


def encode_move(move: Move | None) -> int:
    """
    Get the stable numerical ID of a move.
    """
    if move is None:
        return 0
    return move_encoding_map[move.get_key()]


def decode_move(move_id: int, possible_moves: list[Move]) -> Move | None:
    """
    Get the move corresponding to the stable numerical ID.
    """
    if move_id == 0:
        return None
    for move in possible_moves:
        if move_encoding_map[move.get_key()] == move_id:
            return move
    raise ValueError("Invalid Move ID")


def encode_pass_map(
    pass_map: defaultdict[PlayerId, list[Move]],
) -> defaultdict[PlayerId, list[float]]:
    """
    Encode the pass map. Each float has a denomination of [0, 0.25, 0.5, 0.75, 1]
    for how many times the move was passed (up to 4 times).
    """
    encoded_map = defaultdict(lambda: [0] * MOVE_COUNT)
    for player_id in pass_map:
        for move in pass_map[player_id]:
            encoded_move = encode_move(move)
            encoded_map[player_id][encoded_move] += 0.25
            if encoded_map[player_id][encoded_move] > 1.0:
                raise ValueError(
                    f"Player {player_id} passed on move {move.get_key()} more "
                    "than four times"
                )
    return encoded_map


def encode_hand(hand: list[Card]) -> list[float]:
    """
    Encode a hand into a vector of 13 different ranks with denominations of
    [0, 0.25, 0.5, 0.75, 1] for how many cards of each rank (up to 4).
    """
    counts = defaultdict(int)
    for card in hand:
        counts[card.rank] += 1
        if counts[card.rank] > 4:
            raise ValueError(f"Hand contains more than four {card.rank} cards")
    return [counts[rank] / 4 for rank in Rank]


def encode_view(view: PlayerView) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a player's game state into model inputs.

    The 950 features contain: current move [217], current-move owner [4],
    players passed since that move [4], own hand [13], own played ranks [13],
    and three opponent slots [233 each: exists, active, cards remaining,
    played ranks [13], and pass history [217]].
    """
    player_count = len(view.cards_remaining)

    if not MIN_PLAYERS <= player_count <= MAX_PLAYERS:
        raise ValueError(
            f"Encoder supports {MIN_PLAYERS} to {MAX_PLAYERS} players, "
            f"received {player_count}"
        )
    elif len(view.turn_order) > MAX_PLAYERS:
        raise ValueError(f"Turn order contains more than {MAX_PLAYERS} active players")
    elif not view.turn_order:
        raise ValueError("Turn order cannot be empty")
    elif view.turn_order[0] != view.player_id:
        raise ValueError("Acting player must be first in turn order")

    feature_values: list[float] = []

    # Encode the current move [217]
    current_move_values = [0.0] * MOVE_COUNT
    current_move_values[encode_move(view.current_move)] = 1.0
    feature_values.extend(current_move_values)

    # Encode current-move ownership [4] and passes since that move [4].
    # Slots are ordered as self followed by original-relative opponents.
    relative_player_ids = [
        view.player_id,
        *(
            (view.player_id + offset) % player_count
            for offset in range(1, player_count)
        ),
    ]
    current_move_owner, currently_passed_players = (
        find_current_move_owner_and_passed_players(view.current_trick)
    )
    feature_values.extend(
        float(player_id == current_move_owner) for player_id in relative_player_ids
    )
    feature_values.extend([0.0] * (MAX_PLAYERS - player_count))
    feature_values.extend(
        float(player_id in currently_passed_players)
        for player_id in relative_player_ids
    )
    feature_values.extend([0.0] * (MAX_PLAYERS - player_count))

    # Encode the acting player's hand and played ranks [13] + [13]
    tricks = [*view.trick_history, view.current_trick]
    pass_map = find_pass_map(tricks)
    played_card_map = find_played_card_map(tricks)
    encoded_pass_map = encode_pass_map(pass_map)
    max_cards_per_player = FULL_DECK_SIZE // player_count
    active_players = set(view.turn_order)

    feature_values.extend(encode_hand(list(view.hand)))
    feature_values.extend(encode_hand(played_card_map[view.player_id]))

    # Encode three stable, original-relative opponent slots.
    # [exists: 1] + [active: 1] + [card count: 1]
    # + [played ranks: 13] + [pass counts: 217] = 233 per slot
    opponent_ids = relative_player_ids[1:]
    for player_id in opponent_ids:
        card_count = view.cards_remaining[player_id] / max_cards_per_player
        if not 0.0 <= card_count <= 1.0:
            raise ValueError(
                f"Player {player_id} has an invalid normalized card count "
                f"of {card_count}"
            )

        feature_values.append(1.0)
        feature_values.append(float(player_id in active_players))
        feature_values.append(card_count)
        feature_values.extend(encode_hand(played_card_map[player_id]))
        feature_values.extend(encoded_pass_map[player_id])

    # Missing opponent slots have exists=0, distinguishing them from players
    # who existed but finished and therefore have exists=1 and active=0.
    for _ in range(MAX_PLAYERS - player_count):
        feature_values.extend([0.0] * OPPONENT_SLOT_SIZE)

    if len(feature_values) != STATE_SIZE:
        raise ValueError(
            f"Expected {STATE_SIZE} features, got {len(feature_values)} instead"
        )

    features = torch.tensor(feature_values, dtype=torch.float32)

    # Legal move tensor [217]
    legal_move_values = [False] * MOVE_COUNT
    if view.current_move is not None:
        legal_move_values[encode_move(None)] = 1
    for move in view.possible_moves:
        legal_move_values[encode_move(move)] = True

    legal_moves = torch.tensor(legal_move_values, dtype=torch.bool)

    return features, legal_moves

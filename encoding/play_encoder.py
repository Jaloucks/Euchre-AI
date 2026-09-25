import numpy as np
from engine.deck import Card
from engine.game import PlayerView
from encoding.canonical import card_to_index, canonical_suit_order, relative_seat, index_to_card


# Ordered (name, size). The order here IS the layout of the feature vector.
PLAY_BLOCKS: list[tuple[str, int]] = [
    ('hand', 24),
    ('trick_left', 24),
    ('trick_partner', 24),
    ('trick_right', 24),
    ('played_me', 24),
    ('played_left', 24),
    ('played_partner', 24),
    ('played_right', 24),
    ('void_left', 4),
    ('void_partner', 4),
    ('void_right', 4),
    ('led_suit', 4),
    ('trick_position', 4),
    ('caller', 4),
    ('went_alone', 1),
    ('dealer', 4),
    ('active', 4),
    ('up_card', 24),
    ('tricks_mine', 6),      # 0..5 tricks = 6 possible values
    ('tricks_theirs', 6),
]

# Derived from PLAY_BLOCKS: block name -> slice of the feature vector.
PLAY_OFFSETS: dict[str, slice] = {}
_start = 0
for _name, _size in PLAY_BLOCKS:
    PLAY_OFFSETS[_name] = slice(_start, _start + _size)
    _start += _size
PLAY_DIM: int = _start   # 261

# Relative seat -> the name used in block keys.
SEAT_NAME = {0: 'me', 1: 'left', 2: 'partner', 3: 'right'}


def encode_play_view(view: PlayerView) -> tuple[np.ndarray, np.ndarray]:
    # (features float32 of shape (PLAY_DIM,), all 0/1;  mask bool of shape (24,))
    encoded_play = np.zeros(PLAY_DIM, dtype=np.float32)
    trump = view.trump
    suit_order = canonical_suit_order(trump)

    # My hand
    hand_block = encoded_play[PLAY_OFFSETS['hand']]
    for card in view.hand:
        hand_block[card_to_index(card, trump)] = 1.0

    # Current trick: route each (player, card) to that seat's block
    for player, card in view.cards_played_this_trick:
        rel = relative_seat(view.player_id, player)
        block = encoded_play[PLAY_OFFSETS['trick_' + SEAT_NAME[rel]]]
        block[card_to_index(card, trump)] = 1.0

    # Completed tricks: same routing, one loop deeper
    for trick in view.completed_tricks:
        for player, card in trick:
            rel = relative_seat(view.player_id, player)
            block = encoded_play[PLAY_OFFSETS['played_' + SEAT_NAME[rel]]]
            block[card_to_index(card, trump)] = 1.0

    # Voids: the engine tracks them; we translate real suits -> canonical slots
    for player, suits in view.void_suits.items():
        rel = relative_seat(view.player_id, player)
        if rel == 0:
            continue  # no block for my own voids (my hand already shows them)
        block = encoded_play[PLAY_OFFSETS['void_' + SEAT_NAME[rel]]]
        for suit in suits:
            block[suit_order.index(suit)] = 1.0

    led_suit_block = encoded_play[PLAY_OFFSETS['led_suit']]
    if view.led_suit is not None:
        led_suit_block[suit_order.index(view.led_suit)] = 1.0

    trick_position_block = encoded_play[PLAY_OFFSETS['trick_position']]
    trick_position_block[len(view.cards_played_this_trick)] = 1

    caller_block = encoded_play[PLAY_OFFSETS['caller']]
    caller_block[relative_seat(view.player_id, view.caller)] = 1

    went_alone_block = encoded_play[PLAY_OFFSETS['went_alone']]
    went_alone_block[0] = 1 if view.went_alone else 0

    dealer_block = encoded_play[PLAY_OFFSETS['dealer']]
    dealer_block[relative_seat(view.player_id, view.dealer)] = 1

    active_block = encoded_play[PLAY_OFFSETS['active']]
    for player in view.active_players:
        active_block[relative_seat(view.player_id, player)] = 1

    up_card_block = encoded_play[PLAY_OFFSETS['up_card']]
    up_card_block[card_to_index(view.up_card, trump)] = 1

    tricks_mine_block = encoded_play[PLAY_OFFSETS['tricks_mine']]
    mine = view.tricks_won[view.player_id] + view.tricks_won[(view.player_id + 2) % 4]
    tricks_mine_block[mine] = 1

    tricks_theirs_block = encoded_play[PLAY_OFFSETS['tricks_theirs']]
    theirs = view.tricks_won[(view.player_id + 1) % 4] + view.tricks_won[(view.player_id + 3) % 4]
    tricks_theirs_block[theirs] = 1

    # Legal-move mask
    mask = np.zeros(24, dtype=np.bool_)
    for card in view.legal_cards:
        mask[card_to_index(card, trump)] = True

    return encoded_play, mask


def decode_play_action(index: int, view: PlayerView) -> Card:
    # Inverse of the card-to-index part of encode_play_view. Raises ValueError if
    # the index is out of range or the card is not legal.
    if not 0 <= index < 24:
        raise ValueError(f"Index out of range 0..23: {index}")
    trump = view.trump
    card = index_to_card(index, trump)
    return card


def describe_play_features(x: np.ndarray, trump: str) -> dict[str, np.ndarray]:
    # For debugging: return a dict of block name -> the slice of x for that block.
    # The returned slices are views into x, not copies.
    if x.shape != (PLAY_DIM,):
        raise ValueError(f"Expected shape ({PLAY_DIM},), got {x.shape}")
    suit_order = canonical_suit_order(trump)
    result = {}
    for name, _ in PLAY_BLOCKS:
        block = x[PLAY_OFFSETS[name]]
        if name in ('led_suit', 'dealer', 'caller', 'trick_position'):
            # Convert one-hot to int
            result[name] = np.argmax(block)
        elif name.startswith('void_'):
            # Convert 4-element one-hot to list of suits
            result[name] = [suit_order[i] for i in range(4) if block[i]]
        else:
            result[name] = block.copy()  # copy so caller can mutate safely
    return result

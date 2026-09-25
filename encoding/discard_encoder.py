"""
Discard encoder (Phase 1d).

Turns a DiscardView into (features, mask) for the discard decision.
Frame: canonical relative to view.trump. Seats relative to the dealer (view.player_id).
"""
import numpy as np

from engine.deck import Card
from engine.bidding import DiscardView
from encoding.canonical import card_to_index, index_to_card, relative_seat


DISCARD_BLOCKS: list[tuple[str, int]] = [
    ('hand', 24),        # the six cards (multi-hot)
    ('up_card', 24),     # which of them was picked up (one-hot)
    ('caller', 4),       # who ordered it up, relative to me (the dealer)
    ('went_alone', 1),
]

DISCARD_OFFSETS: dict[str, slice] = {}
_start = 0
for _name, _size in DISCARD_BLOCKS:
    DISCARD_OFFSETS[_name] = slice(_start, _start + _size)
    _start += _size
DISCARD_DIM: int = _start   # 53


def encode_discard_view(view: DiscardView) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a DiscardView into (features float32 of shape (DISCARD_DIM,), all 0/1; mask bool of shape (24,)).
    The mask is True for the 6 cards in view.hand, False for the other 18.
    """
    encoded_discard = np.zeros(DISCARD_DIM, dtype=np.float32)
    mask = np.zeros(24, dtype=np.bool_)

    # Hand
    hand_block = encoded_discard[DISCARD_OFFSETS['hand']]
    for card in view.hand:
        hand_block[card_to_index(card, view.trump)] = 1.0
        mask[card_to_index(card, view.trump)] = True

    # Up card
    up_card_block = encoded_discard[DISCARD_OFFSETS['up_card']]
    up_card_block[card_to_index(view.up_card, view.trump)] = 1.0

    # Caller
    caller_block = encoded_discard[DISCARD_OFFSETS['caller']]
    caller_block[relative_seat(view.player_id, view.caller)] = 1.0  # player_id IS the dealer

    # Went alone
    went_alone_block = encoded_discard[DISCARD_OFFSETS['went_alone']]
    went_alone_block[0] = 1.0 if view.went_alone else 0.0

    return encoded_discard, mask


def decode_discard_action(index: int, view: DiscardView) -> Card:
    """
    Given a discard action index (0..23) and the DiscardView, return the Card that was discarded.
    Raises ValueError if index is not in 0..23 or if the card at that index is not in view.hand.
    """
    if not (0 <= index < 24):
        raise ValueError(f"Index out of range 0..23: {index}")
    trump = view.trump
    card = index_to_card(index, trump)
    if card not in view.hand:
        raise ValueError(f"Discarded card {card} not in hand {view.hand}")
    return card


def describe_discard_features(x: np.ndarray, trump: str | None = None) -> dict:
    """
    For debugging: return a dict of block name -> the slice of x for that block.
    The returned slices are views into x, not copies.
    If trump is provided, the hand and up_card blocks are decoded into lists of Card objects.
    """
    if x.shape != (DISCARD_DIM,):
        raise ValueError(f"Expected shape ({DISCARD_DIM},), got {x.shape}")
    result = {}
    for name, offset in DISCARD_OFFSETS.items():
        block = x[offset]
        if trump is None:
            result[name] = block
        elif name == 'hand' or name == 'up_card':
            cards = [index_to_card(i, trump) for i in range(24) if block[i] > 0.5]
            result[name] = cards
        elif name == 'caller':
            result[name] = int(np.argmax(block))
        elif name == 'went_alone':
            result[name] = bool(block[0])
    return result

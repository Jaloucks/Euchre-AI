import numpy as np
from engine.deck import SAME_COLOR
from engine.bidding import BidView, BidAction
from encoding.canonical import card_to_index, canonical_suit_order, index_to_card, relative_seat

BID_BLOCKS: list[tuple[str, int]] = [
    ('hand', 24),
    ('up_card', 24),
    ('round', 2),
    ('dealer', 4)
]

BID_OFFSETS: dict[str, slice] = {}
_start = 0
for _name, _size in BID_BLOCKS:
    BID_OFFSETS[_name] = slice(_start, _start + _size)
    _start += _size
BID_DIM: int = _start

NUM_BID_ACTIONS: int = 9 
# 0 - pass, 
# 1 - order up, 
# 2 - order up alone, 
# 3 - call same color, 
# 4 - call same color alone, 
# 5 - call cross1, 
# 6 - call cross1 alone, 
# 7 - call cross2, 
# 8 - call cross2 alone

def bid_action_to_index(action: BidAction, up_suit: str) -> int:
    suit_order = canonical_suit_order(up_suit)
    if action.kind == "pass":
        return 0
    elif action.kind == "order_up" and action.suit == up_suit and not action.alone:
        return 1
    elif action.kind == "order_up" and action.suit == up_suit and action.alone:
        return 2
    elif action.kind == "call_suit" and action.suit == SAME_COLOR[up_suit] and not action.alone:
        return 3
    elif action.kind == "call_suit" and action.suit == SAME_COLOR[up_suit] and action.alone:
        return 4
    elif action.kind == "call_suit" and action.suit == suit_order[2] and not action.alone:
        return 5
    elif action.kind == "call_suit" and action.suit == suit_order[2] and action.alone:
        return 6
    elif action.kind == "call_suit" and action.suit == suit_order[3] and not action.alone:
        return 7
    elif action.kind == "call_suit" and action.suit == suit_order[3] and action.alone:
        return 8
    else:
        raise ValueError(f"Invalid bid action: {action} for up_suit={up_suit}")

def index_to_bid_action(index: int, up_suit: str) -> BidAction:
    suit_order = canonical_suit_order(up_suit)
    if index == 0:
        return BidAction(kind="pass")
    elif index == 1:
        return BidAction(kind="order_up", suit=up_suit, alone=False)
    elif index == 2:
        return BidAction(kind="order_up", suit=up_suit, alone=True)
    elif index == 3:
        return BidAction(kind="call_suit", suit=SAME_COLOR[up_suit], alone=False)
    elif index == 4:
        return BidAction(kind="call_suit", suit=SAME_COLOR[up_suit], alone=True)
    elif index == 5:
        return BidAction(kind="call_suit", suit=suit_order[2], alone=False)
    elif index == 6:
        return BidAction(kind="call_suit", suit=suit_order[2], alone=True)
    elif index == 7:
        return BidAction(kind="call_suit", suit=suit_order[3], alone=False)
    elif index == 8:
        return BidAction(kind="call_suit", suit=suit_order[3], alone=True)
    else:
        raise ValueError(f"Invalid bid action index: {index}")

def encode_bid_view(view: BidView, legal_actions: list[BidAction]) -> tuple[np.ndarray, np.ndarray]:
    # features float32 of shape (BID_DIM,), all 0/1
    encoded_bid = np.zeros(BID_DIM, dtype=np.float32)
    trump = view.up_card.suit

    # My hand
    hand_block = encoded_bid[BID_OFFSETS['hand']]
    for card in view.hand:
        hand_block[card_to_index(card, trump)] = 1.0

    # Up-card
    up_card_block = encoded_bid[BID_OFFSETS['up_card']]
    up_card_block[card_to_index(view.up_card, trump)] = 1.0

    # Round
    round_block = encoded_bid[BID_OFFSETS['round']]
    round_block[view.round - 1] = 1.0

    # Dealer
    dealer_block = encoded_bid[BID_OFFSETS['dealer']]
    dealer_block[relative_seat(view.player_id, view.dealer)] = 1.0

    # Mask
    mask = np.zeros(NUM_BID_ACTIONS, dtype=np.bool_)
    for action in legal_actions:
        mask[bid_action_to_index(action, trump)] = True

    return encoded_bid, mask

def describe_bid_features(x: np.ndarray, trump: str | None = None) -> dict[str, np.ndarray]:
    # Returns a dict of the same shape as BID_OFFSETS, with each block decoded into a human-readable form.
    features = {}
    if x.shape != (BID_DIM,):
        raise ValueError(f"Expected shape ({BID_DIM},), got {x.shape}")
    for name, offset in BID_OFFSETS.items():
        block = x[offset]
        if trump is None:
            features[name] = block
        elif name == 'hand' or name == 'up_card':
            cards = [index_to_card(i, trump) for i in range(24) if block[i] > 0.5]
            features[name] = cards
        elif name == 'round':
            features[name] = int(np.argmax(block)) + 1
        elif name == 'dealer':
            features[name] = int(np.argmax(block))
    return features

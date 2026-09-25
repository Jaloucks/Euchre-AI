from engine.deck import Card, SUITS, SAME_COLOR

NUM_CARDS = 24

# Power order inside each canonical block (strongest first).
_TRUMP_RANKS = ["Ace", "King", "Queen", "10", "9"]           # after the two bowers
_NEXT_RANKS = ["Ace", "King", "Queen", "10", "9"]            # Jack is the left bower
_CROSS_RANKS = ["Ace", "King", "Queen", "Jack", "10", "9"]


def relative_seat(me: int, other: int) -> int:
    # 0=me, 1=left opp, 2=partner, 3=right opp
    return (other - me) % 4


def canonical_suit_order(trump: str) -> list[str]:
    # [trump, next (same color), cross_1, cross_2]; cross suits keep SUITS order
    if trump not in SUITS:
        raise ValueError(f"Invalid trump suit: {trump}")
    next_suit = SAME_COLOR[trump]
    cross = [s for s in SUITS if s not in (trump, next_suit)]
    return [trump, next_suit] + cross


def _build_layout(trump: str) -> list[Card]:
    # The single definition of the 24-card canonical layout.
    _, next_suit, cross_1, cross_2 = canonical_suit_order(trump)
    layout = [Card(trump, "Jack"), Card(next_suit, "Jack")]            # 0, 1: bowers
    layout += [Card(trump, r) for r in _TRUMP_RANKS]                   # 2-6
    layout += [Card(next_suit, r) for r in _NEXT_RANKS]                # 7-11
    layout += [Card(cross_1, r) for r in _CROSS_RANKS]                 # 12-17
    layout += [Card(cross_2, r) for r in _CROSS_RANKS]                 # 18-23
    assert len(layout) == NUM_CARDS
    return layout


# Precomputed once at import: index -> card, and card -> index, per trump.
_INDEX_TO_CARD = {t: _build_layout(t) for t in SUITS}
_CARD_TO_INDEX = {t: {c: i for i, c in enumerate(cards)} for t, cards in _INDEX_TO_CARD.items()}


def card_to_index(card: Card, trump: str) -> int:
    # 0..23, per the canonical layout
    if trump not in _CARD_TO_INDEX:
        raise ValueError(f"Invalid trump suit: {trump}")
    try:
        return _CARD_TO_INDEX[trump][card]
    except KeyError:
        raise ValueError(f"Not a Euchre card: {card}") from None


def index_to_card(index: int, trump: str) -> Card:
    # Inverse of card_to_index
    if trump not in _INDEX_TO_CARD:
        raise ValueError(f"Invalid trump suit: {trump}")
    if not 0 <= index < NUM_CARDS:
        raise ValueError(f"Index out of range 0..{NUM_CARDS - 1}: {index}")
    return _INDEX_TO_CARD[trump][index]

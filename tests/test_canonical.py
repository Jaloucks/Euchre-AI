"""
Tests for encoding/canonical.py (Phase 1a).

Canonical 24-card layout, relative to a reference suit `trump`:

    0-6    trump block      R, L, A, K, Q, 10, 9
    7-11   next suit        A, K, Q, 10, 9          (same color as trump, no Jack)
    12-17  cross suit 1     A, K, Q, J, 10, 9       (= canonical_suit_order(trump)[2])
    18-23  cross suit 2     A, K, Q, J, 10, 9       (= canonical_suit_order(trump)[3])

engine.deck is treated as the source of truth for suits, colors, and card power.
"""
import pytest

from engine.deck import Card, SUITS, SAME_COLOR, create_deck, effective_suit, card_rank
from encoding.canonical import (
    NUM_CARDS,
    relative_seat,
    canonical_suit_order,
    card_to_index,
    index_to_card,
)

ALL_TRUMPS = pytest.mark.parametrize("trump", SUITS)

TRUMP_BLOCK = range(0, 7)
NEXT_BLOCK = range(7, 12)
CROSS1_BLOCK = range(12, 18)
CROSS2_BLOCK = range(18, 24)


# ---------- relative_seat ----------

def test_relative_seat_self_is_zero():
    for p in range(4):
        assert relative_seat(p, p) == 0


def test_relative_seat_partner_is_two():
    # Teams are (0,2) and (1,3)
    assert relative_seat(0, 2) == 2
    assert relative_seat(2, 0) == 2
    assert relative_seat(1, 3) == 2
    assert relative_seat(3, 1) == 2


def test_relative_seat_left_is_next_seat_wrapping():
    assert relative_seat(0, 1) == 1
    assert relative_seat(3, 0) == 1  # wraps around the table


def test_relative_seat_right_is_previous_seat_wrapping():
    assert relative_seat(1, 0) == 3
    assert relative_seat(0, 3) == 3


def test_relative_seat_opponents_are_odd():
    # Opponents are always at relative seats 1 and 3
    for me in range(4):
        for other in range(4):
            same_team = (me % 2) == (other % 2)
            assert (relative_seat(me, other) % 2 == 0) == same_team


# ---------- canonical_suit_order ----------

@ALL_TRUMPS
def test_suit_order_starts_with_trump_then_next(trump):
    order = canonical_suit_order(trump)
    assert order[0] == trump
    assert order[1] == SAME_COLOR[trump]


@ALL_TRUMPS
def test_suit_order_is_a_permutation_of_all_four_suits(trump):
    order = canonical_suit_order(trump)
    assert len(order) == 4
    assert set(order) == set(SUITS)


@ALL_TRUMPS
def test_suit_order_cross_suits_are_opposite_color(trump):
    cross = canonical_suit_order(trump)[2:]
    for suit in cross:
        assert suit not in (trump, SAME_COLOR[trump])


@ALL_TRUMPS
def test_suit_order_is_deterministic(trump):
    assert canonical_suit_order(trump) == canonical_suit_order(trump)


def test_suit_order_rejects_invalid_suit():
    with pytest.raises(ValueError):
        canonical_suit_order("Stars")


# ---------- card_to_index / index_to_card: bijection ----------

def test_num_cards_is_24():
    assert NUM_CARDS == 24


@ALL_TRUMPS
def test_card_to_index_is_a_bijection(trump):
    indices = [card_to_index(c, trump) for c in create_deck()]
    assert all(isinstance(i, int) for i in indices), "every card must map to an int (no None)"
    assert sorted(indices) == list(range(NUM_CARDS))


@ALL_TRUMPS
def test_round_trip_card_to_index_to_card(trump):
    for card in create_deck():
        assert index_to_card(card_to_index(card, trump), trump) == card


@ALL_TRUMPS
def test_round_trip_index_to_card_to_index(trump):
    for i in range(NUM_CARDS):
        assert card_to_index(index_to_card(i, trump), trump) == i


@ALL_TRUMPS
def test_index_to_card_produces_all_24_distinct_cards(trump):
    cards = [index_to_card(i, trump) for i in range(NUM_CARDS)]
    assert set(cards) == set(create_deck())


@ALL_TRUMPS
@pytest.mark.parametrize("bad_index", [-1, 24, 100])
def test_index_to_card_rejects_out_of_range(trump, bad_index):
    with pytest.raises(ValueError):
        index_to_card(bad_index, trump)


# ---------- layout: bowers and blocks ----------

@ALL_TRUMPS
def test_right_bower_is_index_0(trump):
    assert card_to_index(Card(trump, "Jack"), trump) == 0


@ALL_TRUMPS
def test_left_bower_is_index_1(trump):
    assert card_to_index(Card(SAME_COLOR[trump], "Jack"), trump) == 1


@ALL_TRUMPS
def test_trump_block_is_exactly_the_effective_trump_cards(trump):
    for card in create_deck():
        is_trump = effective_suit(card, trump) == trump
        assert (card_to_index(card, trump) in TRUMP_BLOCK) == is_trump, card


@ALL_TRUMPS
def test_next_block_is_same_color_suit_without_jack(trump):
    for i in NEXT_BLOCK:
        card = index_to_card(i, trump)
        assert card.suit == SAME_COLOR[trump]
        assert card.rank != "Jack"


@ALL_TRUMPS
def test_cross_blocks_match_canonical_suit_order(trump):
    order = canonical_suit_order(trump)
    for i in CROSS1_BLOCK:
        assert index_to_card(i, trump).suit == order[2]
    for i in CROSS2_BLOCK:
        assert index_to_card(i, trump).suit == order[3]


@ALL_TRUMPS
@pytest.mark.parametrize("block", [TRUMP_BLOCK, NEXT_BLOCK, CROSS1_BLOCK, CROSS2_BLOCK])
def test_each_block_is_ordered_strongest_first(trump, block):
    # Within a block, lower index = stronger card, using the engine's card_rank.
    # For non-trump blocks, rank the cards as if their own suit was led.
    cards = [index_to_card(i, trump) for i in block]
    led = effective_suit(cards[0], trump)
    powers = [card_rank(c, trump, led) for c in cards]
    assert powers == sorted(powers, reverse=True)
    assert len(set(powers)) == len(powers)


# ---------- concrete spot checks (trump = Spades) ----------

@pytest.mark.parametrize("card, expected", [
    (Card("Spades", "Jack"), 0),    # right bower
    (Card("Clubs", "Jack"), 1),     # left bower
    (Card("Spades", "Ace"), 2),
    (Card("Spades", "King"), 3),
    (Card("Spades", "Queen"), 4),
    (Card("Spades", "10"), 5),
    (Card("Spades", "9"), 6),
    (Card("Clubs", "Ace"), 7),      # next suit
    (Card("Clubs", "9"), 11),
])
def test_spades_trump_spot_checks(card, expected):
    assert card_to_index(card, "Spades") == expected


def test_spades_trump_cross_block_spot_checks():
    order = canonical_suit_order("Spades")
    assert index_to_card(12, "Spades") == Card(order[2], "Ace")
    assert index_to_card(15, "Spades") == Card(order[2], "Jack")
    assert index_to_card(17, "Spades") == Card(order[2], "9")
    assert index_to_card(18, "Spades") == Card(order[3], "Ace")
    assert index_to_card(23, "Spades") == Card(order[3], "9")


# ---------- the point of canonicalization ----------

def test_same_role_same_index_across_trumps():
    # The Ace of trump lands on the same index no matter which suit is trump.
    assert len({card_to_index(Card(t, "Ace"), t) for t in SUITS}) == 1
    # Same for the left bower.
    assert len({card_to_index(Card(SAME_COLOR[t], "Jack"), t) for t in SUITS}) == 1

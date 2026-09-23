"""
Tests for engine/deck.py

Assumes: from engine.deck import Card, create_deck, effective_suit, card_rank
Adjust the import below if your module path differs.
"""
import pytest
from engine.deck import Card, create_deck, effective_suit, card_rank


# ---------- Card ----------

def test_card_equality():
    assert Card("Spades", "9") == Card("Spades", "9")
    assert Card("Spades", "9") != Card("Spades", "10")
    assert Card("Spades", "9") != Card("Hearts", "9")


def test_card_hashable_and_dedupes_in_set():
    a = Card("Spades", "9")
    b = Card("Spades", "9")
    assert len({a, b}) == 1  # same card, should collapse to one entry


# ---------- create_deck ----------

def test_deck_has_24_unique_cards():
    deck = create_deck()
    assert len(deck) == 24
    assert len(set(deck)) == 24  # no duplicates


def test_deck_only_has_nine_through_ace():
    deck = create_deck()
    ranks = {card.rank for card in deck}
    assert ranks == {"9", "10", "Jack", "Queen", "King", "Ace"}


# ---------- effective_suit ----------

def test_right_bower_is_trump_suit():
    # Jack of trump suit itself
    card = Card("Spades", "Jack")
    assert effective_suit(card, trump="Spades") == "Spades"


def test_left_bower_counts_as_trump():
    # Jack of Clubs, same color as Spades -> left bower when trump is Spades
    card = Card("Clubs", "Jack")
    assert effective_suit(card, trump="Spades") == "Spades"


def test_non_bower_jack_keeps_its_own_suit():
    # Jack of Hearts is NOT a bower when trump is Spades (different color family)
    card = Card("Hearts", "Jack")
    assert effective_suit(card, trump="Spades") == "Hearts"


def test_plain_card_keeps_its_own_suit():
    card = Card("Diamonds", "9")
    assert effective_suit(card, trump="Spades") == "Diamonds"


@pytest.mark.parametrize("trump", ["Hearts", "Diamonds", "Clubs", "Spades"])
def test_left_bower_resolves_correctly_for_every_trump(trump):
    # For each possible trump, the same-color Jack should become that trump.
    same_color = {
        "Hearts": "Diamonds", "Diamonds": "Hearts",
        "Clubs": "Spades", "Spades": "Clubs",
    }
    left_bower_card = Card(same_color[trump], "Jack")
    assert effective_suit(left_bower_card, trump=trump) == trump


# ---------- card_rank ----------

def test_right_bower_outranks_left_bower():
    right = card_rank(Card("Spades", "Jack"), trump="Spades", led_suit="Spades")
    left = card_rank(Card("Clubs", "Jack"), trump="Spades", led_suit="Spades")
    assert right > left


def test_left_bower_outranks_other_trump():
    left = card_rank(Card("Clubs", "Jack"), trump="Spades", led_suit="Spades")
    other_trump = card_rank(Card("Spades", "Ace"), trump="Spades", led_suit="Spades")
    assert left > other_trump


def test_any_trump_outranks_any_led_suit_card():
    # Lowest trump (9) should still beat the highest led-suit card (Ace)
    low_trump = card_rank(Card("Spades", "9"), trump="Spades", led_suit="Hearts")
    high_led = card_rank(Card("Hearts", "Ace"), trump="Spades", led_suit="Hearts")
    assert low_trump > high_led


def test_led_suit_outranks_off_suit():
    led = card_rank(Card("Hearts", "9"), trump="Spades", led_suit="Hearts")
    off = card_rank(Card("Diamonds", "Ace"), trump="Spades", led_suit="Hearts")
    assert led > off


def test_off_suit_jack_is_not_treated_as_bower():
    # Jack of Hearts, trump is Spades, Hearts led -> should rank as a normal
    # led-suit card, NOT as a bower.
    off_suit_jack = card_rank(Card("Hearts", "Jack"), trump="Spades", led_suit="Hearts")
    led_suit_ace = card_rank(Card("Hearts", "Ace"), trump="Spades", led_suit="Hearts")
    assert off_suit_jack < led_suit_ace  # Jack < Ace within the same led suit

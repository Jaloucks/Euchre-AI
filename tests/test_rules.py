"""
Tests for engine/rules.py

Assumes: from engine.rules import trick_winner, legal_plays
Adjust the import below if your module path differs.
"""
from engine.deck import Card
from engine.rules import trick_winner, legal_plays


# ---------- trick_winner ----------

def test_trick_winner_simple_led_suit_wins():
    cards_played = [
        (0, Card("Hearts", "9")),
        (1, Card("Hearts", "Ace")),
        (2, Card("Diamonds", "King")),
        (3, Card("Hearts", "10")),
    ]
    assert trick_winner(cards_played, trump="Spades") == 1  # Ace of Hearts


def test_trick_winner_trump_beats_led_suit():
    cards_played = [
        (0, Card("Hearts", "Ace")),   # led suit, high card
        (1, Card("Spades", "9")),     # trump, low card, should still win
        (2, Card("Hearts", "King")),
        (3, Card("Hearts", "Queen")),
    ]
    assert trick_winner(cards_played, trump="Spades") == 1


def test_trick_winner_right_bower_wins():
    cards_played = [
        (0, Card("Spades", "Ace")),
        (1, Card("Spades", "Jack")),  # right bower
        (2, Card("Spades", "King")),
        (3, Card("Clubs", "Jack")),   # left bower, still loses to right bower
    ]
    assert trick_winner(cards_played, trump="Spades") == 1


def test_trick_winner_left_bower_led_sets_led_suit_to_trump():
    # Left bower (Jack of Clubs, trump Spades) is led. Its EFFECTIVE suit is
    # Spades, not Clubs -- so a Hearts card should NOT be treated as having
    # followed the led suit, and any actual Spades card should compete with it.
    cards_played = [
        (0, Card("Clubs", "Jack")),   # left bower leads -> led suit is Spades
        (1, Card("Spades", "King")),  # real trump card
        (2, Card("Hearts", "Ace")),   # off suit, irrelevant
        (3, Card("Diamonds", "9")),   # off suit, irrelevant
    ]
    # Left bower still outranks the King of Spades (left bower is 2nd highest
    # trump, above all non-bower trump cards).
    assert trick_winner(cards_played, trump="Spades") == 0


def test_trick_winner_off_suit_cards_never_win():
    cards_played = [
        (0, Card("Hearts", "9")),      # led suit, low
        (1, Card("Diamonds", "Ace")),  # off suit, high rank, can't win
        (2, Card("Clubs", "Ace")),     # off suit, high rank, can't win
        (3, Card("Hearts", "10")),     # led suit, beats the 9
    ]
    assert trick_winner(cards_played, trump="Spades") == 3


# ---------- legal_plays ----------

def test_legal_plays_leading_allows_anything():
    hand = [Card("Hearts", "9"), Card("Spades", "King"), Card("Clubs", "Ace")]
    assert legal_plays(hand, cards_played=[], trump="Diamonds") == hand


def test_legal_plays_must_follow_suit_when_able():
    hand = [Card("Hearts", "9"), Card("Hearts", "King"), Card("Spades", "Ace")]
    cards_played = [(0, Card("Hearts", "10"))]  # Hearts led
    result = legal_plays(hand, cards_played, trump="Diamonds")
    assert set(result) == {Card("Hearts", "9"), Card("Hearts", "King")}


def test_legal_plays_void_in_led_suit_allows_anything():
    hand = [Card("Spades", "9"), Card("Clubs", "King")]
    cards_played = [(0, Card("Hearts", "10"))]  # Hearts led, hand has no Hearts
    result = legal_plays(hand, cards_played, trump="Diamonds")
    assert set(result) == set(hand)


def test_legal_plays_left_bower_counts_as_trump_for_following():
    # Trump is Spades, Spades is led. Hand holds the left bower (Jack of
    # Clubs) but no printed Spades. The left bower counts as trump, so the
    # player IS considered to have the led suit and must play it.
    hand = [Card("Clubs", "Jack"), Card("Hearts", "King"), Card("Diamonds", "9")]
    cards_played = [(0, Card("Spades", "9"))]  # Spades (trump) led
    result = legal_plays(hand, cards_played, trump="Spades")
    assert result == [Card("Clubs", "Jack")]


def test_legal_plays_left_bower_not_required_when_off_suit_led():
    # Hand has NO Hearts card at all (not even effectively) -- the left
    # bower (Clubs Jack, trump Spades) becomes Spades, not Hearts, and the
    # Diamonds 9 stays Diamonds. Player is genuinely void in Hearts, so
    # everything is legal.
    hand = [Card("Clubs", "Jack"), Card("Diamonds", "9"), Card("Diamonds", "King")]
    cards_played = [(0, Card("Hearts", "9"))]  # Hearts led
    result = legal_plays(hand, cards_played, trump="Spades")
    assert set(result) == set(hand)

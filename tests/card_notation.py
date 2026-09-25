"""
Short card notation for tests only: "JH" = Jack of Hearts, "10S" = 10 of Spades.

    c("AD")                -> Card("Diamonds", "Ace")
    cards("JH JD 9C 10S")  -> [Card(...), ...]
"""
from engine.deck import Card

_RANKS = {"9": "9", "10": "10", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
_SUITS = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}


def c(code: str) -> Card:
    return Card(_SUITS[code[-1]], _RANKS[code[:-1]])


def cards(codes: str) -> list[Card]:
    return [c(x) for x in codes.split()]

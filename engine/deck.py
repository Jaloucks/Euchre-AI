import random


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __hash__(self):
        return hash((self.suit, self.rank))

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(suit={self.suit}, rank={self.rank})"

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.suit == other.suit and self.rank == other.rank
        return False


SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['9', '10', 'Jack', 'Queen', 'King', 'Ace']


def create_deck() -> list[Card]:
    return [Card(suit, rank) for suit in SUITS for rank in RANKS]


SAME_COLOR = {
    'Hearts': 'Diamonds',
    'Diamonds': 'Hearts',
    'Clubs': 'Spades',
    'Spades': 'Clubs',
}


def effective_suit(card: Card, trump: str) -> str:
    if card.suit == trump:
        return trump
    if card.rank == 'Jack' and SAME_COLOR[card.suit] == trump:
        return trump
    return card.suit


RANK_ORDER = {'9': 0, '10': 1, 'Jack': 2, 'Queen': 3, 'King': 4, 'Ace': 5}


def card_rank(card: Card, trump: str, led_suit: str) -> int:
    # NOTE: branch order matters. The right-bower check must come before the
    # generic trump checks, or it will be swallowed by them.
    if card.suit == trump and card.rank == 'Jack':
        return 107
    elif effective_suit(card, trump) == trump and card.rank == 'Jack':
        return 106
    elif effective_suit(card, trump) == trump:
        return 100 + RANK_ORDER[card.rank]
    elif card.suit == led_suit:
        return RANK_ORDER[card.rank]
    else:
        return -1


def deal_hands(
    deck: list[Card],
    rng: random.Random,
) -> tuple[list[list[Card]], Card, list[Card]]:
    """
    Shuffle a copy of `deck` and deal. Does NOT mutate the caller's deck.
    Returns (hands, up_card, kitty) where hands[i] is player i's 5 cards.
    """
    deck = deck.copy()
    rng.shuffle(deck)
    hands = [deck[i:i + 5] for i in range(0, 20, 5)]
    up_card = deck[20]
    kitty = deck[21:24]
    return hands, up_card, kitty
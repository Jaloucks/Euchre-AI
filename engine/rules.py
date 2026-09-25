from engine.deck import Card, effective_suit, card_rank


def trick_winner(cards_played: list[tuple[int, Card]], trump: str) -> int:
    """
    Determine which player won a trick. `cards_played` must be non-empty and
    in play order — the first entry establishes the led suit.
    """
    if not cards_played:
        raise ValueError("trick_winner called with no cards played")

    led_suit = effective_suit(cards_played[0][1], trump)
    winning_card = max(cards_played, key=lambda x: card_rank(x[1], trump, led_suit))
    return winning_card[0]


def legal_plays(
    hand: list[Card],
    cards_played: list[tuple[int, Card]],
    trump: str,
) -> list[Card]:
    """
    Which cards in `hand` may legally be played. Always returns a NEW list,
    never the caller's `hand` object — so a caller can mutate the hand
    without silently mutating the legal-play list it's holding.
    """
    if not cards_played:
        return list(hand)

    led_suit = effective_suit(cards_played[0][1], trump)
    has_led_suit = any(effective_suit(card, trump) == led_suit for card in hand)

    if has_led_suit:
        return [card for card in hand if effective_suit(card, trump) == led_suit]
    return list(hand)
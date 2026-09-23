from engine.deck import Card, effective_suit, card_rank

def trick_winner(cards_played: list[tuple[int, Card]], trump: str):
    led_suit = effective_suit(cards_played[0][1], trump)
    winning_card = max(cards_played, key=lambda x: card_rank(x[1], trump, led_suit))
    return winning_card[0]

def legal_plays(hand: list[Card], cards_played: list[tuple[int, Card]], trump: str) -> list[Card]:
    if not cards_played:
        return hand

    led_suit = effective_suit(cards_played[0][1], trump)
    has_led_suit = any(effective_suit(card, trump) == led_suit for card in hand)

    if has_led_suit:
        return [card for card in hand if effective_suit(card, trump) == led_suit]
    else:
        return hand
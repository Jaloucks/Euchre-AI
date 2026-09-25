from encoding.canonical import card_to_index, relative_seat
from engine.bidding import BidAction, BidView, DiscardView
from engine.deck import Card, SAME_COLOR, SUITS, card_rank, effective_suit
from engine.game import PlayerView
from engine.rules import trick_winner

RIGHT_POINTS = 3.0
LEFT_POINTS = 2.5
TRUMP_RANK_POINTS = {"Ace": 2.0, "King": 1.5, "Queen": 1.0, "10": 1.0, "9": 1.0}
OFF_ACE_POINTS = 1.0
UP_CARD_WEIGHT = 0.5
CALL_THRESHOLD = 5.0
ALONE_THRESHOLD = 8.5

def card_points(card: Card, trump: str) -> float:
    if card.rank == "Jack" and card.suit == trump:
        return RIGHT_POINTS
    elif card.rank == "Jack" and card.suit == SAME_COLOR[trump]:
        return LEFT_POINTS
    elif card.suit == trump:
        return TRUMP_RANK_POINTS[card.rank]
    elif card.rank == "Ace":
        return OFF_ACE_POINTS
    else:
        return 0.0

def hand_strength(hand: list[Card], trump: str) -> float:
    points = sum(card_points(card, trump) for card in hand)
    return points

def strength_key(card: Card, trump: str) -> tuple[int, int]:
    return (card_rank(card, trump, effective_suit(card, trump)), -card_to_index(card, trump))

def lowest_card(hand: list[Card], trump: str) -> Card:
    return min(hand, key=lambda card: strength_key(card, trump))

def highest_card(hand: list[Card], trump: str) -> Card:
    return max(hand, key=lambda card: strength_key(card, trump))

def pick_discard(hand: list[Card], trump: str) -> Card:
    off_suit = [card for card in hand if effective_suit(card, trump) != trump]
    if not off_suit:
        return lowest_card(hand, trump)
    non_aces = [card for card in off_suit if card.rank != "Ace"]
    if not non_aces:
        return lowest_card(off_suit, trump)
    singletons = []
    for card in non_aces:
        suit_count = 0
        for other in hand:
            if effective_suit(other, trump) == effective_suit(card, trump):
                suit_count += 1
        if suit_count == 1:
            singletons.append(card)
    if singletons:
        return lowest_card(singletons, trump)
    else:
        return lowest_card(non_aces, trump)

def round1_strength(view: BidView) -> float:
    trump = view.up_card.suit
    if view.dealer == view.player_id:
        dealer_hand = view.hand + [view.up_card]
        dealer_hand.remove(pick_discard(dealer_hand, trump))
        return hand_strength(dealer_hand, trump)
    elif view.dealer == (view.player_id + 2) % 4:
        return hand_strength(view.hand, trump) + UP_CARD_WEIGHT * card_points(view.up_card, trump)
    else:
        return hand_strength(view.hand, trump) - UP_CARD_WEIGHT * card_points(view.up_card, trump)

class HeuristicBot:
    def __init__(self, call_threshold=CALL_THRESHOLD, alone_threshold=ALONE_THRESHOLD):
        self.call_threshold = call_threshold
        self.alone_threshold = alone_threshold

    def choose_bid(self, view: BidView, legal_actions: list[BidAction]) -> BidAction:
        if view.round == 1:
            strength = round1_strength(view)
            if strength >= self.alone_threshold:
                bid_action = BidAction(kind="order_up", suit=view.up_card.suit, alone=True)
            elif strength >= self.call_threshold:
                bid_action = BidAction(kind="order_up", suit=view.up_card.suit, alone=False)
            else:
                bid_action = BidAction(kind="pass")
        else:
            possible_suits = set()
            for action in legal_actions:
                if action.kind == "call_suit":
                    possible_suits.add(action.suit)
            best_suit = None
            best_strength = -1.0
            for suit in SUITS:
                if suit not in possible_suits:
                    continue
                strength = hand_strength(view.hand, suit)
                if strength > best_strength:
                    best_strength = strength
                    best_suit = suit

            if best_strength >= self.alone_threshold:
                bid_action = BidAction(kind="call_suit", suit=best_suit, alone=True)
            elif best_strength >= self.call_threshold:
                bid_action = BidAction(kind="call_suit", suit=best_suit, alone=False)
            elif BidAction(kind="pass") not in legal_actions:
                bid_action = BidAction(kind="call_suit", suit=best_suit, alone=False)
            else:
                bid_action = BidAction(kind="pass")

        assert bid_action in legal_actions, (
            f"HeuristicBot chose {bid_action}, which was not legal given state {view}"
        )

        return bid_action

    def choose_discard(self, view: DiscardView):
        return pick_discard(view.hand, view.trump)

    def choose_play(self, view: PlayerView):
        if len(view.legal_cards) == 1:
            return view.legal_cards[0]
        elif not view.cards_played_this_trick:
            trump_count = 0
            for card in view.hand:
                if effective_suit(card, view.trump) == view.trump:
                    trump_count += 1
            if trump_count >= 2 and view.caller == view.player_id:
                return highest_card(view.hand, view.trump)
            elif 'Ace' in [card.rank for card in view.hand if effective_suit(card, view.trump) != view.trump]:
                aces = [card for card in view.hand if card.rank == 'Ace' and effective_suit(card, view.trump) != view.trump]
                return highest_card(aces, view.trump)
            else:
                alltrump = True
                for card in view.hand:
                    if effective_suit(card, view.trump) != view.trump:
                        alltrump = False
                        break
                if alltrump:
                    return highest_card(view.hand, view.trump)
                else:
                    return lowest_card(view.hand, view.trump)
        else:
            winner = trick_winner(view.cards_played_this_trick, view.trump)
            if relative_seat(view.player_id, winner) == 2:
                return lowest_card(view.legal_cards, view.trump)
            winning_card = None
            for player_id, card in view.cards_played_this_trick:
                if player_id == winner:
                    winning_card = card
            winning_rank = card_rank(winning_card, view.trump, view.led_suit)
            beating_cards = []
            for card in view.legal_cards:
                if card_rank(card, view.trump, view.led_suit) > winning_rank:
                    beating_cards.append(card)
            if beating_cards:
                return lowest_card(beating_cards, view.trump)
            else:
                return lowest_card(view.legal_cards, view.trump)

    def as_callbacks(self):
        return self.choose_bid, self.choose_discard, self.choose_play

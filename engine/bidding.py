from dataclasses import dataclass, field
from engine.deck import Card

SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

@dataclass
class BidAction:
    kind: str          # "pass" | "order_up" | "call_suit"
    suit: str | None = None   # None for pass; suit name otherwise
    alone: bool = False

@dataclass
class BiddingState:
    up_card_suit: str
    dealer: int                    # player id of dealer
    current_player: int
    round: int                     # 1 or 2
    passes_this_round: int = 0     # how many players have passed so far this round

@dataclass
class BidResult:
    winning_player: int
    winning_bid: BidAction

def legal_bid_actions(state: BiddingState) -> list[BidAction]:
    actions = []

    if state.round == 1:
        actions.append(BidAction(kind="pass"))
        actions.append(BidAction(kind="order_up", suit=state.up_card_suit, alone=False))
        actions.append(BidAction(kind="order_up", suit=state.up_card_suit, alone=True))
        return actions

    # round 2
    is_stuck_dealer = (
        state.current_player == state.dealer
        and state.passes_this_round == 3
    )

    if not is_stuck_dealer:
        actions.append(BidAction(kind="pass"))

    for suit in SUITS:
        if suit == state.up_card_suit:
            continue
        actions.append(BidAction(kind="call_suit", suit=suit, alone=False))
        actions.append(BidAction(kind="call_suit", suit=suit, alone=True))

    return actions

def run_bidding(dealer: int, up_card_suit: str, choose_action_fn) -> BidResult:
    state = BiddingState(
        up_card_suit=up_card_suit,
        dealer=dealer,
        current_player=(dealer + 1) % 4,
        round=1,
    )

    while True:
        legal_actions = legal_bid_actions(state)
        action = choose_action_fn(state, legal_actions)

        if action.kind == "pass":
            state.passes_this_round += 1

            if state.round == 1 and state.passes_this_round == 4:
                state.round = 2
                state.passes_this_round = 0
                state.current_player = (state.dealer + 1) % 4
                continue

            if state.round == 2 and state.passes_this_round == 4:
                raise AssertionError("Stuck-the-dealer should have prevented this")

            state.current_player = (state.current_player + 1) % 4
            continue

        if action.kind == "order_up":
            return BidResult(winning_player=state.current_player, winning_bid=action)

        if action.kind == "call_suit":
            return BidResult(winning_player=state.current_player, winning_bid=action)

def resolve_discard(hand: list[Card], up_card: Card, choose_discard_fn) -> tuple[list[Card], Card]:
    new_hand = hand + [up_card]
    discard = choose_discard_fn(new_hand)
    if discard not in new_hand:
        raise ValueError(f"choose_discard_fn returned a card not in hand: {discard}")
    new_hand.remove(discard)
    return new_hand, discard
    
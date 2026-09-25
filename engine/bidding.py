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

@dataclass
class BidView:
    player_id: int
    hand: list[Card]
    up_card: Card
    dealer: int
    round: int
    passes_this_round: int

@dataclass
class DiscardView:
    """
    What the dealer knows when discarding after being ordered up.
    Built by resolve_discard and passed to choose_discard_fn.
    """
    player_id: int          # always the dealer
    hand: list[Card]        # SIX cards: the dealer's 5 + the up card (give the agent a copy)
    up_card: Card           # which of the six was just picked up
    trump: str              # always up_card.suit here
    caller: int             # who ordered it up
    went_alone: bool

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

def run_bidding(dealer: int, up_card: Card, hands: dict[int, list[Card]], choose_action_fn) -> BidResult:
    state = BiddingState(
        up_card_suit=up_card.suit,
        dealer=dealer,
        current_player=(dealer + 1) % 4,
        round=1,
    )

    while True:
        view = BidView(
            player_id=state.current_player,
            hand=list(hands[state.current_player]),   # copy: agent can't touch engine state
            up_card=up_card,
            dealer=state.dealer,
            round=state.round,
            passes_this_round=state.passes_this_round,
        )

        legal_actions = legal_bid_actions(state)
        action = choose_action_fn(view, legal_actions)

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

def resolve_discard(
    hand: list[Card],
    up_card: Card,
    dealer: int,
    caller: int,
    went_alone: bool,
    choose_discard_fn,
) -> tuple[list[Card], Card]:
    """Dealer picks up the up card, then discards one of the six. Does not mutate `hand`."""
    new_hand = hand + [up_card]
    view = DiscardView(
        player_id=dealer,
        hand=list(new_hand),        # copy: agent can't touch engine state
        up_card=up_card,
        trump=up_card.suit,
        caller=caller,
        went_alone=went_alone,
    )
    discard = choose_discard_fn(view)
    if discard not in new_hand:
        raise ValueError(f"choose_discard_fn returned a card not in hand: {discard}")
    new_hand.remove(discard)
    return new_hand, discard
    
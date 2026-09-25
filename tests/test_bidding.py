"""
Tests for engine/bidding.py

Assumes:
  from engine.bidding import BidAction, BiddingState, BidResult,
                              legal_bid_actions, run_bidding, resolve_discard

run_bidding signature (Phase 1c):
  run_bidding(dealer, up_card: Card, hands: dict[int, list[Card]], choose_action_fn)
The callback receives a BidView (see tests/test_bid_view.py), not the BiddingState.

resolve_discard signature (Phase 1d):
  resolve_discard(hand, up_card, dealer, caller, went_alone, choose_discard_fn)
The callback receives a DiscardView (see tests/test_discard_view.py).
  from engine.deck import Card
Adjust imports below if your module paths differ.
"""
import pytest
from engine.deck import Card
from engine.bidding import (
    BidAction,
    BiddingState,
    legal_bid_actions,
    run_bidding,
    resolve_discard,
)


# ---------- legal_bid_actions: round 1 ----------

def test_round1_actions_are_pass_order_up_or_order_up_alone():
    state = BiddingState(up_card_suit="Hearts", dealer=3, current_player=0, round=1)
    actions = legal_bid_actions(state)
    kinds = {(a.kind, a.suit, a.alone) for a in actions}
    assert kinds == {
        ("pass", None, False),
        ("order_up", "Hearts", False),
        ("order_up", "Hearts", True),
    }


# ---------- legal_bid_actions: round 2 ----------

def test_round2_normal_player_can_pass():
    # Not the dealer, or dealer but not everyone else has passed yet.
    state = BiddingState(
        up_card_suit="Hearts", dealer=3, current_player=0, round=2,
        passes_this_round=0,
    )
    actions = legal_bid_actions(state)
    assert any(a.kind == "pass" for a in actions)


def test_round2_excludes_up_card_suit_from_callable_suits():
    state = BiddingState(
        up_card_suit="Hearts", dealer=3, current_player=0, round=2,
        passes_this_round=0,
    )
    actions = legal_bid_actions(state)
    called_suits = {a.suit for a in actions if a.kind == "call_suit"}
    assert "Hearts" not in called_suits
    assert called_suits == {"Diamonds", "Clubs", "Spades"}


def test_round2_stuck_dealer_cannot_pass():
    state = BiddingState(
        up_card_suit="Hearts", dealer=3, current_player=3, round=2,
        passes_this_round=3,  # everyone else already passed
    )
    actions = legal_bid_actions(state)
    assert all(a.kind != "pass" for a in actions)
    # dealer must still have suit-calling options available
    assert any(a.kind == "call_suit" for a in actions)


def test_round2_dealer_not_stuck_if_not_everyones_turn_passed_yet():
    # It's the dealer's seat, but only 2 players have passed so far -- not
    # actually stuck yet (shouldn't normally happen mid-round, but the
    # function should only trigger stuck-dealer on the exact condition).
    state = BiddingState(
        up_card_suit="Hearts", dealer=3, current_player=3, round=2,
        passes_this_round=2,
    )
    actions = legal_bid_actions(state)
    assert any(a.kind == "pass" for a in actions)


# ---------- run_bidding ----------

UP_CARD = Card("Hearts", "10")
HANDS = {
    0: [Card("Clubs", "9"), Card("Clubs", "10"), Card("Clubs", "Queen"), Card("Clubs", "King"), Card("Clubs", "Ace")],
    1: [Card("Spades", "9"), Card("Spades", "10"), Card("Spades", "Queen"), Card("Spades", "King"), Card("Spades", "Ace")],
    2: [Card("Diamonds", "9"), Card("Diamonds", "10"), Card("Diamonds", "Queen"), Card("Diamonds", "King"), Card("Diamonds", "Ace")],
    3: [Card("Hearts", "9"), Card("Hearts", "Queen"), Card("Hearts", "King"), Card("Hearts", "Ace"), Card("Clubs", "Jack")],
}

class ScriptedChooser:
    """
    Test helper: returns a pre-set sequence of actions, one per call,
    regardless of what state/legal_actions it's given. Raises if called
    more times than scripted -- catches loops that run longer than expected.
    """
    def __init__(self, actions: list[BidAction]):
        self._actions = list(actions)
        self.calls = 0

    def __call__(self, state, legal_actions: list[BidAction]) -> BidAction:
        if self.calls >= len(self._actions):
            raise AssertionError("ScriptedChooser called more times than scripted")
        action = self._actions[self.calls]
        self.calls += 1
        assert action in legal_actions, (
            f"Scripted action {action} was not legal given state {state}"
        )
        return action


def test_run_bidding_round1_order_up_ends_immediately():
    dealer = 3
    chooser = ScriptedChooser([
        BidAction(kind="order_up", suit="Hearts", alone=False),
    ])
    result = run_bidding(dealer=dealer, up_card=UP_CARD, hands=HANDS, choose_action_fn=chooser)
    assert result.winning_player == (dealer + 1) % 4  # first player left of dealer
    assert result.winning_bid.kind == "order_up"
    assert result.winning_bid.suit == "Hearts"


def test_run_bidding_all_pass_round1_transitions_to_round2():
    dealer = 3
    chooser = ScriptedChooser([
        BidAction(kind="pass"),  # player 0
        BidAction(kind="pass"),  # player 1
        BidAction(kind="pass"),  # player 2
        BidAction(kind="pass"),  # player 3 (dealer)
        BidAction(kind="call_suit", suit="Clubs", alone=False),  # player 0, round 2
    ])
    result = run_bidding(dealer=dealer, up_card=UP_CARD, hands=HANDS, choose_action_fn=chooser)
    assert result.winning_player == 0  # left of dealer again, round 2
    assert result.winning_bid.kind == "call_suit"
    assert result.winning_bid.suit == "Clubs"
    assert chooser.calls == 5  # 4 passes + 1 call


def test_run_bidding_stuck_dealer_is_forced_to_call():
    dealer = 3
    chooser = ScriptedChooser([
        BidAction(kind="pass"),  # round 1, player 0
        BidAction(kind="pass"),  # round 1, player 1
        BidAction(kind="pass"),  # round 1, player 2
        BidAction(kind="pass"),  # round 1, player 3 (dealer)
        BidAction(kind="pass"),  # round 2, player 0
        BidAction(kind="pass"),  # round 2, player 1
        BidAction(kind="pass"),  # round 2, player 2
        BidAction(kind="call_suit", suit="Clubs", alone=False),  # round 2, dealer, forced
    ])
    result = run_bidding(dealer=dealer, up_card=UP_CARD, hands=HANDS, choose_action_fn=chooser)
    assert result.winning_player == dealer
    assert result.winning_bid.kind == "call_suit"


# ---------- resolve_discard ----------

def test_resolve_discard_normal_case():
    hand = [Card("Hearts", "9"), Card("Hearts", "King"), Card("Spades", "Ace"),
             Card("Clubs", "9"), Card("Diamonds", "Queen")]
    up_card = Card("Hearts", "10")

    new_hand, discard = resolve_discard(
        hand, up_card, dealer=3, caller=1, went_alone=False,
        choose_discard_fn=lambda view: Card("Clubs", "9")
    )

    assert len(new_hand) == 5
    assert discard == Card("Clubs", "9")
    assert discard not in new_hand
    assert up_card in new_hand


def test_resolve_discard_does_not_mutate_original_hand():
    hand = [Card("Hearts", "9"), Card("Hearts", "King"), Card("Spades", "Ace"),
             Card("Clubs", "9"), Card("Diamonds", "Queen")]
    original_len = len(hand)
    up_card = Card("Hearts", "10")

    resolve_discard(hand, up_card, dealer=3, caller=1, went_alone=False,
                    choose_discard_fn=lambda view: view.hand[0])

    assert len(hand) == original_len  # caller's list untouched
    assert up_card not in hand


def test_resolve_discard_raises_on_invalid_discard():
    hand = [Card("Hearts", "9"), Card("Hearts", "King"), Card("Spades", "Ace"),
             Card("Clubs", "9"), Card("Diamonds", "Queen")]
    up_card = Card("Hearts", "10")

    with pytest.raises(ValueError):
        resolve_discard(
            hand, up_card, dealer=3, caller=1, went_alone=False,
            choose_discard_fn=lambda view: Card("Spades", "King")
        )

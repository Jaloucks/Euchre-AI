"""
Tests for DiscardView (Phase 1d engine change).

When the dealer is ordered up, they pick up the up card and discard one of
the six. DiscardView is what the dealer's agent sees at that moment.

    @dataclass
    class DiscardView:             # in engine/bidding.py
        player_id: int             # always the dealer
        hand: list[Card]           # SIX cards: the dealer's 5 + the up card (a copy)
        up_card: Card              # which of the six was picked up
        trump: str                 # = up_card.suit (always, since only an order-up triggers a discard)
        caller: int                # who ordered it up
        went_alone: bool

    resolve_discard(hand, up_card, dealer, caller, went_alone, choose_discard_fn)
        -> (new_five_card_hand, discarded_card)
    choose_discard_fn(view: DiscardView) -> Card
"""
import random

import pytest

import engine.bidding as bidding
from engine.bidding import resolve_discard
from engine.deck import Card
from engine.game import play_hand
from bots.random_bot import RandomBot

DiscardView = getattr(bidding, "DiscardView", None)
pytestmark = pytest.mark.skipif(DiscardView is None, reason="DiscardView not added to engine/bidding.py yet")

HAND = [Card("Hearts", "9"), Card("Hearts", "King"), Card("Spades", "Ace"),
        Card("Clubs", "9"), Card("Diamonds", "Queen")]
UP = Card("Hearts", "10")


class Recorder:
    def __init__(self, choice=lambda v: v.hand[0], tamper=False):
        self.choice = choice
        self.views = []
        self.tamper = tamper

    def __call__(self, view):
        self.views.append(view)
        pick = self.choice(view)
        if self.tamper:
            view.hand.clear()
        return pick


def run(rec, dealer=3, caller=1, went_alone=False):
    return resolve_discard(list(HAND), UP, dealer=dealer, caller=caller,
                           went_alone=went_alone, choose_discard_fn=rec)


def test_callback_receives_a_discard_view():
    rec = Recorder()
    run(rec)
    assert len(rec.views) == 1
    assert isinstance(rec.views[0], DiscardView)


def test_view_fields():
    rec = Recorder()
    run(rec, dealer=2, caller=0, went_alone=True)
    v = rec.views[0]
    assert v.player_id == 2
    assert v.up_card == UP
    assert v.trump == "Hearts"
    assert v.caller == 0
    assert v.went_alone is True


def test_view_hand_is_all_six_cards():
    rec = Recorder()
    run(rec)
    v = rec.views[0]
    assert len(v.hand) == 6
    assert set(v.hand) == set(HAND) | {UP}


def test_returns_five_card_hand_without_discard():
    rec = Recorder(choice=lambda v: Card("Clubs", "9"))
    new_hand, discard = run(rec)
    assert discard == Card("Clubs", "9")
    assert len(new_hand) == 5
    assert discard not in new_hand
    assert UP in new_hand


def test_dealer_may_discard_the_up_card_itself():
    rec = Recorder(choice=lambda v: v.up_card)
    new_hand, discard = run(rec)
    assert discard == UP
    assert set(new_hand) == set(HAND)


def test_tampering_with_view_hand_does_not_affect_result():
    rec = Recorder(choice=lambda v: Card("Clubs", "9"), tamper=True)
    new_hand, _ = run(rec)
    assert len(new_hand) == 5
    assert set(new_hand) == (set(HAND) | {UP}) - {Card("Clubs", "9")}


def test_invalid_discard_raises():
    with pytest.raises(ValueError):
        run(Recorder(choice=lambda v: Card("Spades", "King")))


def test_caller_hand_not_mutated():
    hand = list(HAND)
    resolve_discard(hand, UP, dealer=3, caller=1, went_alone=False,
                    choose_discard_fn=lambda v: v.hand[0])
    assert hand == HAND


# ---------- integration through play_hand ----------

def test_play_hand_discard_views_match_hand_result():
    rng = random.Random(21)
    bot = RandomBot(rng)
    n_checked = 0
    for h in range(300):
        seen = []

        def discard_fn(view):
            seen.append(view)
            return bot.choose_discard(view)   # RandomBot must accept a DiscardView

        r = play_hand(h % 4, rng, bot.choose_bid, discard_fn, bot.choose_play)
        if r.bid_result.winning_bid.kind == "order_up":
            assert len(seen) == 1
            v = seen[0]
            assert v.player_id == r.dealer
            assert v.trump == r.trump == r.up_card.suit
            assert v.up_card == r.up_card and v.up_card in v.hand
            assert v.caller == r.caller
            assert v.went_alone == r.went_alone
            assert len(v.hand) == 6
            assert r.discarded in v.hand
            n_checked += 1
        else:
            assert seen == []
    assert n_checked > 50

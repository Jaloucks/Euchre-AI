"""
Tests for BidView (Phase 1c engine change).

BidView is to bidding what PlayerView is to card play: exactly what the
bidder legitimately knows at the moment of decision.

    @dataclass
    class BidView:                 # in engine/bidding.py
        player_id: int
        hand: list[Card]           # the bidder's own 5 cards (a copy)
        up_card: Card              # the full up card, not just its suit
        dealer: int
        round: int                 # 1 or 2
        passes_this_round: int

    run_bidding(dealer, up_card, hands, choose_action_fn) -> BidResult
    choose_action_fn(view: BidView, legal_actions: list[BidAction]) -> BidAction
"""
import random

import pytest

import engine.bidding as bidding
from engine.deck import Card
from engine.bidding import BidAction, legal_bid_actions, run_bidding
from engine.game import play_hand
from bots.random_bot import RandomBot

BidView = getattr(bidding, "BidView", None)
pytestmark = pytest.mark.skipif(BidView is None, reason="BidView not added to engine/bidding.py yet")

UP_CARD = Card("Hearts", "10")


def make_hands():
    return {
        0: [Card("Clubs", "9"), Card("Clubs", "10"), Card("Clubs", "Queen"), Card("Clubs", "King"), Card("Clubs", "Ace")],
        1: [Card("Spades", "9"), Card("Spades", "10"), Card("Spades", "Queen"), Card("Spades", "King"), Card("Spades", "Ace")],
        2: [Card("Diamonds", "9"), Card("Diamonds", "10"), Card("Diamonds", "Queen"), Card("Diamonds", "King"), Card("Diamonds", "Ace")],
        3: [Card("Hearts", "9"), Card("Hearts", "Queen"), Card("Hearts", "King"), Card("Hearts", "Ace"), Card("Clubs", "Jack")],
    }


class Recorder:
    """Plays a scripted list of actions and records every (view, legal_actions) it saw."""
    def __init__(self, actions, tamper=False):
        self.actions = list(actions)
        self.seen = []
        self.tamper = tamper

    def __call__(self, view, legal_actions):
        self.seen.append((view, list(legal_actions)))
        if self.tamper:
            view.hand.append(Card("Spades", "Jack"))
            view.hand.pop(0)
        return self.actions[len(self.seen) - 1]


ALL_PASS_THEN_CALL = [BidAction(kind="pass")] * 4 + [BidAction(kind="call_suit", suit="Clubs")]


def test_callback_receives_a_bid_view():
    rec = Recorder([BidAction(kind="pass")] * 3 + [BidAction(kind="order_up", suit="Hearts")])
    run_bidding(dealer=3, up_card=UP_CARD, hands=make_hands(), choose_action_fn=rec)
    assert all(isinstance(v, BidView) for v, _ in rec.seen)


def test_seat_round_and_pass_count_progress_correctly():
    rec = Recorder(ALL_PASS_THEN_CALL)
    run_bidding(dealer=3, up_card=UP_CARD, hands=make_hands(), choose_action_fn=rec)
    assert [v.player_id for v, _ in rec.seen] == [0, 1, 2, 3, 0]
    assert [v.round for v, _ in rec.seen] == [1, 1, 1, 1, 2]
    assert [v.passes_this_round for v, _ in rec.seen] == [0, 1, 2, 3, 0]
    assert all(v.dealer == 3 for v, _ in rec.seen)


def test_view_hand_is_the_bidders_own_hand():
    hands = make_hands()
    rec = Recorder(ALL_PASS_THEN_CALL)
    run_bidding(dealer=3, up_card=UP_CARD, hands=hands, choose_action_fn=rec)
    for v, _ in rec.seen:
        assert v.hand == hands[v.player_id]


def test_view_has_the_full_up_card():
    rec = Recorder(ALL_PASS_THEN_CALL)
    run_bidding(dealer=3, up_card=UP_CARD, hands=make_hands(), choose_action_fn=rec)
    assert all(v.up_card == UP_CARD for v, _ in rec.seen)


def test_legal_actions_match_rules():
    rec = Recorder(ALL_PASS_THEN_CALL)
    run_bidding(dealer=3, up_card=UP_CARD, hands=make_hands(), choose_action_fn=rec)
    round1 = [a for v, a in rec.seen if v.round == 1]
    assert all({x.kind for x in legal} == {"pass", "order_up"} for legal in round1)
    round2 = [a for v, a in rec.seen if v.round == 2]
    assert all(all(x.suit != "Hearts" for x in legal if x.kind == "call_suit") for legal in round2)


def test_mutating_view_hand_does_not_touch_engine_hands():
    hands = make_hands()
    before = {p: list(h) for p, h in hands.items()}
    rec = Recorder(ALL_PASS_THEN_CALL, tamper=True)
    run_bidding(dealer=3, up_card=UP_CARD, hands=hands, choose_action_fn=rec)
    assert hands == before


def test_run_bidding_result_unchanged_by_new_signature():
    rec = Recorder(ALL_PASS_THEN_CALL)
    result = run_bidding(dealer=3, up_card=UP_CARD, hands=make_hands(), choose_action_fn=rec)
    assert result.winning_player == 0
    assert result.winning_bid == BidAction(kind="call_suit", suit="Clubs")


# ---------- integration through play_hand ----------

def _collect_bid_views(n_hands=200, seed=11):
    rng = random.Random(seed)
    bot = RandomBot(rng)
    records = []

    def bid_fn(view, legal_actions):
        records.append((view, list(legal_actions)))
        return bot.choose_bid(view, legal_actions)

    results = []
    for h in range(n_hands):
        n_before = len(records)
        r = play_hand(h % 4, rng, bid_fn, bot.choose_discard, bot.choose_play)
        results.append((r, records[n_before:]))
    return results


def test_play_hand_gives_bidders_real_hands():
    for result, recs in _collect_bid_views():
        hands_seen = {}
        for v, _ in recs:
            assert len(v.hand) == 5
            assert v.up_card == result.up_card
            assert v.up_card not in v.hand
            assert v.dealer == result.dealer
            hands_seen.setdefault(v.player_id, set(v.hand))
            assert set(v.hand) == hands_seen[v.player_id]   # same hand every time you're asked
        all_cards = [c for h in hands_seen.values() for c in h]
        assert len(all_cards) == len(set(all_cards))           # no card in two hands

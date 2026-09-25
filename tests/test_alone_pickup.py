"""
Rule change (open item 1, decided in Phase 2): when the dealer's PARTNER
orders up alone, the dealer sits out, so the dealer does NOT pick up the up
card and no discard happens.

    In play_hand: the dealer picks up only if the winning bid is "order_up"
    AND NOT (went_alone AND caller == (dealer + 2) % 4).

Every other order-up still triggers the pickup, including:
  - dealer's partner orders up WITHOUT going alone
  - an opponent orders up alone (the dealer is defending and plays)
  - the dealer orders up alone
"""
import random

import pytest

from engine.bidding import BidAction
from engine.game import play_hand


def scripted_bid(orderer, alone):
    """Everyone passes until `orderer` orders up (round 1)."""
    def bid(view, legal):
        if view.round == 1 and view.player_id == orderer:
            return BidAction(kind="order_up", suit=view.up_card.suit, alone=alone)
        if any(a.kind == "pass" for a in legal):
            return BidAction(kind="pass")
        return legal[0]
    return bid


def run(dealer, orderer, alone, seed=0):
    rng = random.Random(seed)
    discards = []

    def discard(view):
        discards.append(view)
        return view.hand[0]

    def play(view):
        return rng.choice(view.legal_cards)

    result = play_hand(dealer, rng, scripted_bid(orderer, alone), discard, play)
    return result, discards


@pytest.mark.parametrize("dealer", [0, 1, 2, 3])
def test_partner_of_dealer_alone_skips_pickup(dealer):
    partner = (dealer + 2) % 4
    for seed in range(20):
        result, discards = run(dealer, partner, alone=True, seed=seed)
        assert result.caller == partner and result.went_alone
        assert discards == [], "dealer sits out, so no discard view"
        assert result.discarded is None


@pytest.mark.parametrize("dealer", [0, 1, 2, 3])
def test_up_card_is_out_of_play_when_pickup_skipped(dealer):
    partner = (dealer + 2) % 4
    for seed in range(20):
        result, _ = run(dealer, partner, alone=True, seed=seed)
        played = [card for trick in result.completed_tricks for _, card in trick]
        assert result.up_card not in played
        assert all(pid != dealer for trick in result.completed_tricks for pid, _ in trick)


@pytest.mark.parametrize("dealer", [0, 1, 2, 3])
def test_partner_of_dealer_not_alone_still_picks_up(dealer):
    result, discards = run(dealer, (dealer + 2) % 4, alone=False)
    assert len(discards) == 1
    assert result.discarded is not None


@pytest.mark.parametrize("dealer", [0, 1, 2, 3])
@pytest.mark.parametrize("offset", [1, 3])
def test_opponent_alone_still_makes_dealer_pick_up(dealer, offset):
    result, discards = run(dealer, (dealer + offset) % 4, alone=True)
    assert len(discards) == 1
    assert discards[0].player_id == dealer
    assert result.discarded is not None


@pytest.mark.parametrize("dealer", [0, 1, 2, 3])
def test_dealer_alone_still_picks_up(dealer):
    result, discards = run(dealer, dealer, alone=True)
    assert len(discards) == 1
    assert result.discarded is not None

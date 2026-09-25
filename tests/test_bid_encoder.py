"""
Tests for encoding/bid_encoder.py (Phase 1c).

Frame: every card and suit is canonical relative to the UP-CARD SUIT
(canonical_suit_order(up_suit) = [up, next, cross1, cross2]), in both rounds.

    BID_BLOCKS  = [("hand", 24), ("up_card", 24), ("round", 2), ("dealer", 4)]   # 54 dims
    BID_OFFSETS, BID_DIM

    Fixed action space (NUM_BID_ACTIONS = 9):
        0 pass
        1 order_up            2 order_up alone
        3 call next           4 call next alone
        5 call cross1         6 call cross1 alone
        7 call cross2         8 call cross2 alone

    bid_action_to_index(action, up_suit) -> int
    index_to_bid_action(index, up_suit) -> BidAction
    encode_bid_view(view, legal_actions) -> (features float32 (BID_DIM,), mask bool (9,))
    describe_bid_features(x) -> dict keyed by block name
"""
import random

import numpy as np
import pytest

be = pytest.importorskip("encoding.bid_encoder", reason="encoding/bid_encoder.py not written yet")

import engine.bidding as bidding
from engine.bidding import BidAction, BiddingState, legal_bid_actions
from engine.deck import Card, SUITS, SAME_COLOR
from engine.game import play_hand
from encoding.canonical import card_to_index, canonical_suit_order, relative_seat
from bots.random_bot import RandomBot

BidView = getattr(bidding, "BidView", None)
pytestmark = pytest.mark.skipif(BidView is None, reason="BidView not added to engine/bidding.py yet")

EXPECTED_BLOCKS = [("hand", 24), ("up_card", 24), ("round", 2), ("dealer", 4)]
ALL_SUITS = pytest.mark.parametrize("up_suit", SUITS)


# ---------- helpers ----------

def C(rank, suit):
    return Card(suit, rank)


def make_view(**overrides):
    d = dict(
        player_id=0,
        hand=[C("Jack", "Hearts"), C("Jack", "Diamonds"), C("Ace", "Hearts"), C("9", "Clubs"), C("King", "Spades")],
        up_card=C("10", "Hearts"),
        dealer=3,
        round=1,
        passes_this_round=0,
    )
    d.update(overrides)
    return BidView(**d)


def legal_for(view):
    state = BiddingState(
        up_card_suit=view.up_card.suit, dealer=view.dealer,
        current_player=view.player_id, round=view.round,
        passes_this_round=view.passes_this_round,
    )
    return legal_bid_actions(state)


def block(x, name):
    return x[be.BID_OFFSETS[name]]


def onehot(n, i):
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


# ---------- layout ----------

def test_block_layout_matches_spec():
    assert list(be.BID_BLOCKS) == EXPECTED_BLOCKS


def test_bid_dim():
    assert be.BID_DIM == 54


def test_offsets_are_contiguous():
    pos = 0
    for name, size in EXPECTED_BLOCKS:
        assert (be.BID_OFFSETS[name].start, be.BID_OFFSETS[name].stop) == (pos, pos + size)
        pos += size


def test_num_bid_actions():
    assert be.NUM_BID_ACTIONS == 9


# ---------- action indexing ----------

@ALL_SUITS
def test_pass_is_index_0(up_suit):
    assert be.bid_action_to_index(BidAction(kind="pass"), up_suit) == 0
    assert be.index_to_bid_action(0, up_suit) == BidAction(kind="pass")


@ALL_SUITS
def test_order_up_indices(up_suit):
    assert be.bid_action_to_index(BidAction("order_up", up_suit, False), up_suit) == 1
    assert be.bid_action_to_index(BidAction("order_up", up_suit, True), up_suit) == 2


@ALL_SUITS
def test_call_next_indices(up_suit):
    nxt = SAME_COLOR[up_suit]
    assert be.bid_action_to_index(BidAction("call_suit", nxt, False), up_suit) == 3
    assert be.bid_action_to_index(BidAction("call_suit", nxt, True), up_suit) == 4


@ALL_SUITS
def test_call_cross_indices_follow_canonical_order(up_suit):
    _, _, c1, c2 = canonical_suit_order(up_suit)
    assert be.bid_action_to_index(BidAction("call_suit", c1, False), up_suit) == 5
    assert be.bid_action_to_index(BidAction("call_suit", c1, True), up_suit) == 6
    assert be.bid_action_to_index(BidAction("call_suit", c2, False), up_suit) == 7
    assert be.bid_action_to_index(BidAction("call_suit", c2, True), up_suit) == 8


@ALL_SUITS
def test_index_round_trip(up_suit):
    for i in range(be.NUM_BID_ACTIONS):
        assert be.bid_action_to_index(be.index_to_bid_action(i, up_suit), up_suit) == i


@ALL_SUITS
def test_every_legal_engine_action_round_trips(up_suit):
    # Decoded actions must compare equal to the engine's own BidAction objects.
    for rnd, passes, player in [(1, 0, 0), (2, 0, 0), (2, 3, 3)]:
        state = BiddingState(up_card_suit=up_suit, dealer=3, current_player=player,
                             round=rnd, passes_this_round=passes)
        for a in legal_bid_actions(state):
            assert be.index_to_bid_action(be.bid_action_to_index(a, up_suit), up_suit) == a


@ALL_SUITS
def test_calling_the_turned_down_suit_is_rejected(up_suit):
    with pytest.raises(ValueError):
        be.bid_action_to_index(BidAction("call_suit", up_suit, False), up_suit)


@pytest.mark.parametrize("bad", [-1, 9, 100])
def test_index_to_bid_action_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        be.index_to_bid_action(bad, "Hearts")


# ---------- features ----------

def test_output_shapes_and_dtypes():
    v = make_view()
    x, mask = be.encode_bid_view(v, legal_for(v))
    assert x.dtype == np.float32 and x.shape == (be.BID_DIM,)
    assert mask.dtype == np.bool_ and mask.shape == (be.NUM_BID_ACTIONS,)


def test_values_are_binary():
    v = make_view()
    x, _ = be.encode_bid_view(v, legal_for(v))
    assert set(np.unique(x)).issubset({0.0, 1.0})


def test_hand_is_canonical_in_up_suit_frame():
    v = make_view()   # up card is a Heart, so Hearts is the frame's "trump"
    x, _ = be.encode_bid_view(v, legal_for(v))
    expected = np.zeros(24, dtype=np.float32)
    for c in v.hand:
        expected[card_to_index(c, "Hearts")] = 1.0
    assert np.array_equal(block(x, "hand"), expected)
    assert block(x, "hand")[0] == 1.0   # J♥ = right bower if ordered up
    assert block(x, "hand")[1] == 1.0   # J♦ = left bower if ordered up


def test_up_card_one_hot():
    v = make_view()
    x, _ = be.encode_bid_view(v, legal_for(v))
    assert np.array_equal(block(x, "up_card"), onehot(24, card_to_index(C("10", "Hearts"), "Hearts")))


@pytest.mark.parametrize("rnd", [1, 2])
def test_round_one_hot(rnd):
    v = make_view(round=rnd)
    x, _ = be.encode_bid_view(v, legal_for(v))
    assert np.array_equal(block(x, "round"), onehot(2, rnd - 1))


@pytest.mark.parametrize("player, dealer, rel", [(0, 3, 3), (0, 0, 0), (2, 3, 1), (1, 3, 2)])
def test_dealer_is_relative_seat(player, dealer, rel):
    v = make_view(player_id=player, dealer=dealer)
    x, _ = be.encode_bid_view(v, legal_for(v))
    assert np.array_equal(block(x, "dealer"), onehot(4, rel))


# ---------- mask ----------

def test_round1_mask():
    v = make_view(round=1)
    _, mask = be.encode_bid_view(v, legal_for(v))
    assert list(np.flatnonzero(mask)) == [0, 1, 2]


def test_round2_mask_normal():
    v = make_view(round=2, passes_this_round=0)
    _, mask = be.encode_bid_view(v, legal_for(v))
    assert list(np.flatnonzero(mask)) == [0, 3, 4, 5, 6, 7, 8]


def test_round2_mask_stuck_dealer_cannot_pass():
    v = make_view(player_id=3, dealer=3, round=2, passes_this_round=3)
    _, mask = be.encode_bid_view(v, legal_for(v))
    assert list(np.flatnonzero(mask)) == [3, 4, 5, 6, 7, 8]


def test_mask_comes_from_legal_actions_argument():
    v = make_view()
    _, mask = be.encode_bid_view(v, [BidAction(kind="pass")])
    assert list(np.flatnonzero(mask)) == [0]


# ---------- symmetry ----------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_invariant_to_seat_rotation(k):
    v = make_view(round=2, passes_this_round=1, player_id=1, dealer=3)
    w = make_view(round=2, passes_this_round=1, player_id=(1 + k) % 4, dealer=(3 + k) % 4)
    a, ma = be.encode_bid_view(v, legal_for(v))
    b, mb = be.encode_bid_view(w, legal_for(w))
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


def test_invariant_to_suit_relabeling():
    # Hearts->Spades, Diamonds->Clubs (up/next pair); Clubs->Hearts, Spades->Diamonds (cross pair)
    m = {"Hearts": "Spades", "Diamonds": "Clubs", "Clubs": "Hearts", "Spades": "Diamonds"}
    assert canonical_suit_order("Spades") == [m[s] for s in canonical_suit_order("Hearts")]
    v = make_view(round=2)
    w = make_view(round=2, hand=[Card(m[c.suit], c.rank) for c in v.hand],
                  up_card=Card(m[v.up_card.suit], v.up_card.rank))
    a, ma = be.encode_bid_view(v, legal_for(v))
    b, mb = be.encode_bid_view(w, legal_for(w))
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


# ---------- describe ----------

def test_describe_returns_every_block():
    v = make_view()
    x, _ = be.encode_bid_view(v, legal_for(v))
    assert set(be.describe_bid_features(x)) == {n for n, _ in EXPECTED_BLOCKS}


def test_describe_rejects_wrong_length():
    with pytest.raises(ValueError):
        be.describe_bid_features(np.zeros(3, dtype=np.float32))


# ---------- integration over real random games ----------

@pytest.fixture(scope="module")
def real_bids():
    rng = random.Random(3)
    bot = RandomBot(rng)
    recs = []

    def bid_fn(view, legal):
        recs.append((view, list(legal)))
        return bot.choose_bid(view, legal)

    for h in range(300):
        play_hand(h % 4, rng, bid_fn, bot.choose_discard, bot.choose_play)
    return recs


def test_real_bids_invariants(real_bids):
    assert len(real_bids) > 300
    for v, legal in real_bids:
        x, mask = be.encode_bid_view(v, legal)
        assert block(x, "hand").sum() == 5
        up_idx = int(np.flatnonzero(block(x, "up_card"))[0])
        assert up_idx in (0, 2, 3, 4, 5, 6)   # the up card is always in the frame's trump block, never the left bower
        assert block(x, "hand")[up_idx] == 0.0  # the up card is not in anyone's hand
        assert mask.sum() == len(legal)


def test_real_bids_masked_choice_decodes_to_legal_action(real_bids):
    rng = np.random.default_rng(0)
    for v, legal in real_bids:
        _, mask = be.encode_bid_view(v, legal)
        i = int(rng.choice(np.flatnonzero(mask)))
        assert be.index_to_bid_action(i, v.up_card.suit) in legal

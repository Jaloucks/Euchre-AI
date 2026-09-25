"""
Tests for encoding/discard_encoder.py (Phase 1d).

Frame: canonical relative to view.trump (always the up-card suit here).

    DISCARD_BLOCKS = [("hand", 24), ("up_card", 24), ("caller", 4), ("went_alone", 1)]   # 53 dims
    DISCARD_OFFSETS, DISCARD_DIM

    encode_discard_view(view) -> (features float32 (DISCARD_DIM,), mask bool (24,))
        mask is True exactly at the six cards in view.hand
    decode_discard_action(index, view) -> Card
    describe_discard_features(x, trump=None) -> dict keyed by block name
"""
import random

import numpy as np
import pytest

de = pytest.importorskip("encoding.discard_encoder", reason="encoding/discard_encoder.py not written yet")

import engine.bidding as bidding
from engine.deck import Card
from engine.game import play_hand
from encoding.canonical import card_to_index, canonical_suit_order
from bots.random_bot import RandomBot

DiscardView = getattr(bidding, "DiscardView", None)
pytestmark = pytest.mark.skipif(DiscardView is None, reason="DiscardView not added to engine/bidding.py yet")

EXPECTED_BLOCKS = [("hand", 24), ("up_card", 24), ("caller", 4), ("went_alone", 1)]


def C(rank, suit):
    return Card(suit, rank)


def make_view(**overrides):
    d = dict(
        player_id=3,
        hand=[C("Jack", "Hearts"), C("Jack", "Diamonds"), C("10", "Hearts"),
              C("9", "Clubs"), C("King", "Spades"), C("Queen", "Diamonds")],
        up_card=C("10", "Hearts"),
        trump="Hearts",
        caller=1,
        went_alone=False,
    )
    d.update(overrides)
    return DiscardView(**d)


def block(x, name):
    return x[de.DISCARD_OFFSETS[name]]


def onehot(n, i):
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


# ---------- layout ----------

def test_block_layout_matches_spec():
    assert list(de.DISCARD_BLOCKS) == EXPECTED_BLOCKS


def test_discard_dim():
    assert de.DISCARD_DIM == 53


def test_offsets_are_contiguous():
    pos = 0
    for name, size in EXPECTED_BLOCKS:
        assert (de.DISCARD_OFFSETS[name].start, de.DISCARD_OFFSETS[name].stop) == (pos, pos + size)
        pos += size


# ---------- features ----------

def test_output_shapes_and_dtypes():
    x, mask = de.encode_discard_view(make_view())
    assert x.dtype == np.float32 and x.shape == (de.DISCARD_DIM,)
    assert mask.dtype == np.bool_ and mask.shape == (24,)


def test_values_are_binary():
    x, _ = de.encode_discard_view(make_view())
    assert set(np.unique(x)).issubset({0.0, 1.0})


def test_hand_is_canonical_multi_hot_of_six():
    v = make_view()
    x, _ = de.encode_discard_view(v)
    expected = np.zeros(24, dtype=np.float32)
    for c in v.hand:
        expected[card_to_index(c, "Hearts")] = 1.0
    assert np.array_equal(block(x, "hand"), expected)
    assert block(x, "hand").sum() == 6
    assert block(x, "hand")[0] == 1.0 and block(x, "hand")[1] == 1.0   # both bowers


def test_up_card_is_one_hot_and_also_in_hand():
    v = make_view()
    x, _ = de.encode_discard_view(v)
    idx = card_to_index(v.up_card, "Hearts")
    assert np.array_equal(block(x, "up_card"), onehot(24, idx))
    assert block(x, "hand")[idx] == 1.0


@pytest.mark.parametrize("caller, rel", [(3, 0), (0, 1), (1, 2), (2, 3)])
def test_caller_is_relative_to_dealer(caller, rel):
    # player_id (dealer) = 3: caller 3 = me, 0 = left, 1 = partner, 2 = right
    x, _ = de.encode_discard_view(make_view(caller=caller))
    assert np.array_equal(block(x, "caller"), onehot(4, rel))


@pytest.mark.parametrize("alone", [False, True])
def test_went_alone_flag(alone):
    x, _ = de.encode_discard_view(make_view(went_alone=alone))
    assert block(x, "went_alone")[0] == float(alone)


# ---------- mask and decode ----------

def test_mask_is_exactly_the_six_cards():
    v = make_view()
    _, mask = de.encode_discard_view(v)
    expected = np.zeros(24, dtype=bool)
    for c in v.hand:
        expected[card_to_index(c, "Hearts")] = True
    assert np.array_equal(mask, expected)


def test_decode_inverts_canonical_index():
    v = make_view()
    for c in v.hand:
        assert de.decode_discard_action(card_to_index(c, v.trump), v) == c


def test_decode_rejects_out_of_range():
    with pytest.raises(ValueError):
        de.decode_discard_action(24, make_view())


# ---------- symmetry ----------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_invariant_to_seat_rotation(k):
    a, ma = de.encode_discard_view(make_view(player_id=3, caller=1))
    b, mb = de.encode_discard_view(make_view(player_id=(3 + k) % 4, caller=(1 + k) % 4))
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


def test_invariant_to_suit_relabeling():
    m = {"Hearts": "Spades", "Diamonds": "Clubs", "Clubs": "Hearts", "Spades": "Diamonds"}
    assert canonical_suit_order("Spades") == [m[s] for s in canonical_suit_order("Hearts")]
    v = make_view()
    w = make_view(hand=[Card(m[c.suit], c.rank) for c in v.hand],
                  up_card=Card(m[v.up_card.suit], v.up_card.rank), trump=m[v.trump])
    a, ma = de.encode_discard_view(v)
    b, mb = de.encode_discard_view(w)
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


# ---------- describe ----------

def test_describe_returns_every_block():
    x, _ = de.encode_discard_view(make_view())
    assert set(de.describe_discard_features(x)) == {n for n, _ in EXPECTED_BLOCKS}


def test_describe_rejects_wrong_length():
    with pytest.raises(ValueError):
        de.describe_discard_features(np.zeros(5, dtype=np.float32))


# ---------- integration over real random games ----------

def test_real_discards_encode_and_decode():
    rng = random.Random(5)
    bot = RandomBot(rng)
    views = []

    def discard_fn(view):
        views.append(view)
        return bot.choose_discard(view)

    for h in range(300):
        play_hand(h % 4, rng, bot.choose_bid, discard_fn, bot.choose_play)
    assert len(views) > 50

    pick = np.random.default_rng(0)
    for v in views:
        x, mask = de.encode_discard_view(v)
        assert mask.sum() == 6
        assert block(x, "hand").sum() == 6
        assert block(x, "up_card").sum() == 1
        up_idx = int(np.flatnonzero(block(x, "up_card"))[0])
        assert up_idx in (0, 2, 3, 4, 5, 6)   # up card is always trump, never the left bower
        i = int(pick.choice(np.flatnonzero(mask)))
        assert de.decode_discard_action(i, v) in v.hand

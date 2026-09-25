"""
Tests for encoding/play_encoder.py (Phase 1b).

Spec summary (see chat for the reasoning):

    encode_play_view(view) -> (features, mask)
        features: np.float32, shape (PLAY_DIM,), every value 0.0 or 1.0
        mask:     np.bool_,   shape (24,), True exactly at legal cards' canonical indices
    decode_play_action(index, view) -> Card
    describe_play_features(features, trump) -> dict keyed by block name (debug aid)

    PLAY_BLOCKS: ordered list of (name, size); PLAY_OFFSETS: name -> slice; PLAY_DIM: total.

All seats are relative to the viewer (encoding.canonical.relative_seat).
All cards/suits are canonical relative to view.trump (encoding.canonical).
"""
import random

import numpy as np
import pytest

pe = pytest.importorskip("encoding.play_encoder", reason="encoding/play_encoder.py not written yet")

from engine.deck import Card, SUITS, create_deck
from engine.game import PlayerView, play_hand
from encoding.canonical import card_to_index, canonical_suit_order
from bots.random_bot import RandomBot
from void_oracle import derive_voids  # tests/ is on sys.path under pytest

EXPECTED_BLOCKS = [
    ("hand", 24),
    ("trick_left", 24), ("trick_partner", 24), ("trick_right", 24),
    ("played_me", 24), ("played_left", 24), ("played_partner", 24), ("played_right", 24),
    ("void_left", 4), ("void_partner", 4), ("void_right", 4),
    ("led_suit", 4),
    ("trick_position", 4),
    ("caller", 4),
    ("went_alone", 1),
    ("dealer", 4),
    ("active", 4),
    ("up_card", 24),
    ("tricks_mine", 6), ("tricks_theirs", 6),
]


# ---------- helpers ----------

def C(rank, suit):
    return Card(suit, rank)


def make_view(**overrides) -> PlayerView:
    """A valid mid-hand view for player 0, trump Spades. Override any field."""
    defaults = dict(
        player_id=0,
        hand=[C("Jack", "Spades"), C("Ace", "Hearts"), C("9", "Diamonds")],
        legal_cards=[C("Jack", "Spades"), C("Ace", "Hearts"), C("9", "Diamonds")],
        trump="Spades",
        up_card=C("Queen", "Spades"),
        caller=1,
        went_alone=False,
        active_players=[0, 1, 2, 3],
        dealer=3,
        led_suit=None,
        cards_played_this_trick=[],
        completed_tricks=[],
        tricks_won={0: 0, 1: 0, 2: 0, 3: 0},
    )
    defaults.update(overrides)
    if "void_suits" not in overrides:
        # Hand-built views get voids the way the engine would compute them.
        defaults["void_suits"] = derive_voids(
            defaults["completed_tricks"], defaults["cards_played_this_trick"], defaults["trump"]
        )
    return PlayerView(**defaults)


def block(x, name):
    return x[pe.PLAY_OFFSETS[name]]


def onehot(size, i):
    v = np.zeros(size, dtype=np.float32)
    v[i] = 1.0
    return v


def cards_vec(cards, trump):
    v = np.zeros(24, dtype=np.float32)
    for c in cards:
        v[card_to_index(c, trump)] = 1.0
    return v


def rotate_seats(view: PlayerView, k: int) -> PlayerView:
    """Same situation, every seat id shifted by k."""
    r = lambda p: (p + k) % 4
    return PlayerView(
        player_id=r(view.player_id),
        hand=list(view.hand),
        legal_cards=list(view.legal_cards),
        trump=view.trump,
        up_card=view.up_card,
        caller=r(view.caller),
        went_alone=view.went_alone,
        active_players=sorted(r(p) for p in view.active_players),
        dealer=r(view.dealer),
        led_suit=view.led_suit,
        cards_played_this_trick=[(r(p), c) for p, c in view.cards_played_this_trick],
        completed_tricks=[[(r(p), c) for p, c in t] for t in view.completed_tricks],
        tricks_won={r(p): n for p, n in view.tricks_won.items()},
        void_suits={r(p): set(s) for p, s in view.void_suits.items()},
    )


def relabel_suits(view: PlayerView, m: dict) -> PlayerView:
    """Same situation with suits renamed by m (must map color pairs to color pairs)."""
    rc = lambda c: Card(m[c.suit], c.rank)
    return PlayerView(
        player_id=view.player_id,
        hand=[rc(c) for c in view.hand],
        legal_cards=[rc(c) for c in view.legal_cards],
        trump=m[view.trump],
        up_card=rc(view.up_card),
        caller=view.caller,
        went_alone=view.went_alone,
        active_players=list(view.active_players),
        dealer=view.dealer,
        led_suit=None if view.led_suit is None else m[view.led_suit],
        cards_played_this_trick=[(p, rc(c)) for p, c in view.cards_played_this_trick],
        completed_tricks=[[(p, rc(c)) for p, c in t] for t in view.completed_tricks],
        tricks_won=dict(view.tricks_won),
        void_suits={p: {m[x] for x in s} for p, s in view.void_suits.items()},
    )


def collect_views(n_hands=300, seed=0):
    """Real PlayerViews from random games."""
    rng = random.Random(seed)
    bot = RandomBot(rng)
    views = []

    def play_fn(view):
        views.append(view)
        return bot.choose_play(view)

    for h in range(n_hands):
        play_hand(h % 4, rng, bot.choose_bid, bot.choose_discard, play_fn)
    return views


@pytest.fixture(scope="module")
def real_views():
    return collect_views()


# ---------- layout ----------

def test_block_layout_matches_spec():
    assert list(pe.PLAY_BLOCKS) == EXPECTED_BLOCKS


def test_play_dim_is_sum_of_blocks():
    assert pe.PLAY_DIM == sum(size for _, size in EXPECTED_BLOCKS) == 261


def test_offsets_are_contiguous_and_in_order():
    pos = 0
    for name, size in EXPECTED_BLOCKS:
        s = pe.PLAY_OFFSETS[name]
        assert (s.start, s.stop) == (pos, pos + size), name
        pos += size
    assert pos == pe.PLAY_DIM


# ---------- output contract ----------

def test_output_shapes_and_dtypes():
    x, mask = pe.encode_play_view(make_view())
    assert isinstance(x, np.ndarray) and x.dtype == np.float32 and x.shape == (pe.PLAY_DIM,)
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == (24,)


def test_encoding_does_not_mutate_view():
    v = make_view(cards_played_this_trick=[(3, C("9", "Hearts"))], led_suit="Hearts")
    before = repr(v)
    pe.encode_play_view(v)
    assert repr(v) == before


def test_encoding_is_deterministic():
    v = make_view()
    a, ma = pe.encode_play_view(v)
    b, mb = pe.encode_play_view(v)
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


# ---------- hand and mask ----------

def test_hand_block_is_multi_hot_of_hand():
    v = make_view()
    x, _ = pe.encode_play_view(v)
    assert np.array_equal(block(x, "hand"), cards_vec(v.hand, "Spades"))


def test_right_bower_in_hand_sets_index_0():
    x, _ = pe.encode_play_view(make_view())
    assert block(x, "hand")[0] == 1.0


def test_mask_is_exactly_legal_cards():
    v = make_view(
        led_suit="Hearts",
        cards_played_this_trick=[(3, C("9", "Hearts"))],
        legal_cards=[C("Ace", "Hearts")],   # must follow suit
    )
    _, mask = pe.encode_play_view(v)
    expected = np.zeros(24, dtype=bool)
    expected[card_to_index(C("Ace", "Hearts"), "Spades")] = True
    assert np.array_equal(mask, expected)


def test_decode_inverts_canonical_index():
    v = make_view()
    for card in v.legal_cards:
        assert pe.decode_play_action(card_to_index(card, v.trump), v) == card


# ---------- current trick ----------

def test_current_trick_blocks_by_relative_seat():
    # Player 0 is 4th to act: left(1) led, partner(2), right(3) followed.
    v = make_view(
        led_suit="Hearts",
        cards_played_this_trick=[
            (1, C("King", "Hearts")), (2, C("10", "Hearts")), (3, C("Jack", "Spades")),
        ],
        legal_cards=[C("Ace", "Hearts")],
    )
    x, _ = pe.encode_play_view(v)
    assert np.array_equal(block(x, "trick_left"), cards_vec([C("King", "Hearts")], "Spades"))
    assert np.array_equal(block(x, "trick_partner"), cards_vec([C("10", "Hearts")], "Spades"))
    assert np.array_equal(block(x, "trick_right"), cards_vec([C("Jack", "Spades")], "Spades"))


def test_seats_that_have_not_played_are_all_zero():
    v = make_view(led_suit="Hearts", cards_played_this_trick=[(3, C("9", "Hearts"))])
    x, _ = pe.encode_play_view(v)
    assert block(x, "trick_left").sum() == 0
    assert block(x, "trick_partner").sum() == 0
    assert block(x, "trick_right").sum() == 1


@pytest.mark.parametrize("n_played", [0, 1, 2, 3])
def test_trick_position_is_number_of_cards_already_played(n_played):
    plays = [(3, C("9", "Hearts")), (0, C("10", "Hearts")), (1, C("Queen", "Hearts"))]
    # viewer is the next player after those plays
    viewer = (3 + n_played) % 4
    v = make_view(
        player_id=viewer,
        led_suit="Hearts" if n_played else None,
        cards_played_this_trick=plays[:n_played],
    )
    x, _ = pe.encode_play_view(v)
    assert np.array_equal(block(x, "trick_position"), onehot(4, n_played))


# ---------- led suit ----------

def test_led_suit_zero_when_leading():
    x, _ = pe.encode_play_view(make_view(led_suit=None))
    assert block(x, "led_suit").sum() == 0


@pytest.mark.parametrize("led, idx", [("Spades", 0), ("Clubs", 1)])
def test_led_suit_is_canonical(led, idx):
    # trump Spades: Spades -> 0 (trump), Clubs -> 1 (next)
    x, _ = pe.encode_play_view(make_view(led_suit=led, cards_played_this_trick=[(3, C("9", led))]))
    assert np.array_equal(block(x, "led_suit"), onehot(4, idx))


def test_led_suit_cross_uses_canonical_order():
    order = canonical_suit_order("Spades")
    x, _ = pe.encode_play_view(make_view(led_suit=order[3], cards_played_this_trick=[(3, C("9", order[3]))]))
    assert np.array_equal(block(x, "led_suit"), onehot(4, 3))


# ---------- history ----------

def test_played_blocks_collect_completed_tricks_by_seat():
    tricks = [
        [(1, C("Ace", "Clubs")), (2, C("9", "Clubs")), (3, C("10", "Clubs")), (0, C("King", "Clubs"))],
        [(1, C("Ace", "Diamonds")), (2, C("King", "Diamonds")), (3, C("Queen", "Diamonds")), (0, C("10", "Diamonds"))],
    ]
    v = make_view(completed_tricks=tricks, tricks_won={0: 0, 1: 2, 2: 0, 3: 0})
    x, _ = pe.encode_play_view(v)
    assert np.array_equal(block(x, "played_me"), cards_vec([C("King", "Clubs"), C("10", "Diamonds")], "Spades"))
    assert np.array_equal(block(x, "played_left"), cards_vec([C("Ace", "Clubs"), C("Ace", "Diamonds")], "Spades"))
    assert np.array_equal(block(x, "played_partner"), cards_vec([C("9", "Clubs"), C("King", "Diamonds")], "Spades"))
    assert np.array_equal(block(x, "played_right"), cards_vec([C("10", "Clubs"), C("Queen", "Diamonds")], "Spades"))


def test_current_trick_is_not_counted_as_completed_history():
    v = make_view(led_suit="Hearts", cards_played_this_trick=[(3, C("9", "Hearts"))])
    x, _ = pe.encode_play_view(v)
    assert block(x, "played_right").sum() == 0


# ---------- voids ----------

def test_failing_to_follow_marks_void_in_led_suit():
    # Hearts led by left(1); partner(2) plays a Diamond -> partner void in Hearts.
    tricks = [[(1, C("Ace", "Hearts")), (2, C("9", "Diamonds")), (3, C("10", "Hearts")), (0, C("King", "Hearts"))]]
    x, _ = pe.encode_play_view(make_view(completed_tricks=tricks, tricks_won={0: 0, 1: 1, 2: 0, 3: 0}))
    hearts = canonical_suit_order("Spades").index("Hearts")
    assert np.array_equal(block(x, "void_partner"), onehot(4, hearts))
    assert block(x, "void_left").sum() == 0
    assert block(x, "void_right").sum() == 0


def test_left_bower_does_not_follow_its_printed_suit():
    # Trump Spades. Clubs led; right(3) plays Jack of Clubs = left bower = TRUMP.
    # So right did NOT follow clubs -> right is void in Clubs (canonical index 1).
    tricks = [[(1, C("9", "Clubs")), (2, C("Ace", "Clubs")), (3, C("Jack", "Clubs")), (0, C("King", "Clubs"))]]
    x, _ = pe.encode_play_view(make_view(completed_tricks=tricks, tricks_won={0: 0, 1: 0, 2: 0, 3: 1}))
    assert np.array_equal(block(x, "void_right"), onehot(4, 1))


def test_left_bower_follows_a_trump_lead():
    # Trump Spades led; left(1) plays Jack of Clubs (trump) -> NOT void.
    tricks = [[(0, C("Ace", "Spades")), (1, C("Jack", "Clubs")), (2, C("9", "Spades")), (3, C("10", "Spades"))]]
    x, _ = pe.encode_play_view(make_view(completed_tricks=tricks, tricks_won={0: 0, 1: 1, 2: 0, 3: 0}))
    assert block(x, "void_left").sum() == 0


def test_void_detected_in_current_trick():
    # Viewer 0 is last to act; left(1) led Hearts, partner(2) threw a Diamond.
    v = make_view(
        led_suit="Hearts",
        cards_played_this_trick=[(1, C("9", "Hearts")), (2, C("Ace", "Diamonds")), (3, C("10", "Hearts"))],
        legal_cards=[C("Ace", "Hearts")],
    )
    x, _ = pe.encode_play_view(v)
    hearts = canonical_suit_order("Spades").index("Hearts")
    assert np.array_equal(block(x, "void_partner"), onehot(4, hearts))


def test_void_is_sticky_across_tricks():
    tricks = [
        [(1, C("Ace", "Hearts")), (2, C("9", "Diamonds")), (3, C("10", "Hearts")), (0, C("King", "Hearts"))],
        [(1, C("Ace", "Clubs")), (2, C("Queen", "Clubs")), (3, C("9", "Clubs")), (0, C("10", "Clubs"))],
    ]
    x, _ = pe.encode_play_view(make_view(completed_tricks=tricks, tricks_won={0: 0, 1: 2, 2: 0, 3: 0}))
    hearts = canonical_suit_order("Spades").index("Hearts")
    assert block(x, "void_partner")[hearts] == 1.0


def test_encoder_reads_void_suits_from_view():
    # The encoder should use view.void_suits, not re-derive voids itself.
    v = make_view(void_suits={0: set(), 1: {"Hearts", "Spades"}, 2: set(), 3: {"Clubs"}})
    x, _ = pe.encode_play_view(v)
    order = canonical_suit_order("Spades")
    expected_left = np.zeros(4, dtype=np.float32)
    expected_left[order.index("Hearts")] = 1.0
    expected_left[order.index("Spades")] = 1.0
    assert np.array_equal(block(x, "void_left"), expected_left)
    assert np.array_equal(block(x, "void_right"), onehot(4, order.index("Clubs")))
    assert block(x, "void_partner").sum() == 0


def test_own_voids_are_not_encoded():
    v = make_view(void_suits={0: {"Hearts"}, 1: set(), 2: set(), 3: set()})
    x, _ = pe.encode_play_view(v)
    for name in ("void_left", "void_partner", "void_right"):
        assert block(x, name).sum() == 0


# ---------- hand context ----------

@pytest.mark.parametrize("caller, rel", [(0, 0), (1, 1), (2, 2), (3, 3)])
def test_caller_is_relative_seat(caller, rel):
    x, _ = pe.encode_play_view(make_view(player_id=0, caller=caller))
    assert np.array_equal(block(x, "caller"), onehot(4, rel))


def test_caller_relative_for_non_zero_viewer():
    # Viewer 2, caller 0 -> caller is viewer's partner
    x, _ = pe.encode_play_view(make_view(player_id=2, caller=0))
    assert np.array_equal(block(x, "caller"), onehot(4, 2))


def test_dealer_is_relative_seat():
    x, _ = pe.encode_play_view(make_view(player_id=1, dealer=0))
    assert np.array_equal(block(x, "dealer"), onehot(4, 3))  # dealer is to viewer's right


def test_went_alone_flag_and_active_mask():
    # Caller 1 goes alone; partner 3 sits out. Viewer 0 -> seat 3 is relative 3.
    v = make_view(caller=1, went_alone=True, active_players=[0, 1, 2])
    x, _ = pe.encode_play_view(v)
    assert block(x, "went_alone")[0] == 1.0
    assert np.array_equal(block(x, "active"), np.array([1, 1, 1, 0], dtype=np.float32))


def test_not_alone_all_active():
    x, _ = pe.encode_play_view(make_view())
    assert block(x, "went_alone")[0] == 0.0
    assert np.array_equal(block(x, "active"), np.ones(4, dtype=np.float32))


def test_up_card_is_canonical_one_hot():
    x, _ = pe.encode_play_view(make_view(up_card=C("Queen", "Spades")))
    assert np.array_equal(block(x, "up_card"), onehot(24, card_to_index(C("Queen", "Spades"), "Spades")))


def test_tricks_won_are_summed_per_team():
    # Viewer 0 + partner 2 = 1 + 2 = 3; opponents 1 + 3 = 0 + 1 = 1
    x, _ = pe.encode_play_view(make_view(tricks_won={0: 1, 1: 0, 2: 2, 3: 1}))
    assert np.array_equal(block(x, "tricks_mine"), onehot(6, 3))
    assert np.array_equal(block(x, "tricks_theirs"), onehot(6, 1))


def test_tricks_won_relative_to_viewer_team():
    x, _ = pe.encode_play_view(make_view(player_id=1, tricks_won={0: 1, 1: 0, 2: 2, 3: 1}))
    assert np.array_equal(block(x, "tricks_mine"), onehot(6, 1))
    assert np.array_equal(block(x, "tricks_theirs"), onehot(6, 3))


# ---------- symmetry: the whole point of canonical + relative encoding ----------

def _rich_view():
    return make_view(
        player_id=0,
        hand=[C("Ace", "Hearts"), C("9", "Diamonds")],
        legal_cards=[C("Ace", "Hearts")],
        caller=1, dealer=3,
        led_suit="Hearts",
        cards_played_this_trick=[(1, C("9", "Hearts")), (2, C("Jack", "Clubs")), (3, C("10", "Hearts"))],
        completed_tricks=[
            [(1, C("Ace", "Clubs")), (2, C("9", "Clubs")), (3, C("10", "Clubs")), (0, C("King", "Clubs"))],
            [(1, C("Ace", "Diamonds")), (2, C("King", "Diamonds")), (3, C("Queen", "Diamonds")), (0, C("Jack", "Spades"))],
        ],
        tricks_won={0: 1, 1: 1, 2: 0, 3: 0},
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_encoding_is_invariant_to_seat_rotation(k):
    a, ma = pe.encode_play_view(_rich_view())
    b, mb = pe.encode_play_view(rotate_seats(_rich_view(), k))
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


def test_encoding_is_invariant_to_suit_relabeling():
    # Spades->Hearts, Clubs->Diamonds (trump/next pair), Hearts->Clubs, Diamonds->Spades (cross pair).
    # Under the canonical cross order, cross1/cross2 map onto each other, so encodings must match.
    m = {"Spades": "Hearts", "Clubs": "Diamonds", "Hearts": "Clubs", "Diamonds": "Spades"}
    assert canonical_suit_order("Hearts") == [m[s] for s in canonical_suit_order("Spades")]
    a, ma = pe.encode_play_view(_rich_view())
    b, mb = pe.encode_play_view(relabel_suits(_rich_view(), m))
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


# ---------- describe (debug aid) ----------

def test_describe_returns_every_block():
    x, _ = pe.encode_play_view(_rich_view())
    d = pe.describe_play_features(x, "Spades")
    assert set(d.keys()) == {name for name, _ in EXPECTED_BLOCKS}


def test_describe_rejects_wrong_length():
    with pytest.raises(ValueError):
        pe.describe_play_features(np.zeros(10, dtype=np.float32), "Spades")


# ---------- integration over real random games ----------

def test_real_views_basic_invariants(real_views):
    assert len(real_views) > 1000
    for v in real_views:
        x, mask = pe.encode_play_view(v)
        assert x.shape == (pe.PLAY_DIM,) and x.dtype == np.float32
        assert set(np.unique(x)).issubset({0.0, 1.0})
        assert block(x, "hand").sum() == len(v.hand)
        assert mask.sum() == len(v.legal_cards) >= 1
        assert np.all(block(x, "hand")[mask] == 1.0)          # legal cards are in hand
        assert block(x, "trick_position").sum() == 1
        assert block(x, "caller").sum() == 1 and block(x, "dealer").sum() == 1
        assert block(x, "up_card").sum() == 1
        assert block(x, "tricks_mine").sum() == 1 and block(x, "tricks_theirs").sum() == 1


def test_real_views_cards_seen_are_disjoint(real_views):
    # A card can only be in one place: my hand, current trick, or history.
    names = ["hand", "trick_left", "trick_partner", "trick_right",
             "played_me", "played_left", "played_partner", "played_right"]
    for v in real_views:
        x, _ = pe.encode_play_view(v)
        total = sum(block(x, n) for n in names)
        assert total.max() <= 1.0


def test_real_views_masked_choice_decodes_to_legal_card(real_views):
    rng = np.random.default_rng(0)
    for v in real_views:
        _, mask = pe.encode_play_view(v)
        idx = int(rng.choice(np.flatnonzero(mask)))
        assert pe.decode_play_action(idx, v) in v.legal_cards


def test_real_views_inactive_partner_has_no_voids_or_plays(real_views):
    alone = [v for v in real_views if v.went_alone and v.player_id != (v.caller + 2) % 4]
    assert alone, "expected some went-alone hands in the sample"
    for v in alone:
        x, _ = pe.encode_play_view(v)
        sitting_out = (v.caller + 2) % 4
        rel = (sitting_out - v.player_id) % 4
        if rel == 0:
            continue
        name = {1: "left", 2: "partner", 3: "right"}[rel]
        assert block(x, "active")[rel] == 0.0
        assert block(x, f"void_{name}").sum() == 0
        assert block(x, f"played_{name}").sum() == 0
        assert block(x, f"trick_{name}").sum() == 0

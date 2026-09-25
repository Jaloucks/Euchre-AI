"""
Smoke tests for engine/game.py — full-hand orchestration.

These don't check strategy; they check that a complete hand can be played
thousands of times with random-but-legal decisions without blowing up, and
that the resulting state is always internally consistent.

Assumes deal_hands lives in engine.deck. Adjust imports if yours differs.
"""
import random
import pytest

from engine.deck import Card, create_deck
from engine.game import play_hand, HandContext, HandResult


# ---------- random agents ----------

def make_random_agents(rng: random.Random):
    """
    Returns (choose_bid_fn, choose_discard_fn, choose_play_fn) that pick
    uniformly at random from whatever is legal. Deliberately dumb — the
    point is to hit weird states, not to play well.
    """
    def choose_bid_fn(state, legal_actions):
        return rng.choice(legal_actions)

    def choose_discard_fn(view):
        return rng.choice(view.hand)   # view is a DiscardView

    def choose_play_fn(view):
        return rng.choice(view.legal_cards)

    return choose_bid_fn, choose_discard_fn, choose_play_fn


def play_one(seed: int) -> HandResult:
    rng = random.Random(seed)
    bid_fn, discard_fn, play_fn = make_random_agents(rng)
    dealer = seed % 4
    return play_hand(dealer, rng, bid_fn, discard_fn, play_fn)


# ---------- the actual smoke run ----------

N_HANDS = 2000


def test_many_random_hands_do_not_raise():
    for seed in range(N_HANDS):
        play_one(seed)  # any exception fails the test


def test_scores_are_always_valid():
    """Exactly one team scores, and only 1, 2, or 4 points."""
    for seed in range(N_HANDS):
        result = play_one(seed)
        scoring_teams = [t for t, pts in result.scores.items() if pts > 0]
        assert len(scoring_teams) == 1, (
            f"seed={seed}: expected exactly one scoring team, got {result.scores}"
        )
        pts = result.scores[scoring_teams[0]]
        assert pts in (1, 2, 4), f"seed={seed}: illegal point value {pts}"


def test_five_tricks_always_played():
    for seed in range(N_HANDS):
        result = play_one(seed)
        assert len(result.completed_tricks) == 5, f"seed={seed}"
        assert sum(result.tricks_won.values()) == 5, f"seed={seed}"


def test_trick_sizes_match_alone_status():
    """4 cards per trick normally, 3 when someone went alone."""
    for seed in range(N_HANDS):
        result = play_one(seed)
        expected = 3 if result.went_alone else 4
        for i, trick in enumerate(result.completed_tricks):
            assert len(trick) == expected, (
                f"seed={seed}, trick {i}: expected {expected} cards, "
                f"got {len(trick)} (went_alone={result.went_alone})"
            )


def test_sitting_out_player_never_plays_or_wins():
    for seed in range(N_HANDS):
        result = play_one(seed)
        if not result.went_alone:
            continue
        partner = (result.caller + 2) % 4
        assert result.tricks_won[partner] == 0, f"seed={seed}"
        for trick in result.completed_tricks:
            players_in_trick = [p for p, _ in trick]
            assert partner not in players_in_trick, (
                f"seed={seed}: sitting-out player {partner} played a card"
            )


def test_no_duplicate_cards_played_within_a_hand():
    """A card can only be played once per hand."""
    for seed in range(N_HANDS):
        result = play_one(seed)
        all_cards = [card for trick in result.completed_tricks for _, card in trick]
        assert len(all_cards) == len(set(all_cards)), (
            f"seed={seed}: a card was played more than once"
        )


def test_discard_only_happens_on_order_up():
    """...except when the dealer's partner went alone: the dealer sits out, no pickup."""
    for seed in range(N_HANDS):
        result = play_one(seed)
        dealer_sits_out = result.went_alone and result.caller == (result.dealer + 2) % 4
        if result.bid_result.winning_bid.kind == "order_up" and not dealer_sits_out:
            assert result.discarded is not None, f"seed={seed}"
        else:
            assert result.discarded is None, f"seed={seed}"


def test_trump_is_never_the_turned_down_suit_in_round_two():
    """If trump was called in round 2, it can't be the up-card's suit."""
    for seed in range(N_HANDS):
        result = play_one(seed)
        if result.bid_result.winning_bid.kind == "call_suit":
            assert result.trump != result.up_card.suit, f"seed={seed}"


def test_same_seed_produces_identical_hand():
    """Reproducibility — required for debugging self-play later."""
    a = play_one(12345)
    b = play_one(12345)
    assert a.scores == b.scores
    assert a.trump == b.trump
    assert a.caller == b.caller
    assert a.completed_tricks == b.completed_tricks


def test_alone_hands_actually_occur():
    """
    Sanity check on the test itself: if random agents never go alone,
    these tests aren't exercising the alone code paths at all.
    """
    alone_count = sum(1 for seed in range(500) if play_one(seed).went_alone)
    assert alone_count > 0, "no alone hands generated — alone paths untested"

# ---------- void_suits in PlayerView ----------

from bots.random_bot import RandomBot
from void_oracle import derive_voids  # tests/ is on sys.path under pytest


def _collect_views(n_hands=200, seed=7, tamper=False):
    rng = random.Random(seed)
    bot = RandomBot(rng)
    views = []

    def play_fn(view):
        views.append(view)
        if tamper:
            # A misbehaving agent scribbles on its view; the engine must not care.
            view.void_suits.setdefault(view.player_id, set()).add("JUNK")
        return bot.choose_play(view)

    for h in range(n_hands):
        play_hand(h % 4, rng, bot.choose_bid, bot.choose_discard, play_fn)
    return views


def test_void_suits_match_public_history():
    views = _collect_views()
    for v in views:
        expected = derive_voids(v.completed_tricks, v.cards_played_this_trick, v.trump)
        assert v.void_suits == expected


def test_void_suits_actually_get_populated():
    views = _collect_views()
    assert any(any(s) for v in views for s in v.void_suits.values())


def test_void_suits_only_hold_real_suit_names():
    for v in _collect_views():
        for suits in v.void_suits.values():
            assert suits <= {"Hearts", "Diamonds", "Clubs", "Spades"}


def test_mutating_a_view_does_not_leak_into_engine_state():
    # Each callback writes "JUNK" into its own entry of its view. If the engine
    # handed out its real dict instead of a copy, JUNK would show up in later
    # views under that player's id when someone ELSE is viewing.
    for v in _collect_views(tamper=True):
        for p, suits in v.void_suits.items():
            if p != v.player_id:
                assert "JUNK" not in suits

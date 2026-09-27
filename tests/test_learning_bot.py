"""
Tests for rl/learning_bot.py (Phase 3, step 3d part 2).

    @dataclass
    class Decision:
        features: np.ndarray   # (261,) float32, exactly what encode_play_view returned
        mask: np.ndarray       # (24,) bool, exactly what encode_play_view returned
        action: int            # canonical index (0-23) of the card played
        log_prob: float        # log-probability of that action when it was chosen
        player_id: int         # view.player_id

    LearningPlayBot(net, rng: random.Random, training: bool = True)
        .net, .rng, .training
        choose_play(view) -> Card
        pop_trajectory() -> list[Decision]   # returns every recorded Decision (oldest first) and clears them

    choose_play, training mode:
        1. features, mask = encode_play_view(view)
        2. logits, _ = net.forward(features)
        3. log_probs = masked_log_softmax(logits, mask); probs = exp(log_probs)
        4. u = rng.random()        <- exactly ONE call per decision, even for a forced move
        5. action = the first index whose running total (cumulative sum) of probs is > u.
           If there is none (rounding), use the last legal index.
        6. record Decision(features, mask, action, float(log_probs[action]), view.player_id)
        7. return decode_play_action(action, view)

    choose_play, eval mode (training=False):
        the legal index with the highest logit (ties -> lowest index). No rng, no recording.

The bot only ever calls net.forward(x), so the tests sometimes use a fake net
that returns chosen logits.
"""
import math
import random

import numpy as np
import pytest

lb = pytest.importorskip("rl.learning_bot", reason="rl/learning_bot.py not written yet")
Decision = lb.Decision
LearningPlayBot = lb.LearningPlayBot

from rl.policy_net import PolicyMLP, masked_log_softmax
from bots.composite_bot import CompositeBot
from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from encoding.canonical import card_to_index
from encoding.play_encoder import encode_play_view
from engine.game import play_hand
from evaluation.match import play_match, seat_callbacks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_net(seed=0):
    return PolicyMLP(261, (32,), 24, random.Random(seed))


class FakeNet:
    """forward() always returns the same logits. Records how often it was called."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float64)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.logits.copy(), None


class FixedRng:
    """random() returns the given values in order. Anything else fails the test."""

    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.values.pop(0)

    def __getattr__(self, name):
        raise AssertionError(f"LearningPlayBot should only call rng.random(), not rng.{name}")


class CountingRng(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.random_calls = 0

    def random(self):
        self.random_calls += 1
        return super().random()


def collect_views(n_hands=40, seed=0):
    """Real PlayerViews from real hands (heuristic everywhere)."""
    views = []
    h = HeuristicBot()

    def play(view):
        views.append(view)
        return h.choose_play(view)

    rng = random.Random(seed)
    for i in range(n_hands):
        play_hand(i % 4, rng, h.choose_bid, h.choose_discard, play)
    return views


def view_with_legal(k_min):
    for v in collect_views():
        if len(v.legal_cards) >= k_min:
            return v
    raise RuntimeError("no view found")


def logits_for(mask, probs_by_legal):
    """Logits whose masked softmax gives exactly these probabilities on the legal slots (in index order)."""
    logits = np.full(24, 50.0)                 # illegal slots get a HIGH score: must be ignored
    legal = np.flatnonzero(mask)
    for idx, p in zip(legal, probs_by_legal):
        logits[idx] = math.log(p)
    return logits


def run_learner_hands(bot, n_hands, seed=0):
    """Learner plays team 0 (seats 0 and 2); heuristic-bid + random-play on team 1."""
    learner = CompositeBot(HeuristicBot(), bot)
    opponent = CompositeBot(HeuristicBot(), RandomBot(random.Random(seed + 100)))
    bid, discard, play = seat_callbacks(learner, opponent)
    rng = random.Random(seed)
    return [play_hand(i % 4, rng, bid, discard, play) for i in range(n_hands)]


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def test_decision_fields():
    d = Decision(features=np.zeros(261, dtype=np.float32), mask=np.ones(24, dtype=bool),
                 action=3, log_prob=-0.5, player_id=2)
    assert d.action == 3 and d.log_prob == -0.5 and d.player_id == 2
    assert d.features.shape == (261,) and d.mask.shape == (24,)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_stores_net_rng_and_training_flag():
    net, rng = make_net(), random.Random(0)
    bot = LearningPlayBot(net, rng)
    assert bot.net is net and bot.rng is rng
    assert bot.training is True
    assert LearningPlayBot(net, rng, training=False).training is False


def test_starts_with_no_decisions():
    assert LearningPlayBot(make_net(), random.Random(0)).pop_trajectory() == []


# ---------------------------------------------------------------------------
# training mode: what gets recorded
# ---------------------------------------------------------------------------

def test_returns_a_legal_card_and_records_one_decision():
    view = view_with_legal(2)
    bot = LearningPlayBot(make_net(), random.Random(0))
    card = bot.choose_play(view)
    assert card in view.legal_cards
    decisions = bot.pop_trajectory()
    assert len(decisions) == 1
    assert isinstance(decisions[0], Decision)


def test_recorded_decision_matches_the_view_and_the_card():
    view = view_with_legal(3)
    net = make_net(1)
    bot = LearningPlayBot(net, random.Random(1))
    card = bot.choose_play(view)
    d = bot.pop_trajectory()[0]

    features, mask = encode_play_view(view)
    np.testing.assert_array_equal(d.features, features)
    np.testing.assert_array_equal(d.mask, mask)
    assert d.action == card_to_index(card, view.trump)
    assert d.player_id == view.player_id
    assert isinstance(d.action, int)
    assert isinstance(d.log_prob, float)


def test_recorded_log_prob_is_the_nets_log_probability_of_that_card():
    view = view_with_legal(3)
    net = make_net(2)
    bot = LearningPlayBot(net, random.Random(2))
    bot.choose_play(view)
    d = bot.pop_trajectory()[0]
    logits, _ = net.forward(d.features)
    expected = masked_log_softmax(logits, d.mask)[d.action]
    assert d.log_prob == pytest.approx(expected, abs=1e-12)
    assert d.log_prob <= 0.0


def test_decisions_pile_up_until_popped_then_clear():
    views = [v for v in collect_views() if len(v.legal_cards) >= 1][:5]
    bot = LearningPlayBot(make_net(), random.Random(3))
    for v in views:
        bot.choose_play(v)
    decisions = bot.pop_trajectory()
    assert [d.player_id for d in decisions] == [v.player_id for v in views]   # oldest first
    assert bot.pop_trajectory() == []


def test_forced_move_is_recorded_with_log_prob_zero():
    view = next(v for v in collect_views() if len(v.legal_cards) == 1)
    bot = LearningPlayBot(make_net(), random.Random(4))
    assert bot.choose_play(view) == view.legal_cards[0]
    d = bot.pop_trajectory()[0]
    assert d.log_prob == 0.0


def test_exactly_one_rng_call_per_decision_even_when_forced():
    views = collect_views(n_hands=5)
    assert any(len(v.legal_cards) == 1 for v in views)
    rng = CountingRng(5)
    bot = LearningPlayBot(make_net(), rng)
    for v in views:
        bot.choose_play(v)
    assert rng.random_calls == len(views)


def test_records_both_seats_of_team_zero_in_a_real_hand():
    bot = LearningPlayBot(make_net(), random.Random(6))
    learner = CompositeBot(HeuristicBot(), bot)
    opponent = CompositeBot(HeuristicBot(), RandomBot(random.Random(106)))
    bid, discard, play = seat_callbacks(learner, opponent)
    rng = random.Random(6)
    for i in range(30):
        result = play_hand(i % 4, rng, bid, discard, play)
        decisions = bot.pop_trajectory()          # pop after EVERY hand, like the trainer will
        played_by_team0 = [(p, c) for trick in result.completed_tricks for (p, c) in trick if p % 2 == 0]
        assert [d.player_id for d in decisions] == [p for p, _ in played_by_team0]
        assert [d.action for d in decisions] == [card_to_index(c, result.trump) for _, c in played_by_team0]


# ---------------------------------------------------------------------------
# training mode: how the card is picked (the weighted die)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("u, which", [
    (0.0, 0), (0.1, 0), (0.19, 0),       # first legal card has 20%
    (0.21, 1), (0.5, 1), (0.69, 1),      # second has 50%  (running total 0.7)
    (0.71, 2), (0.99, 2),                # third has 30%   (running total 1.0)
])
def test_die_roll_picks_by_running_total(u, which):
    view = view_with_legal(3)
    _, mask = encode_play_view(view)
    legal = np.flatnonzero(mask)
    probs = [0.2, 0.5, 0.3] + [1e-12] * (len(legal) - 3)
    bot = LearningPlayBot(FakeNet(logits_for(mask, probs)), FixedRng(u))
    card = bot.choose_play(view)
    assert card_to_index(card, view.trump) == legal[which]


def test_die_roll_rounding_falls_back_to_last_legal_card():
    view = view_with_legal(3)
    _, mask = encode_play_view(view)
    legal = np.flatnonzero(mask)
    probs = [1.0 / len(legal)] * len(legal)
    bot = LearningPlayBot(FakeNet(logits_for(mask, probs)), FixedRng(1.0))   # u = 1.0: past every total
    card = bot.choose_play(view)
    assert card_to_index(card, view.trump) == legal[-1]
    assert card in view.legal_cards


def test_die_roll_frequencies_match_the_probabilities():
    view = view_with_legal(3)
    _, mask = encode_play_view(view)
    legal = np.flatnonzero(mask)
    probs = [0.2, 0.5, 0.3] + [1e-12] * (len(legal) - 3)
    bot = LearningPlayBot(FakeNet(logits_for(mask, probs)), random.Random(7))
    counts = {i: 0 for i in legal}
    for _ in range(4000):
        counts[card_to_index(bot.choose_play(view), view.trump)] += 1
    bot.pop_trajectory()
    for k, p in enumerate([0.2, 0.5, 0.3]):
        assert counts[legal[k]] / 4000 == pytest.approx(p, abs=0.03)


def test_illegal_cards_are_never_played_even_with_huge_scores():
    # logits_for gives every illegal slot a score of 50, far above the legal ones.
    view = view_with_legal(2)
    _, mask = encode_play_view(view)
    n = int(mask.sum())
    bot = LearningPlayBot(FakeNet(logits_for(mask, [1.0 / n] * n)), random.Random(8))
    for _ in range(200):
        assert bot.choose_play(view) in view.legal_cards


def test_same_seed_same_cards():
    a = LearningPlayBot(make_net(9), random.Random(9))
    b = LearningPlayBot(make_net(9), random.Random(9))
    run_learner_hands(a, 20, seed=9)
    run_learner_hands(b, 20, seed=9)
    da, db = a.pop_trajectory(), b.pop_trajectory()
    assert len(da) > 100
    assert [d.action for d in da] == [d.action for d in db]


def test_legal_over_many_real_hands():
    bot = LearningPlayBot(make_net(10), random.Random(10))
    results = run_learner_hands(bot, 500, seed=10)   # the engine raises on an illegal card
    assert len(results) == 500


# ---------------------------------------------------------------------------
# eval mode
# ---------------------------------------------------------------------------

def test_eval_mode_plays_the_highest_scoring_legal_card():
    view = view_with_legal(3)
    _, mask = encode_play_view(view)
    legal = np.flatnonzero(mask)
    logits = np.full(24, 99.0)             # illegal slots score highest: must be ignored
    logits[legal] = 0.0
    logits[legal[1]] = 2.0
    bot = LearningPlayBot(FakeNet(logits), FixedRng(), training=False)
    card = bot.choose_play(view)
    assert card_to_index(card, view.trump) == legal[1]


def test_eval_mode_ties_go_to_the_lowest_index():
    view = view_with_legal(3)
    _, mask = encode_play_view(view)
    legal = np.flatnonzero(mask)
    logits = np.full(24, 99.0)
    logits[legal] = 1.0
    bot = LearningPlayBot(FakeNet(logits), FixedRng(), training=False)
    assert card_to_index(bot.choose_play(view), view.trump) == legal[0]


def test_eval_mode_uses_no_rng_and_records_nothing():
    views = collect_views(n_hands=3)
    bot = LearningPlayBot(make_net(), FixedRng(), training=False)   # FixedRng() fails if random() is called
    for v in views:
        assert bot.choose_play(v) in v.legal_cards
    assert bot.pop_trajectory() == []


def test_can_switch_between_modes():
    view = view_with_legal(2)
    bot = LearningPlayBot(make_net(), random.Random(11))
    bot.training = False
    bot.choose_play(view)
    assert bot.pop_trajectory() == []
    bot.training = True
    bot.choose_play(view)
    assert len(bot.pop_trajectory()) == 1


def test_eval_mode_works_in_play_match():
    bot = LearningPlayBot(make_net(12), random.Random(12), training=False)
    learner = CompositeBot(HeuristicBot(), bot)
    result = play_match(learner, HeuristicBot(), n_pairs=5, seed=12)
    assert result.games == 10
    assert bot.pop_trajectory() == []


# ---------------------------------------------------------------------------
# the lesson, checked: an untrained net plays like RandomBot
# ---------------------------------------------------------------------------

def test_untrained_net_plays_about_as_well_as_random_play():
    bot = LearningPlayBot(make_net(13), random.Random(13))
    learner = CompositeBot(HeuristicBot(), bot)
    opponent = CompositeBot(HeuristicBot(), RandomBot(random.Random(14)))
    result = play_match(learner, opponent, n_pairs=150, seed=13)
    bot.pop_trajectory()
    assert 0.38 < result.win_rate < 0.62

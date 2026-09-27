"""
Tests for rl/reinforce.py (Phase 3, step 3e).

Part 1: the reward and the baseline
    hand_reward(result, team, trick_weight=0.0) -> float
        = result.scores[team] - result.scores[1 - team]
          + trick_weight * (team's tricks - opponents' tricks)
        team's tricks = sum of result.tricks_won[p] for every player p with p % 2 == team

    EMABaseline(decay=0.99)
        .decay
        .value            starts at 0.0
        .update(reward)   value = decay * value + (1 - decay) * reward

Part 2: the REINFORCE nudge
    reinforce_grad(net, decisions, advantages, n_hands) -> (loss: float, grads: list)
        decisions:  list of Decision (from LearningPlayBot)
        advantages: one number per decision (reward - baseline for that decision's hand)
        n_hands:    how many hands the decisions came from
        loss  = -sum(advantage_i * log pi(action_i | features_i)) / n_hands
        grads = net.backward(cache, dlogits), with
                dlogits_i = advantage_i * (probs_i - onehot(action_i)) / n_hands
        ValueError if a stored log_prob doesn't match the net's current one (atol 1e-6),
                   if len(decisions) != len(advantages), or if n_hands < 1
        No decisions -> (0.0, all-zero grads)
        Does not change the net's weights or the decisions.
"""
import random
from types import SimpleNamespace

import numpy as np
import pytest

rf = pytest.importorskip("rl.reinforce", reason="rl/reinforce.py not written yet")
hand_reward = rf.hand_reward
EMABaseline = rf.EMABaseline

from rl.policy_net import PolicyMLP, masked_log_softmax
from rl.learning_bot import Decision, LearningPlayBot
from rl.optim import Adam, SGD
from bots.composite_bot import CompositeBot
from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from encoding.play_encoder import encode_play_view
from engine.game import play_hand
from evaluation.match import seat_callbacks


def fake_result(scores, tricks_won):
    # hand_reward only needs .scores and .tricks_won
    return SimpleNamespace(scores=scores, tricks_won=tricks_won)


# ===========================================================================
# Part 1a: hand_reward
# ===========================================================================

# (scores, tricks_won by player, reward for team 0 with trick_weight 0) -- decision 2's table
REWARD_TABLE = [
    ({0: 1, 1: 0}, {0: 2, 1: 1, 2: 1, 3: 1}, 1),     # we called and made it (3 tricks)
    ({0: 1, 1: 0}, {0: 2, 1: 0, 2: 2, 3: 1}, 1),     # made it with 4 tricks: still 1
    ({0: 2, 1: 0}, {0: 3, 1: 0, 2: 2, 3: 0}, 2),     # we marched
    ({0: 4, 1: 0}, {0: 5, 1: 0, 2: 0, 3: 0}, 4),     # went alone and marched (partner sat out)
    ({0: 2, 1: 0}, {0: 2, 1: 1, 2: 1, 3: 1}, 2),     # we euchred them (they called, took 2)
    ({0: 0, 1: 1}, {0: 1, 1: 2, 2: 1, 3: 1}, -1),    # they made it
    ({0: 0, 1: 2}, {0: 0, 1: 3, 2: 0, 3: 2}, -2),    # they marched
    ({0: 0, 1: 2}, {0: 1, 1: 2, 2: 1, 3: 1}, -2),    # they euchred us (we called, took only 2)
    ({0: 0, 1: 4}, {0: 0, 1: 5, 2: 0, 3: 0}, -4),    # they went alone and marched
]


@pytest.mark.parametrize("scores, tricks, expected", REWARD_TABLE)
def test_reward_table_for_team_zero(scores, tricks, expected):
    r = hand_reward(fake_result(scores, tricks), team=0)
    assert r == expected
    assert isinstance(r, float)


@pytest.mark.parametrize("scores, tricks, expected", REWARD_TABLE)
def test_other_team_gets_the_opposite_reward(scores, tricks, expected):
    assert hand_reward(fake_result(scores, tricks), team=1) == -expected


def test_trick_weight_adds_the_trick_difference():
    # Made it 3 tricks to 2: 1 point + 0.1 * (3 - 2)
    result = fake_result({0: 1, 1: 0}, {0: 2, 1: 1, 2: 1, 3: 1})
    assert hand_reward(result, 0, trick_weight=0.1) == pytest.approx(1.1)
    assert hand_reward(result, 1, trick_weight=0.1) == pytest.approx(-1.1)


def test_trick_weight_example_from_the_lessons():
    # Defending: they made it 3-2.  -1 point + 0.1 * (2 - 3) = -1.1
    result = fake_result({0: 0, 1: 1}, {0: 1, 1: 2, 2: 1, 3: 1})
    assert hand_reward(result, 0, trick_weight=0.1) == pytest.approx(-1.1)


def test_trick_weight_on_a_march():
    # Marched 5-0: 2 points + 0.1 * 5
    result = fake_result({0: 2, 1: 0}, {0: 3, 1: 0, 2: 2, 3: 0})
    assert hand_reward(result, 0, trick_weight=0.1) == pytest.approx(2.5)


def test_default_trick_weight_is_zero():
    result = fake_result({0: 1, 1: 0}, {0: 2, 1: 0, 2: 2, 3: 1})
    assert hand_reward(result, 0) == hand_reward(result, 0, trick_weight=0.0) == 1.0


def test_real_hands_give_only_the_six_possible_rewards():
    h = HeuristicBot()
    r = RandomBot(random.Random(1))
    bid, discard, play = seat_callbacks(CompositeBot(h, r), h)
    rng = random.Random(0)
    seen = set()
    for i in range(300):
        result = play_hand(i % 4, rng, bid, discard, play)
        reward = hand_reward(result, 0)
        assert reward in {-4.0, -2.0, -1.0, 1.0, 2.0, 4.0}
        assert hand_reward(result, 1) == -reward
        seen.add(reward)
    assert {-2.0, -1.0, 1.0, 2.0} <= seen


# ===========================================================================
# Part 1b: EMABaseline
# ===========================================================================

def test_baseline_starts_at_zero_with_default_decay():
    b = EMABaseline()
    assert b.value == 0.0
    assert b.decay == 0.99


def test_baseline_update_matches_hand_computation():
    b = EMABaseline(decay=0.9)
    b.update(1.0)
    assert b.value == pytest.approx(0.1)            # 0.9 * 0 + 0.1 * 1
    b.update(1.0)
    assert b.value == pytest.approx(0.19)           # 0.9 * 0.1 + 0.1 * 1
    b.update(-2.0)
    assert b.value == pytest.approx(-0.029)         # 0.9 * 0.19 + 0.1 * (-2)


def test_baseline_settles_on_the_average_reward():
    b = EMABaseline(decay=0.99)
    for _ in range(3000):
        b.update(3.0)
    assert b.value == pytest.approx(3.0, abs=1e-6)


def test_baseline_tracks_a_noisy_average():
    g = np.random.default_rng(0)
    b = EMABaseline(decay=0.99)
    for _ in range(5000):
        b.update(float(g.choice([-2.0, -1.0, 1.0, 2.0], p=[0.1, 0.3, 0.4, 0.2])))
    # true mean = -0.2 - 0.3 + 0.4 + 0.4 = 0.3
    assert b.value == pytest.approx(0.3, abs=0.35)


# ===========================================================================
# Part 2: reinforce_grad
# ===========================================================================

def small_net(seed=0, hidden=(16,)):
    return PolicyMLP(261, hidden, 24, random.Random(seed))


def real_decisions(net, n_hands=6, seed=0):
    """Decisions recorded by a LearningPlayBot in real hands; returns (decisions, hand_of_each)."""
    bot = LearningPlayBot(net, random.Random(seed))
    learner = CompositeBot(HeuristicBot(), bot)
    opponent = CompositeBot(HeuristicBot(), RandomBot(random.Random(seed + 1)))
    bid, discard, play = seat_callbacks(learner, opponent)
    rng = random.Random(seed + 2)
    decisions, hand_of = [], []
    for i in range(n_hands):
        play_hand(i % 4, rng, bid, discard, play)
        ds = bot.pop_trajectory()
        decisions += ds
        hand_of += [i] * len(ds)
    return decisions, hand_of


def reference_loss(net, decisions, advantages, n_hands):
    total = 0.0
    for d, a in zip(decisions, advantages):
        logits, _ = net.forward(d.features)
        total += a * masked_log_softmax(logits, d.mask)[d.action]
    return -total / n_hands


def test_returns_float_loss_and_grads_shaped_like_params():
    net = small_net()
    decisions, _ = real_decisions(net)
    adv = [1.0] * len(decisions)
    loss, grads = rf.reinforce_grad(net, decisions, adv, n_hands=6)
    assert isinstance(loss, float)
    assert len(grads) == len(net.params())
    for g, p in zip(grads, net.params()):
        assert g.shape == p.shape


def test_loss_matches_the_formula():
    net = small_net(1)
    decisions, hand_of = real_decisions(net, seed=1)
    g = np.random.default_rng(1)
    adv = list(g.normal(0, 1.5, len(decisions)))
    loss, _ = rf.reinforce_grad(net, decisions, adv, n_hands=6)
    assert loss == pytest.approx(reference_loss(net, decisions, adv, 6), abs=1e-9)


def test_gradients_match_nudging_every_weight():
    # Finite-difference check of the whole REINFORCE loss (spot-checks 25 numbers per param).
    net = small_net(2, hidden=(8,))
    decisions, _ = real_decisions(net, n_hands=4, seed=2)
    g = np.random.default_rng(2)
    adv = list(g.normal(0, 1, len(decisions)))
    _, grads = rf.reinforce_grad(net, decisions, adv, n_hands=4)
    eps = 1e-6
    for param, grad in zip(net.params(), grads):
        idxs = list(np.ndindex(param.shape))
        picks = g.choice(len(idxs), size=min(25, len(idxs)), replace=False)
        for k in picks:
            i = idxs[k]
            old = param[i]
            param[i] = old + eps
            up = reference_loss(net, decisions, adv, 4)
            param[i] = old - eps
            down = reference_loss(net, decisions, adv, 4)
            param[i] = old
            assert grad[i] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-8)


def test_zero_advantages_give_zero_loss_and_gradients():
    net = small_net(3)
    decisions, _ = real_decisions(net, seed=3)
    loss, grads = rf.reinforce_grad(net, decisions, [0.0] * len(decisions), n_hands=6)
    assert loss == 0.0
    for gr in grads:
        assert np.all(gr == 0.0)


def test_forced_moves_give_zero_gradients():
    # Lesson step 2: one legal card -> probability 1 -> nothing to learn.
    net = small_net(4)
    decisions, _ = real_decisions(net, n_hands=10, seed=4)
    forced = [d for d in decisions if d.mask.sum() == 1]
    assert len(forced) >= 5
    loss, grads = rf.reinforce_grad(net, forced, [2.0] * len(forced), n_hands=10)
    assert loss == pytest.approx(0.0, abs=1e-12)
    for gr in grads:
        np.testing.assert_allclose(gr, 0.0, atol=1e-12)


def test_illegal_cards_get_no_gradient():
    # Every decision has the same mask -> the output bias gradient is 0 at illegal slots.
    net = small_net(5)
    decisions, _ = real_decisions(net, n_hands=10, seed=5)
    mask = decisions[0].mask
    same = [d for d in decisions if np.array_equal(d.mask, mask)]
    _, grads = rf.reinforce_grad(net, same, [1.0] * len(same), n_hands=1)
    assert np.all(grads[-1][~mask] == 0.0)


def test_dividing_by_n_hands():
    net = small_net(6)
    decisions, _ = real_decisions(net, seed=6)
    adv = [1.0] * len(decisions)
    loss1, grads1 = rf.reinforce_grad(net, decisions, adv, n_hands=3)
    loss2, grads2 = rf.reinforce_grad(net, decisions, adv, n_hands=6)
    assert loss2 == pytest.approx(loss1 / 2)
    for a, b in zip(grads1, grads2):
        np.testing.assert_allclose(b, a / 2, atol=1e-12)


@pytest.mark.parametrize("advantage, direction", [(+1.0, +1), (-1.0, -1)])
def test_one_small_step_moves_the_played_card_the_right_way(advantage, direction):
    # Lesson step 3: good result -> that card more likely; bad result -> less likely.
    net = small_net(7)
    decisions, _ = real_decisions(net, n_hands=3, seed=7)
    d = next(d for d in decisions if d.mask.sum() >= 2)
    before = d.log_prob
    _, grads = rf.reinforce_grad(net, [d], [advantage], n_hands=1)
    SGD(net.params(), lr=0.01).step(grads)
    logits, _ = net.forward(d.features)
    after = masked_log_softmax(logits, d.mask)[d.action]
    assert (after - before) * direction > 0


def test_stale_log_prob_raises():
    # Decision #37: the stored log_prob must match what the net says now.
    net = small_net(8)
    decisions, _ = real_decisions(net, seed=8)
    d = next(d for d in decisions if d.mask.sum() >= 2)
    bad = Decision(d.features, d.mask, d.action, d.log_prob - 0.1, d.player_id)
    with pytest.raises(ValueError):
        rf.reinforce_grad(net, decisions + [bad], [1.0] * (len(decisions) + 1), n_hands=6)


def test_weights_changed_after_recording_raises():
    net = small_net(9)
    decisions, _ = real_decisions(net, seed=9)
    net.params()[-1][:] += 0.5 * np.arange(24)      # change the output biases after recording
    with pytest.raises(ValueError):
        rf.reinforce_grad(net, decisions, [1.0] * len(decisions), n_hands=6)


def test_length_mismatch_raises():
    net = small_net(10)
    decisions, _ = real_decisions(net, seed=10)
    with pytest.raises(ValueError):
        rf.reinforce_grad(net, decisions, [1.0] * (len(decisions) - 1), n_hands=6)


def test_n_hands_must_be_positive():
    net = small_net(11)
    decisions, _ = real_decisions(net, seed=11)
    with pytest.raises(ValueError):
        rf.reinforce_grad(net, decisions, [1.0] * len(decisions), n_hands=0)


def test_no_decisions_gives_zero():
    net = small_net(12)
    loss, grads = rf.reinforce_grad(net, [], [], n_hands=1)
    assert loss == 0.0
    assert len(grads) == len(net.params())
    for g, p in zip(grads, net.params()):
        assert g.shape == p.shape and np.all(g == 0.0)


def test_does_not_change_weights_or_decisions():
    net = small_net(13)
    decisions, _ = real_decisions(net, seed=13)
    before_w = [p.copy() for p in net.params()]
    before_d = [(d.features.copy(), d.mask.copy(), d.action, d.log_prob) for d in decisions]
    rf.reinforce_grad(net, decisions, [1.0] * len(decisions), n_hands=6)
    for p, b in zip(net.params(), before_w):
        np.testing.assert_array_equal(p, b)
    for d, (f, m, a, lp) in zip(decisions, before_d):
        np.testing.assert_array_equal(d.features, f)
        np.testing.assert_array_equal(d.mask, m)
        assert d.action == a and d.log_prob == lp


# ===========================================================================
# Everything together: the "bandit" test
# ===========================================================================

def test_bandit_learns_to_play_the_rewarded_card():
    """
    One fixed situation with several legal cards. Reward 1 if the net plays card k, else 0.
    Play it, record, compute the nudge with a baseline, Adam step, repeat.
    The net should end up playing card k almost always.
    """
    h = HeuristicBot()
    views = []

    def grab(view):
        views.append(view)
        return h.choose_play(view)

    rng = random.Random(0)
    while not any(len(v.legal_cards) >= 4 for v in views):
        play_hand(0, rng, h.choose_bid, h.choose_discard, grab)
    view = next(v for v in views if len(v.legal_cards) >= 4)
    _, mask = encode_play_view(view)
    k = int(np.flatnonzero(mask)[2])               # reward the 3rd legal card, not the 1st

    net = PolicyMLP(261, (32,), 24, random.Random(0))
    bot = LearningPlayBot(net, random.Random(1))
    opt = Adam(net.params(), lr=0.01)
    baseline = EMABaseline(decay=0.9)

    for _ in range(60):
        rewards = []
        for _ in range(16):                        # 16 "hands" of one decision each
            bot.choose_play(view)
        decisions = bot.pop_trajectory()
        for d in decisions:
            rewards.append(1.0 if d.action == k else 0.0)
        advantages = [r - baseline.value for r in rewards]
        for r in rewards:
            baseline.update(r)
        _, grads = rf.reinforce_grad(net, decisions, advantages, n_hands=len(decisions))
        opt.step(grads)

    features, mask = encode_play_view(view)
    logits, _ = net.forward(features)
    p = np.exp(masked_log_softmax(logits, mask))
    assert p[k] > 0.9

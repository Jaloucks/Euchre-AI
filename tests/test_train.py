"""
Tests for rl/train.py (Phase 3, step 3f): the training loop.

    @dataclass
    class TrainConfig:
        hands=100_000; batch_hands=64; lr=1e-3; hidden=(128, 128)
        trick_weight=0.0; baseline="ema"; ema_decay=0.99      # baseline: "ema" or "none"
        opponent="random-play"                                 # any name make_opponent accepts
        eval_every=10_000; eval_pairs=200; seed=0; csv_path=None

    train(config) -> (net, history)
        net:     the trained PolicyMLP (261 -> hidden -> 24)
        history: one dict per evaluation, keys "hands", "avg_reward", "win_rate", "ci"
                 - first row is BEFORE any training: hands=0, avg_reward=None
                 - then one row every eval_every hands
                 - avg_reward = mean hand_reward over the hands since the previous row
                 - win_rate / ci come from play_match(learner, opponent, eval_pairs, eval_seed)
                   in EVAL mode, with the SAME eval_seed every time
        Every batch_hands hands: one reinforce_grad call (n_hands=batch_hands) + one Adam step.
        Each decision's advantage = that hand's reward - baseline value BEFORE updating with that hand
        (baseline "none" -> advantage = reward).
        ValueError for an unknown baseline or opponent.
        If csv_path is set, the CSV (header: hands,avg_reward,win_rate,ci) is rewritten after every evaluation.

    main(argv=None) -> 0
        --hands --batch-hands --lr --hidden (any number of ints) --trick-weight --baseline
        --ema-decay --opponent --eval-every --eval-pairs --seed --csv
        builds a TrainConfig, calls train(config), prints one line per history row.

train.py must import these by name and call them by that name, because some tests
replace them to watch what the loop does:
    from rl.reinforce import reinforce_grad, hand_reward
    from evaluation.match import play_match
"""
import csv
import random

import numpy as np
import pytest

tr = pytest.importorskip("rl.train", reason="rl/train.py not written yet")
TrainConfig = tr.TrainConfig
train = tr.train

from rl.policy_net import PolicyMLP
import rl.reinforce as reinforce_module


def tiny(**overrides):
    """A config small enough to run in well under a second."""
    base = dict(hands=256, batch_hands=64, hidden=(16,), eval_every=128, eval_pairs=3, seed=0)
    base.update(overrides)
    return TrainConfig(**base)


@pytest.fixture
def spy(monkeypatch):
    """Replace rl.train.reinforce_grad with a wrapper that records every call."""
    calls = []
    real = reinforce_module.reinforce_grad

    def wrapper(net, decisions, advantages, n_hands):
        calls.append({"decisions": list(decisions), "advantages": list(advantages), "n_hands": n_hands})
        return real(net, decisions, advantages, n_hands)

    monkeypatch.setattr(tr, "reinforce_grad", wrapper)
    return calls


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

def test_config_defaults():
    c = TrainConfig()
    assert c.hands == 100_000
    assert c.batch_hands == 64
    assert c.lr == 1e-3
    assert tuple(c.hidden) == (128, 128)
    assert c.trick_weight == 0.0
    assert c.baseline == "ema"
    assert c.ema_decay == 0.99
    assert c.opponent == "random-play"
    assert c.eval_every == 10_000
    assert c.eval_pairs == 200
    assert c.seed == 0
    assert c.csv_path is None


# ---------------------------------------------------------------------------
# train(): what comes back
# ---------------------------------------------------------------------------

def test_returns_a_policy_net_with_the_configured_shape():
    net, _ = train(tiny(hidden=(12, 8)))
    assert isinstance(net, PolicyMLP)
    assert [W.shape for W in net.weights] == [(261, 12), (12, 8), (8, 24)]


def test_history_has_a_row_before_training_and_one_per_eval():
    _, history = train(tiny())
    assert [row["hands"] for row in history] == [0, 128, 256]
    for row in history:
        assert set(row) == {"hands", "avg_reward", "win_rate", "ci"}


def test_first_row_is_before_any_training():
    _, history = train(tiny())
    assert history[0]["hands"] == 0
    assert history[0]["avg_reward"] is None


def test_row_values_are_sensible():
    _, history = train(tiny())
    for row in history:
        assert 0.0 <= row["win_rate"] <= 1.0
        assert row["ci"] >= 0.0
    for row in history[1:]:
        assert -4.0 <= row["avg_reward"] <= 4.0


def test_zero_hands_means_no_training():
    net0, history = train(tiny(hands=0))
    assert [row["hands"] for row in history] == [0]
    net1, _ = train(tiny(hands=128))
    assert any(not np.array_equal(a, b) for a, b in zip(net0.params(), net1.params()))


def test_same_config_same_result():
    net_a, hist_a = train(tiny())
    net_b, hist_b = train(tiny())
    assert hist_a == hist_b
    for a, b in zip(net_a.params(), net_b.params()):
        np.testing.assert_array_equal(a, b)


def test_different_seed_different_result():
    net_a, _ = train(tiny(seed=0))
    net_b, _ = train(tiny(seed=1))
    assert not np.array_equal(net_a.weights[0], net_b.weights[0])


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        train(tiny(baseline="median"))


def test_unknown_opponent_raises():
    with pytest.raises(ValueError):
        train(tiny(opponent="grandmaster"))


def test_heuristic_opponent_runs():
    _, history = train(tiny(opponent="heuristic", hands=128))
    assert len(history) == 2


# ---------------------------------------------------------------------------
# train(): what the loop sends to reinforce_grad
# ---------------------------------------------------------------------------

def test_one_update_per_batch(spy):
    train(tiny(hands=256, batch_hands=64))
    assert len(spy) == 4
    assert all(call["n_hands"] == 64 for call in spy)


def test_one_advantage_per_decision_and_only_team_zero(spy):
    train(tiny())
    for call in spy:
        assert len(call["advantages"]) == len(call["decisions"]) > 0
        assert all(d.player_id in (0, 2) for d in call["decisions"])


def test_eval_games_are_not_recorded(spy):
    # A team plays at most 10 cards per hand. If eval games leaked into the
    # notebook, the next batch would hold thousands of decisions.
    train(tiny(hands=256, batch_hands=64, eval_every=64, eval_pairs=5))
    for call in spy:
        assert len(call["decisions"]) <= 10 * 64


def test_no_baseline_means_advantage_is_the_raw_reward(spy):
    train(tiny(baseline="none"))
    values = {a for call in spy for a in call["advantages"]}
    assert values <= {-4.0, -2.0, -1.0, 1.0, 2.0, 4.0}


def test_ema_baseline_shifts_the_advantages(spy):
    train(tiny(baseline="ema"))
    values = {a for call in spy for a in call["advantages"]}
    assert not values <= {-4.0, -2.0, -1.0, 1.0, 2.0, 4.0}


def test_first_hand_uses_baseline_zero_then_ema_updates(spy):
    # Hand 1: baseline starts at 0, so its advantage is the raw reward.
    # Hand 2: baseline = (1 - decay) * reward_1, so advantage_2 = reward_2 - that.
    train(tiny(baseline="ema", ema_decay=0.5, hands=64))
    call = spy[0]
    decisions, adv = call["decisions"], call["advantages"]
    # decisions of hand 1 are the first ones; all share one advantage value.
    first = adv[0]
    assert first in {-4.0, -2.0, -1.0, 1.0, 2.0, 4.0}
    k = 1
    while adv[k] == first and k < len(adv) - 1:
        k += 1
    second = adv[k]
    # second = reward_2 - 0.5 * reward_1, and reward_2 is one of the six values
    assert (second + 0.5 * first) in {-4.0, -2.0, -1.0, 1.0, 2.0, 4.0}


def test_trick_weight_reaches_the_rewards(spy):
    train(tiny(baseline="none", trick_weight=0.1))
    values = {round(a, 6) for call in spy for a in call["advantages"]}
    assert not values <= {-4.0, -2.0, -1.0, 1.0, 2.0, 4.0}


def test_decisions_are_never_stale(spy):
    # reinforce_grad raises on a stale log_prob; reaching the end means the order
    # (record -> update -> record with the new weights) is right.
    train(tiny(hands=320, batch_hands=32))
    assert len(spy) == 10


def test_avg_reward_is_the_mean_over_each_window(monkeypatch):
    rewards = []
    real = reinforce_module.hand_reward

    def recording(result, team, trick_weight=0.0):
        r = real(result, team, trick_weight)
        rewards.append(r)
        return r

    monkeypatch.setattr(tr, "hand_reward", recording)
    _, history = train(tiny(hands=256, eval_every=128))
    assert len(rewards) == 256
    assert history[1]["avg_reward"] == pytest.approx(np.mean(rewards[:128]))
    assert history[2]["avg_reward"] == pytest.approx(np.mean(rewards[128:256]))


def test_every_evaluation_uses_the_same_deals_in_eval_mode(monkeypatch):
    # Same eval seed each time -> every point on the curve is measured on the same deals.
    import evaluation.match as match_module
    seen = []

    def recording(bot_a, bot_b, n_pairs, seed, target=10):
        seen.append((n_pairs, seed, bot_a.player.training))
        return match_module.play_match(bot_a, bot_b, n_pairs, seed, target)

    monkeypatch.setattr(tr, "play_match", recording)
    train(tiny(hands=256, eval_every=64, eval_pairs=3))
    assert len(seen) == 5
    assert {n for n, _, _ in seen} == {3}
    assert len({seed for _, seed, _ in seen}) == 1
    assert not any(training for _, _, training in seen)     # the net plays its best card, no dice


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def test_csv_is_written(tmp_path):
    path = tmp_path / "curve.csv"
    _, history = train(tiny(csv_path=str(path)))
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["hands", "avg_reward", "win_rate", "ci"]
    assert [int(r["hands"]) for r in rows] == [row["hands"] for row in history]
    assert rows[0]["avg_reward"] == ""                     # None is written as an empty cell
    for r, row in zip(rows[1:], history[1:]):
        assert float(r["win_rate"]) == pytest.approx(row["win_rate"])
        assert float(r["avg_reward"]) == pytest.approx(row["avg_reward"])


def test_no_csv_path_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train(tiny())
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# main(): the command line
# ---------------------------------------------------------------------------

def test_main_builds_the_config_from_the_flags(monkeypatch):
    seen = {}

    def fake_train(config):
        seen["config"] = config
        return None, [{"hands": 0, "avg_reward": None, "win_rate": 0.5, "ci": 0.1}]

    monkeypatch.setattr(tr, "train", fake_train)
    code = tr.main([
        "--hands", "500", "--batch-hands", "32", "--lr", "0.002", "--hidden", "64", "32", "16",
        "--trick-weight", "0.1", "--baseline", "none", "--ema-decay", "0.9",
        "--opponent", "heuristic", "--eval-every", "100", "--eval-pairs", "7",
        "--seed", "5", "--csv", "out.csv",
    ])
    assert code == 0
    c = seen["config"]
    assert c.hands == 500 and c.batch_hands == 32 and c.lr == 0.002
    assert tuple(c.hidden) == (64, 32, 16)
    assert c.trick_weight == 0.1 and c.baseline == "none" and c.ema_decay == 0.9
    assert c.opponent == "heuristic" and c.eval_every == 100 and c.eval_pairs == 7
    assert c.seed == 5 and c.csv_path == "out.csv"


def test_main_defaults_match_train_config(monkeypatch):
    seen = {}

    def fake_train(config):
        seen["config"] = config
        return None, [{"hands": 0, "avg_reward": None, "win_rate": 0.5, "ci": 0.1}]

    monkeypatch.setattr(tr, "train", fake_train)
    assert tr.main([]) == 0
    c, d = seen["config"], TrainConfig()
    assert (c.hands, c.batch_hands, c.lr, tuple(c.hidden), c.trick_weight, c.baseline,
            c.ema_decay, c.opponent, c.eval_every, c.eval_pairs, c.seed, c.csv_path) == \
           (d.hands, d.batch_hands, d.lr, tuple(d.hidden), d.trick_weight, d.baseline,
            d.ema_decay, d.opponent, d.eval_every, d.eval_pairs, d.seed, d.csv_path)


def test_main_runs_for_real_and_prints_each_row(capsys, tmp_path):
    path = tmp_path / "run.csv"
    code = tr.main(["--hands", "128", "--eval-every", "64", "--eval-pairs", "2",
                    "--hidden", "8", "--csv", str(path)])
    assert code == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) >= 3            # rows for hands 0, 64, 128
    assert path.exists()


def test_main_rejects_unknown_baseline():
    with pytest.raises(SystemExit):
        tr.main(["--baseline", "median"])

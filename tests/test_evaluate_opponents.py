"""
Tests for the --opponent option and make_opponent() in evaluation/evaluate.py
(Phase 3, step 3d part 1).

    OPPONENTS = ("random", "random-play", "heuristic")
    make_opponent(name, rng) -> bot
        "random"      -> RandomBot(rng)
        "random-play" -> CompositeBot(HeuristicBot(), RandomBot(rng))   # the Phase 3 yardstick
        "heuristic"   -> HeuristicBot()
        anything else -> ValueError

    main(argv) accepts --opponent (default "random", so the Phase 2 behavior is unchanged).
    The first printed line names the opponent.
"""
import random

import pytest

evaluate = pytest.importorskip("evaluation.evaluate")
pytest.importorskip("bots.composite_bot", reason="bots/composite_bot.py not written yet")

from bots.composite_bot import CompositeBot
from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot


def test_opponent_names():
    assert tuple(evaluate.OPPONENTS) == ("random", "random-play", "heuristic")


def test_make_random():
    rng = random.Random(0)
    bot = evaluate.make_opponent("random", rng)
    assert isinstance(bot, RandomBot)
    assert bot.rng is rng


def test_make_random_play():
    rng = random.Random(0)
    bot = evaluate.make_opponent("random-play", rng)
    assert isinstance(bot, CompositeBot)
    assert isinstance(bot.bidder, HeuristicBot)
    assert isinstance(bot.player, RandomBot)
    assert bot.player.rng is rng


def test_make_heuristic():
    assert isinstance(evaluate.make_opponent("heuristic", random.Random(0)), HeuristicBot)


def test_unknown_opponent_raises():
    with pytest.raises(ValueError):
        evaluate.make_opponent("grandmaster", random.Random(0))


@pytest.mark.parametrize("name", ["random", "random-play", "heuristic"])
def test_main_runs_with_each_opponent(name, capsys):
    assert evaluate.main(["--pairs", "3", "--seed", "1", "--opponent", name]) == 0
    out = capsys.readouterr().out
    assert name in out.splitlines()[0]
    assert "win rate" in out


def test_main_default_opponent_is_random(capsys):
    assert evaluate.main(["--pairs", "3", "--seed", "1"]) == 0
    assert "random" in capsys.readouterr().out.splitlines()[0]


def test_main_rejects_unknown_opponent():
    with pytest.raises(SystemExit):
        evaluate.main(["--pairs", "3", "--opponent", "grandmaster"])


def test_heuristic_mirror_is_exactly_even(capsys):
    evaluate.main(["--pairs", "5", "--opponent", "heuristic"])
    assert "0.500" in capsys.readouterr().out

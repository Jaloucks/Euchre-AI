"""
Tests for evaluation/match.py and evaluation/evaluate.py (Phase 2b).

Spec summary:

    @dataclass
    class MatchResult:
        games: int          # total games played (= 2 * n_pairs)
        wins_a: int         # games won by bot A's team
        points_a: int       # total points scored by A's team, summed over games
        points_b: int
        win_rate -> float   (property) wins_a / games
        std_error -> float  (property) sqrt(p * (1 - p) / games)

    seat_callbacks(team0_bot, team1_bot) -> (choose_bid_fn, choose_discard_fn, choose_play_fn)
        Routes each decision to team0_bot when view.player_id % 2 == 0, else team1_bot.

    play_match(bot_a, bot_b, n_pairs, seed, target=10) -> MatchResult
        Duplicate format. For each pair:
          - draw one game seed from random.Random(seed)
          - play game 1 with A as team 0, game 2 with A as team 1,
            both with random.Random(game_seed) and the same first_dealer
        first_dealer rotates across pairs so every seat deals first equally.
        Calls engine.game.play_game through the name `play_game` imported into
        evaluation.match (so tests can substitute a fake).
        n_pairs < 1 -> ValueError.

    evaluation/evaluate.py:
        main(argv=None) -> int
            --pairs N (default 500), --seed S (default 0)
            Runs HeuristicBot (A) vs RandomBot (B), prints a report that includes
            the words "win rate", returns 0.
"""
import math
import random

import pytest

match = pytest.importorskip("evaluation.match", reason="evaluation/match.py not written yet")

from engine.game import GameResult, play_hand
from bots.random_bot import RandomBot

MatchResult = match.MatchResult
seat_callbacks = match.seat_callbacks
play_match = match.play_match


# ---------- MatchResult ----------

def test_match_result_win_rate_and_std_error():
    r = MatchResult(games=100, wins_a=75, points_a=900, points_b=500)
    assert r.win_rate == 0.75
    assert r.std_error == pytest.approx(math.sqrt(0.75 * 0.25 / 100))


def test_std_error_is_zero_for_a_sweep():
    assert MatchResult(games=10, wins_a=10, points_a=100, points_b=0).std_error == 0.0


# ---------- seat_callbacks ----------

class SeatRecorder(RandomBot):
    """A RandomBot that remembers which player_ids it was asked to decide for."""
    def __init__(self, rng):
        super().__init__(rng)
        self.seen = set()

    def choose_bid(self, view, legal):
        self.seen.add(view.player_id)
        return super().choose_bid(view, legal)

    def choose_discard(self, view):
        self.seen.add(view.player_id)
        return super().choose_discard(view)

    def choose_play(self, view):
        self.seen.add(view.player_id)
        return super().choose_play(view)


def test_seat_callbacks_route_by_team():
    rng = random.Random(4)
    a, b = SeatRecorder(rng), SeatRecorder(rng)
    fns = seat_callbacks(a, b)
    for h in range(50):
        play_hand(h % 4, rng, *fns)
    assert a.seen == {0, 2}
    assert b.seen == {1, 3}


# ---------- play_match with a fake play_game ----------

class Named:
    """Bot stub whose bid answer is just its name — lets the fake see who sits where."""
    def __init__(self, name):
        self.name = name

    def choose_bid(self, view, legal):
        return self.name

    def choose_discard(self, view):
        return self.name

    def choose_play(self, view):
        return self.name


class _View:
    def __init__(self, pid):
        self.player_id = pid


@pytest.fixture
def fake_games(monkeypatch):
    """Replace play_game inside evaluation.match; team 0 always wins 10-3."""
    calls = []

    def fake_play_game(rng, choose_bid_fn, choose_discard_fn, choose_play_fn,
                       target=10, first_dealer=0, max_hands=200):
        calls.append(dict(
            draw=rng.random(),
            first_dealer=first_dealer,
            target=target,
            seat0=choose_bid_fn(_View(0), []),
            seat1=choose_bid_fn(_View(1), []),
            play2=choose_play_fn(_View(2)),
            discard3=choose_discard_fn(_View(3)),
        ))
        return GameResult(winning_team=0, final_scores={0: 10, 1: 3}, hands=[])

    monkeypatch.setattr(match, "play_game", fake_play_game)
    return calls


def test_two_games_per_pair(fake_games):
    r = play_match(Named("A"), Named("B"), n_pairs=5, seed=0)
    assert r.games == 10
    assert len(fake_games) == 10


def test_pairs_swap_teams(fake_games):
    play_match(Named("A"), Named("B"), n_pairs=4, seed=0)
    for g1, g2 in zip(fake_games[0::2], fake_games[1::2]):
        assert (g1["seat0"], g1["seat1"]) == ("A", "B")
        assert (g2["seat0"], g2["seat1"]) == ("B", "A")
        assert g1["play2"] == g1["seat0"] and g1["discard3"] == g1["seat1"]
        assert g2["play2"] == g2["seat0"] and g2["discard3"] == g2["seat1"]


def test_pairs_share_deal_seed_and_first_dealer(fake_games):
    play_match(Named("A"), Named("B"), n_pairs=8, seed=3)
    firsts = []
    for g1, g2 in zip(fake_games[0::2], fake_games[1::2]):
        assert g1["draw"] == g2["draw"], "both games of a pair must see identical deals"
        assert g1["first_dealer"] == g2["first_dealer"]
        firsts.append(g1["first_dealer"])
    assert len({g["draw"] for g in fake_games[0::2]}) == 8, "each pair needs its own deals"
    assert sorted(firsts[:4]) == [0, 1, 2, 3]


def test_wins_and_points_credited_to_the_right_bot(fake_games):
    # Team 0 always wins 10-3, and A is team 0 in exactly one game per pair.
    r = play_match(Named("A"), Named("B"), n_pairs=6, seed=0)
    assert r.wins_a == 6
    assert r.points_a == 6 * (10 + 3)
    assert r.points_b == 6 * (3 + 10)


def test_target_is_passed_through(fake_games):
    play_match(Named("A"), Named("B"), n_pairs=1, seed=0, target=5)
    assert all(g["target"] == 5 for g in fake_games)


def test_seed_changes_deals(fake_games):
    play_match(Named("A"), Named("B"), n_pairs=1, seed=0)
    play_match(Named("A"), Named("B"), n_pairs=1, seed=1)
    assert fake_games[0]["draw"] != fake_games[2]["draw"]


@pytest.mark.parametrize("n", [0, -1])
def test_n_pairs_must_be_positive(n):
    with pytest.raises(ValueError):
        play_match(Named("A"), Named("B"), n_pairs=n, seed=0)


# ---------- real games ----------

def _heuristic_bot():
    hb = pytest.importorskip("bots.heuristic_bot", reason="bots/heuristic_bot.py not written yet")
    return hb.HeuristicBot()


def test_mirror_match_is_exactly_even():
    # A deterministic bot against itself: both games of a pair are the same
    # game with the labels swapped, so A wins exactly one of each pair.
    r = play_match(_heuristic_bot(), _heuristic_bot(), n_pairs=20, seed=0)
    assert r.wins_a == 20
    assert r.points_a == r.points_b


def test_match_is_reproducible():
    r1 = play_match(_heuristic_bot(), RandomBot(random.Random(3)), n_pairs=10, seed=42)
    r2 = play_match(_heuristic_bot(), RandomBot(random.Random(3)), n_pairs=10, seed=42)
    assert r1 == r2


def test_heuristic_clearly_beats_random():
    r = play_match(_heuristic_bot(), RandomBot(random.Random(0)), n_pairs=100, seed=0)
    assert r.games == 200
    assert r.win_rate >= 0.90
    assert r.points_a > 3 * r.points_b


# ---------- CLI ----------

def test_evaluate_main_prints_report(capsys):
    evaluate = pytest.importorskip("evaluation.evaluate", reason="evaluation/evaluate.py not written yet")
    assert evaluate.main(["--pairs", "3", "--seed", "1"]) == 0
    out = capsys.readouterr().out.lower()
    assert "win rate" in out

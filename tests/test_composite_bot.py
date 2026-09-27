"""
Tests for bots/composite_bot.py (Phase 3, step 3d part 1).

    CompositeBot(bidder, player)
        .bidder, .player
        choose_bid(view, legal_actions) -> bidder.choose_bid(view, legal_actions)
        choose_discard(view)            -> bidder.choose_discard(view)
        choose_play(view)               -> player.choose_play(view)
        as_callbacks() -> (choose_bid, choose_discard, choose_play)

It passes the arguments through unchanged and returns exactly what the inner
bot returned. It adds no logic of its own.
"""
import random

import pytest

cb = pytest.importorskip("bots.composite_bot", reason="bots/composite_bot.py not written yet")
CompositeBot = cb.CompositeBot

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from engine.game import play_game, play_hand
from evaluation.match import play_match, seat_callbacks


class RecordingBot:
    """Returns fixed sentinel values and records every call it receives."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def choose_bid(self, view, legal_actions):
        self.calls.append(("bid", view, legal_actions))
        return (self.name, "bid")

    def choose_discard(self, view):
        self.calls.append(("discard", view))
        return (self.name, "discard")

    def choose_play(self, view):
        self.calls.append(("play", view))
        return (self.name, "play")


def test_stores_the_two_bots():
    bidder, player = RecordingBot("B"), RecordingBot("P")
    bot = CompositeBot(bidder, player)
    assert bot.bidder is bidder
    assert bot.player is player


def test_bid_goes_to_the_bidder_with_the_same_arguments():
    bidder, player = RecordingBot("B"), RecordingBot("P")
    bot = CompositeBot(bidder, player)
    view, legal = object(), [object(), object()]
    assert bot.choose_bid(view, legal) == ("B", "bid")
    assert bidder.calls == [("bid", view, legal)]
    assert player.calls == []


def test_discard_goes_to_the_bidder():
    bidder, player = RecordingBot("B"), RecordingBot("P")
    bot = CompositeBot(bidder, player)
    view = object()
    assert bot.choose_discard(view) == ("B", "discard")
    assert bidder.calls == [("discard", view)]
    assert player.calls == []


def test_play_goes_to_the_player():
    bidder, player = RecordingBot("B"), RecordingBot("P")
    bot = CompositeBot(bidder, player)
    view = object()
    assert bot.choose_play(view) == ("P", "play")
    assert player.calls == [("play", view)]
    assert bidder.calls == []


def test_as_callbacks_returns_the_three_methods_in_order():
    bidder, player = RecordingBot("B"), RecordingBot("P")
    bid, discard, play = CompositeBot(bidder, player).as_callbacks()
    assert bid(object(), []) == ("B", "bid")
    assert discard(object()) == ("B", "discard")
    assert play(object()) == ("P", "play")


def test_heuristic_plus_heuristic_is_exactly_heuristic():
    # Same decisions as HeuristicBot, so a duplicate match is exactly even (decision #29).
    mixed = CompositeBot(HeuristicBot(), HeuristicBot())
    result = play_match(mixed, HeuristicBot(), n_pairs=20, seed=3)
    assert result.wins_a == result.games // 2
    assert result.points_a == result.points_b


def test_full_games_run_with_mixed_bots():
    rng = random.Random(0)
    team0 = CompositeBot(HeuristicBot(), RandomBot(random.Random(1)))
    team1 = CompositeBot(RandomBot(random.Random(2)), HeuristicBot())
    bid, discard, play = seat_callbacks(team0, team1)
    for _ in range(30):
        result = play_game(rng, bid, discard, play)
        assert max(result.final_scores.values()) >= 10


def test_heuristic_bidding_with_random_play_bids_like_heuristic():
    # Every bid and discard the composite makes is the one HeuristicBot would make.
    heuristic = HeuristicBot()
    mixed = CompositeBot(HeuristicBot(), RandomBot(random.Random(4)))
    seen = {"bid": 0, "discard": 0}

    def bid_fn(view, legal):
        action = mixed.choose_bid(view, legal)
        assert action == heuristic.choose_bid(view, legal)
        seen["bid"] += 1
        return action

    def discard_fn(view):
        card = mixed.choose_discard(view)
        assert card == heuristic.choose_discard(view)
        seen["discard"] += 1
        return card

    rng = random.Random(5)
    for i in range(200):
        play_hand(i % 4, rng, bid_fn, discard_fn, mixed.choose_play)
    assert seen["bid"] > 200 and seen["discard"] > 50


def test_heuristic_clearly_beats_heuristic_bidding_with_random_play():
    # The Phase 3 yardstick (decision #38): about 92% with equal bidding.
    opponent = CompositeBot(HeuristicBot(), RandomBot(random.Random(1)))
    result = play_match(HeuristicBot(), opponent, n_pairs=150, seed=0)
    assert 0.80 < result.win_rate < 0.99

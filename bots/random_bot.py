"""
Random baseline agent.

This is the yardstick your self-play agent has to beat before anything else
counts as progress. It implements the same three callback interfaces your
networks will implement:

    choose_bid_fn(state, legal_actions) -> BidAction
    choose_discard_fn(view)              -> Card
    choose_play_fn(view)                -> Card
"""
import random


class RandomBot:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose_bid(self, state, legal_actions):
        return self.rng.choice(legal_actions)

    def choose_discard(self, view):
        return self.rng.choice(view.hand)

    def choose_play(self, view):
        return self.rng.choice(view.legal_cards)

    def as_callbacks(self):
        """Convenience: unpack into the trio play_hand/play_game expect."""
        return self.choose_bid, self.choose_discard, self.choose_play
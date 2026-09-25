from dataclasses import dataclass
import math
import random

from engine.game import play_game


@dataclass
class MatchResult:
    games: int
    wins_a: int
    points_a: int
    points_b:int

    @property
    def win_rate(self): 
        return self.wins_a / self.games

    @property
    def std_error(self):
        return math.sqrt(self.win_rate * (1 - self.win_rate) / self.games)

def seat_callbacks(team0_bot, team1_bot):
    def choose_bid_fn(view, legal_actions):
        if view.player_id % 2 == 0:
            return team0_bot.choose_bid(view, legal_actions)
        else:
            return team1_bot.choose_bid(view, legal_actions)

    def choose_discard_fn(view):
        if view.player_id % 2 == 0:
            return team0_bot.choose_discard(view)
        else:
            return team1_bot.choose_discard(view)

    def choose_play_fn(view):
        if view.player_id % 2 == 0:
            return team0_bot.choose_play(view)
        else:
            return team1_bot.choose_play(view)

    return choose_bid_fn, choose_discard_fn, choose_play_fn

def play_match(bot_a, bot_b, n_pairs, seed, target=10) -> MatchResult:
    if n_pairs < 1:
        raise ValueError(f"n_pairs cannot be less than 1. n_pairs={n_pairs}")
    seed_generator = random.Random(seed)
    wins_a = 0
    points_a = 0
    points_b = 0

    for i in range(n_pairs):
        game_seed = seed_generator.randrange(2**32)
        first_dealer = i % 4

        for a_team in (0, 1):
            if a_team == 0:
                bid, discard, play = seat_callbacks(bot_a, bot_b)
            else:
                bid, discard, play = seat_callbacks(bot_b, bot_a)

            result = play_game(random.Random(game_seed), bid, discard, play, target=target, first_dealer=first_dealer)
            if result.winning_team == a_team:
                wins_a += 1
            points_a += result.final_scores[a_team]
            points_b += result.final_scores[1- a_team]

    return MatchResult(
        games=(n_pairs*2),
        wins_a=wins_a,
        points_a=points_a,
        points_b=points_b
    )
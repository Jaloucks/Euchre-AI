import argparse
import random
from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from evaluation.match import play_match


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='get win rate of games simulation'
    )
    parser.add_argument(
        '--pairs', default=500, type=int,
        help='number of game pairs to play'
    )
    parser.add_argument(
        '--seed', default=0, type=int,
        help='seed for the games to play for comparing runs'
    )

    args = parser.parse_args(argv)

    bot_a = HeuristicBot()
    bot_b = RandomBot(random.Random(args.seed))

    result = play_match(
        bot_a,
        bot_b,
        args.pairs,
        args.seed
    )

    margin = 1.96 * result.std_error
    print(f"{type(bot_a).__name__} vs {type(bot_b).__name__}  ({args.pairs} pairs, {result.games} games, seed {args.seed})")
    print(f"win rate: {result.win_rate:.3f} +/- {margin:.3f}   (95% CI {result.win_rate - margin:.3f}-{result.win_rate + margin:.3f})")
    print(f"points:   {result.points_a} - {result.points_b}   ({result.points_a / result.games:.1f} vs {result.points_b / result.games:.1f} per game)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
import argparse
import csv
from dataclasses import dataclass
import random

import numpy as np

from bots.composite_bot import CompositeBot
from bots.heuristic_bot import HeuristicBot
from engine.game import play_hand
from evaluation.evaluate import make_opponent
from evaluation.match import play_match, seat_callbacks
from rl.learning_bot import LearningPlayBot
from rl.optim import Adam
from rl.policy_net import PolicyMLP
from rl.reinforce import EMABaseline, hand_reward, reinforce_grad


@dataclass
class TrainConfig:
    hands: int = 100000
    batch_hands: int = 64
    lr: float = 1e-3
    hidden: tuple[int, ...] = (128, 128)
    trick_weight: float = 0.0
    baseline: str = "ema"
    ema_decay: float = 0.99
    opponent: str = "random-play"
    eval_every: int = 10000
    eval_pairs: int = 200
    seed: int = 0
    csv_path: str | None = None

def train(config):
    if config.baseline not in ("ema", "none"):
        raise ValueError(f"config.baseline must be \"ema\" or \"none\", is {config.baseline}")
    master = random.Random(config.seed)
    net = PolicyMLP(261, tuple(config.hidden), 24, random.Random(master.randrange(2**32)))
    bot = LearningPlayBot(net, random.Random(master.randrange(2**32)))
    opponent = make_opponent(config.opponent, random.Random(master.randrange(2**32)))
    deal_rng = random.Random(master.randrange(2**32))
    eval_seed = master.randrange(2**32)
    learner = CompositeBot(HeuristicBot(), bot)
    bid, discard, play = seat_callbacks(learner, opponent)
    optimizer = Adam(net.params(), lr=config.lr)
    baseline = EMABaseline(config.ema_decay)
    history = []

    def evaluate(hands_done, rewards):
        bot.training = False
        result = play_match(learner, opponent, config.eval_pairs, eval_seed)
        bot.training = True
        history.append({"hands": hands_done, "avg_reward": float(np.mean(rewards)) if rewards else None, "win_rate": result.win_rate, "ci": 1.96 * result.std_error})
        if config.csv_path:
            with open(config.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["hands", "avg_reward", "win_rate", "ci"])
                writer.writeheader()
                writer.writerows(history)

    evaluate(0, [])
    batch_decisions, batch_adv, batch_hands = [], [], 0
    rewards = []
    for i in range(config.hands):
        result = play_hand(i % 4, deal_rng, bid, discard, play)
        reward = hand_reward(result, 0, config.trick_weight)
        decisions = bot.pop_trajectory()

        b = baseline.value if config.baseline == "ema" else 0.0
        batch_decisions += decisions
        batch_adv += [reward - b] * len(decisions)
        baseline.update(reward)

        rewards.append(reward)
        batch_hands += 1

        if batch_hands == config.batch_hands:
            _, grads = reinforce_grad(net, batch_decisions, batch_adv, batch_hands)
            optimizer.step(grads)
            batch_decisions, batch_adv, batch_hands = [], [], 0

        if (i + 1) % config.eval_every == 0:
            evaluate(i + 1, rewards)
            rewards = []

    return net, history

def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3: train the card-play net with REINFORCE against a fixed opponent")
    defaults = TrainConfig()
    parser.add_argument("--hands", type=int, default=defaults.hands)
    parser.add_argument("--batch-hands", type=int, default=defaults.batch_hands)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--hidden", type=int, nargs="*", default=list(defaults.hidden))
    parser.add_argument("--trick-weight", type=float, default=defaults.trick_weight)
    parser.add_argument("--baseline", choices=["ema", "none"], default=defaults.baseline)
    parser.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--opponent", default=defaults.opponent)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-pairs", type=int, default=defaults.eval_pairs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)

    config = TrainConfig(
        hands=args.hands,
        batch_hands=args.batch_hands,
        lr=args.lr,
        hidden=tuple(args.hidden),
        trick_weight=args.trick_weight,
        baseline=args.baseline,
        ema_decay=args.ema_decay,
        opponent=args.opponent,
        eval_every=args.eval_every,
        eval_pairs=args.eval_pairs,
        seed=args.seed,
        csv_path=args.csv,
    )
    _, history = train(config)

    for row in history:
        if row["avg_reward"] is None:
            avg = "   -  "
        else:
            avg = f"{row['avg_reward']:+.3f}"
        print(f"hands {row['hands']:>7}  avg reward {avg}  win rate {row['win_rate']:.3f} +/- {row['ci']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
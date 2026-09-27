from dataclasses import dataclass

import numpy as np

from encoding.play_encoder import decode_play_action, encode_play_view
from rl.policy_net import masked_softmax


@dataclass
class Decision:
    features: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    player_id: int

class LearningPlayBot:
    def __init__(self, net, rng, training=True):
        self.net = net
        self.rng = rng
        self.training = training
        self.trajectory = []

    def choose_play(self, view):
        features, mask = encode_play_view(view)
        logits, _ = self.net.forward(features)
        if not self.training:
            action = np.argmax(np.where(mask, logits, -np.inf))
            return decode_play_action(action, view)
        probs = masked_softmax(logits, mask)
        u = self.rng.random()
        running_totals = np.cumsum(probs)
        action = int(np.searchsorted(running_totals, u, side="right"))
        if action == 24 or not mask[action]:
            action = int(np.flatnonzero(mask)[-1])
        self.trajectory.append(Decision(features, mask, action, float(np.log(probs[action])), view.player_id))
        return decode_play_action(action, view)

    def pop_trajectory(self):
        traj = self.trajectory
        self.trajectory = []
        return traj
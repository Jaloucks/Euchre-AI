import numpy as np

from rl.policy_net import masked_log_softmax


def hand_reward(result, team, trick_weight=0.0):
    our_tricks = 0
    their_tricks = 0
    for player, tricks in result.tricks_won.items():
        if team == player % 2:
            our_tricks += tricks
        else:
            their_tricks += tricks

    return float((result.scores[team] - result.scores[1 - team]) + trick_weight * (our_tricks - their_tricks))

class EMABaseline:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = 0.0

    def update(self, reward):
        self.value = self.decay * self.value + (1 - self.decay) * reward

def reinforce_grad(net, decisions, advantages, n_hands) -> tuple[float, np.ndarray]:
    if len(decisions) != len(advantages):
        raise ValueError(f"decisions length of {len(decisions)} does not equal advantages length {len(advantages)}")
    if n_hands < 1:
        raise ValueError("number of hands cannot be less than 1")
    if not decisions:
        return 0.0, [np.zeros_like(p) for p in net.params()]
    features = np.stack([d.features for d in decisions])
    masks = np.stack([d.mask for d in decisions])
    actions = np.array([d.action for d in decisions])
    adv = np.asarray(advantages, dtype=np.float64)
    logits, cache = net.forward(features)
    log_probs = masked_log_softmax(logits, masks)
    rows = np.arange(len(decisions))
    chosen = log_probs[rows, actions]
    stored = np.array([d.log_prob for d in decisions])
    if not np.allclose(chosen, stored, rtol=0, atol=1e-6):
        raise ValueError("stale log_prob")
    loss = -np.sum(adv * chosen) / n_hands
    probs = np.exp(log_probs)
    probs[rows, actions] -= 1.0
    dlogits = adv[:, None] * probs / n_hands
    grads = net.backward(cache, dlogits)
    return float(loss), grads
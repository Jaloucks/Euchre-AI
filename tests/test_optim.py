"""
Tests for rl/optim.py (Phase 3, step 3c): SGD and Adam.

    SGD(params, lr)                                          .step(grads)
    Adam(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)  .step(grads)

    params: a list of numpy arrays (e.g. net.params()). The optimizer keeps this list.
    grads:  a list of arrays in the same order and shapes (e.g. net.backward(...)).
    step() changes the arrays in `params` IN PLACE (p -= ...), never replaces them,
    and never modifies `grads`.

    SGD:   p  <-  p - lr * g
    Adam (per number, t = how many steps so far, counting this one):
           m  <-  beta1 * m + (1 - beta1) * g          (running average of g)
           v  <-  beta2 * v + (1 - beta2) * g * g      (running average of g squared)
           m_hat = m / (1 - beta1 ** t)                (bias correction: m and v start at 0)
           v_hat = v / (1 - beta2 ** t)
           p  <-  p - lr * m_hat / (sqrt(v_hat) + eps)
"""
import math
import random

import numpy as np
import pytest

optim = pytest.importorskip("rl.optim", reason="rl/optim.py not written yet")
SGD = optim.SGD
Adam = optim.Adam


def bowl_grads(params):
    # loss = sum of w^2 over every number  ->  gradient = 2w
    return [2.0 * p for p in params]


class ScalarAdam:
    # Independent, one-number-at-a-time copy of the Adam formulas above (test reference).
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = 0.0
        self.v = 0.0

    def change(self, g, t):
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** t)
        v_hat = self.v / (1 - self.b2 ** t)
        return -self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

def test_sgd_one_step_matches_hand_computation():
    W = np.array([[1.0, -2.0], [0.5, 0.0]])
    b = np.array([0.5])
    opt = SGD([W, b], lr=0.1)
    opt.step([np.array([[0.5, -1.0], [0.0, 2.0]]), np.array([-0.43])])
    np.testing.assert_allclose(W, [[0.95, -1.9], [0.5, -0.2]])
    np.testing.assert_allclose(b, [0.543])   # the lesson example: 0.5 - 0.1 * (-0.43)


def test_sgd_updates_in_place():
    a = np.array([1.0, 2.0])
    params = [a]
    SGD(params, lr=0.5).step([np.array([1.0, 1.0])])
    assert params[0] is a                    # same array object, not a replacement
    np.testing.assert_allclose(a, [0.5, 1.5])


def test_sgd_does_not_modify_grads():
    g = np.array([1.0, -1.0])
    SGD([np.zeros(2)], lr=0.1).step([g])
    np.testing.assert_array_equal(g, [1.0, -1.0])


def test_sgd_zero_gradient_changes_nothing():
    a = np.array([3.0, -4.0])
    SGD([a], lr=0.1).step([np.zeros(2)])
    np.testing.assert_array_equal(a, [3.0, -4.0])


def test_sgd_rolls_down_the_bowl():
    params = [np.array([1.0, -3.0]), np.array([[2.0], [0.5]])]
    opt = SGD(params, lr=0.1)
    for _ in range(200):
        opt.step(bowl_grads(params))
    for p in params:
        assert np.all(np.abs(p) < 1e-6)


def test_sgd_too_big_a_step_blows_up():
    # The lesson table: step size 1.1 overshoots further every step.
    w = np.array([1.0])
    opt = SGD([w], lr=1.1)
    for _ in range(10):
        opt.step(bowl_grads([w]))
    assert abs(w[0]) > 5.0


def test_sgd_can_be_used_for_many_steps_with_one_object():
    w = np.array([1.0])
    opt = SGD([w], lr=0.1)
    opt.step([np.array([1.0])])
    opt.step([np.array([1.0])])
    np.testing.assert_allclose(w, [0.8])


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

def test_adam_default_settings():
    # Same behavior as passing the documented defaults explicitly.
    a1, a2 = np.array([0.3, -0.2]), np.array([0.3, -0.2])
    o1 = Adam([a1])
    o2 = Adam([a2], lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    g = np.random.default_rng(0)
    for _ in range(5):
        grad = g.normal(0, 1, 2)
        o1.step([grad])
        o2.step([grad.copy()])
    np.testing.assert_array_equal(a1, a2)


def test_adam_first_step_moves_every_weight_by_lr_whatever_the_gradient_size():
    # After bias correction, step 1 is lr * g / |g| = lr * sign(g).
    # Big and small gradients move their weights the same distance.
    w = np.zeros(4)
    Adam([w], lr=0.01).step([np.array([-0.43, -43.0, 0.001, 5.0])])
    np.testing.assert_allclose(w, [0.01, 0.01, -0.01, -0.01], rtol=1e-4)


def test_adam_lesson_example_two_steps():
    # Lesson numbers: gradient -0.43, then +0.43 (noise flips the sign).
    w = np.array([0.0])
    opt = Adam([w], lr=0.001)
    opt.step([np.array([-0.43])])
    assert w[0] == pytest.approx(0.001, rel=1e-6)            # step 1: exactly lr
    opt.step([np.array([0.43])])
    assert w[0] == pytest.approx(0.001 - 0.0000526316, rel=1e-4)  # step 2: tiny move back


def test_adam_zero_gradient_first_step_changes_nothing():
    w = np.array([1.5, -2.5])
    Adam([w], lr=0.1).step([np.zeros(2)])
    np.testing.assert_array_equal(w, [1.5, -2.5])


def test_adam_matches_the_formulas_over_many_steps():
    # Two params with different shapes; every number has its own m and v,
    # and the step counter t is shared.
    rng = np.random.default_rng(1)
    params = [rng.normal(0, 1, (3, 2)), rng.normal(0, 1, (4,))]
    start = [p.copy() for p in params]
    opt = Adam(params, lr=0.05, beta1=0.8, beta2=0.99, eps=1e-6)
    refs = [[ScalarAdam(0.05, 0.8, 0.99, 1e-6) for _ in range(p.size)] for p in params]
    expected = [s.copy() for s in start]
    for t in range(1, 21):
        grads = [rng.normal(0, 1, p.shape) for p in params]
        if t % 4 == 0:
            grads[1][:] = 0.0                                 # some steps with zero gradient
        opt.step(grads)
        for e, g, ref in zip(expected, grads, refs):
            flat_e, flat_g = e.reshape(-1), g.reshape(-1)
            for k in range(flat_e.size):
                flat_e[k] += ref[k].change(float(flat_g[k]), t)
    for p, e in zip(params, expected):
        np.testing.assert_allclose(p, e, rtol=1e-10, atol=1e-12)


def test_adam_updates_in_place():
    a = np.array([1.0, 2.0])
    params = [a]
    Adam(params, lr=0.1).step([np.array([1.0, -1.0])])
    assert params[0] is a
    np.testing.assert_allclose(a, [0.9, 2.1], rtol=1e-6)


def test_adam_does_not_modify_grads():
    g = np.array([0.5, -0.25])
    opt = Adam([np.zeros(2)], lr=0.1)
    opt.step([g])
    opt.step([g])
    np.testing.assert_array_equal(g, [0.5, -0.25])


def test_adam_ignores_the_scale_of_the_gradients():
    # Decision 2: with Adam, multiplying every reward (so every gradient) by a
    # constant barely changes anything.
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 5)
    b = a.copy()
    oa, ob = Adam([a], lr=0.01), Adam([b], lr=0.01)
    for _ in range(30):
        g = rng.normal(0, 1, 5)
        oa.step([g])
        ob.step([g * 1000.0])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_adam_rolls_down_the_bowl():
    params = [np.array([3.0, -2.0, 0.5]), np.array([[1.0], [-1.0]])]
    opt = Adam(params, lr=0.05)
    for _ in range(1000):
        opt.step(bowl_grads(params))
    for p in params:
        assert np.all(np.abs(p) < 0.01)


def test_adam_handles_badly_scaled_bowl_that_sgd_cannot():
    # loss = 1000 * x^2 + 0.001 * y^2. One step size can't suit both directions.
    def grads(w):
        return [np.array([2000.0 * w[0], 0.002 * w[1]])]

    w_sgd = np.array([1.0, 1.0])
    sgd = SGD([w_sgd], lr=0.0009)       # any bigger and x blows up (needs lr < 0.001)
    w_adam = np.array([1.0, 1.0])
    adam = Adam([w_adam], lr=0.01)
    for _ in range(300):
        sgd.step(grads(w_sgd))
        adam.step(grads(w_adam))
    assert abs(w_sgd[1]) > 0.99         # SGD: y has barely moved
    assert abs(w_adam[0]) < 0.05 and abs(w_adam[1]) < 0.05   # Adam: both near 0


# ---------------------------------------------------------------------------
# Everything together: 3a + 3b + 3c
# ---------------------------------------------------------------------------

def test_adam_teaches_policy_net_to_prefer_one_card():
    # One fixed situation, 5 legal cards. Push toward card 14 using the
    # "played card" nudge from the lessons: dlogits = probabilities - onehot.
    pn = pytest.importorskip("rl.policy_net")
    net = pn.PolicyMLP(261, (32,), 24, random.Random(0))
    x = (np.random.default_rng(0).random(261) < 0.12).astype(np.float32)
    mask = np.zeros(24, dtype=bool)
    mask[[0, 5, 9, 14, 20]] = True
    target = 14

    opt = Adam(net.params(), lr=1e-2)
    for _ in range(100):
        logits, cache = net.forward(x)
        p = pn.masked_softmax(logits, mask)
        dlogits = p.copy()
        dlogits[target] -= 1.0
        opt.step(net.backward(cache, dlogits))

    logits, _ = net.forward(x)
    assert pn.masked_softmax(logits, mask)[target] > 0.95


def test_sgd_teaches_policy_net_to_prefer_one_card():
    pn = pytest.importorskip("rl.policy_net")
    net = pn.PolicyMLP(261, (32,), 24, random.Random(1))
    x = (np.random.default_rng(1).random(261) < 0.12).astype(np.float32)
    mask = np.zeros(24, dtype=bool)
    mask[[1, 2, 3]] = True
    target = 2

    opt = SGD(net.params(), lr=0.1)
    for _ in range(300):
        logits, cache = net.forward(x)
        p = pn.masked_softmax(logits, mask)
        dlogits = p.copy()
        dlogits[target] -= 1.0
        opt.step(net.backward(cache, dlogits))

    logits, _ = net.forward(x)
    assert pn.masked_softmax(logits, mask)[target] > 0.95

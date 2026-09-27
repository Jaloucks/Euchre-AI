"""
Tests for rl/policy_net.py, step 3a: masked softmax + forward pass.
(Step 3b adds backprop tests to this same file.)

Spec summary (full spec in the project doc claude/phase3-spec.md):

    masked_log_softmax(logits, mask) -> log-probabilities, same shape as logits
        legal entries:   logits - logsumexp(legal logits)      (per row)
        illegal entries: -inf
        works on shape (24,) or (N, 24); float64 output; inputs are not modified
        ValueError if any row has no legal entry
    masked_softmax(logits, mask) -> exp(masked_log_softmax(...)); illegal entries exactly 0.0

    PolicyMLP(input_dim, hidden, output_dim, rng: random.Random)
        .weights[i] shape (in, out), .biases[i] shape (out,), all float64
        layer sizes = [input_dim, *hidden, output_dim]
        init: hidden layers  W ~ Normal(0, sqrt(2 / fan_in))  (He)
              output layer   W ~ Normal(0, 0.01)
              all biases     0
        params() -> [W0, b0, W1, b1, ...]  (the real arrays, not copies)
        forward(x) -> (logits, cache)
            x shape (input_dim,) -> logits shape (output_dim,)
            x shape (N, input_dim) -> logits shape (N, output_dim)
            ReLU after every hidden layer, nothing after the output layer
            ValueError if x's last dimension != input_dim
"""
import random

import numpy as np
import pytest

pn = pytest.importorskip("rl.policy_net", reason="rl/policy_net.py not written yet")

masked_log_softmax = pn.masked_log_softmax
masked_softmax = pn.masked_softmax
PolicyMLP = pn.PolicyMLP


def make_net(hidden=(128, 128), seed=0, input_dim=261, output_dim=24):
    return PolicyMLP(input_dim, hidden, output_dim, random.Random(seed))


def random_features(n, dim=261, density=0.12, seed=0):
    # 0/1 vectors roughly as sparse as the real play encoder's output.
    g = np.random.default_rng(seed)
    return (g.random((n, dim)) < density).astype(np.float32)


# ---------------------------------------------------------------------------
# masked_log_softmax / masked_softmax
# ---------------------------------------------------------------------------

def test_softmax_matches_hand_computed_example():
    # The step 1 example: scores 2, 1, 0 -> 66.5%, 24.5%, 9.0%
    logits = np.array([2.0, 1.0, 0.0])
    mask = np.array([True, True, True])
    e = np.exp([2.0, 1.0, 0.0])
    np.testing.assert_allclose(masked_softmax(logits, mask), e / e.sum())
    np.testing.assert_allclose(masked_log_softmax(logits, mask), np.log(e / e.sum()))


def test_illegal_entries_are_exactly_zero_and_minus_inf():
    logits = np.array([0.5, 0.2, 2.0, 1.0])
    mask = np.array([True, True, False, False])
    p = masked_softmax(logits, mask)
    lp = masked_log_softmax(logits, mask)
    assert p[2] == 0.0 and p[3] == 0.0
    assert lp[2] == -np.inf and lp[3] == -np.inf


def test_step2_example_legal_cards_share_everything():
    # A-clubs 0.5, 9-clubs 0.2 legal; right bower 2.0 and K-diamonds 1.0 illegal.
    logits = np.array([0.5, 0.2, 2.0, 1.0])
    mask = np.array([True, True, False, False])
    p = masked_softmax(logits, mask)
    e = np.exp([0.5, 0.2])
    np.testing.assert_allclose(p[:2], e / e.sum())
    assert p.sum() == pytest.approx(1.0)


def test_illegal_scores_have_no_effect_even_if_huge():
    mask = np.array([True, False, True, False, True])
    a = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    b = np.array([1.0, 1e9, 2.0, -1e9, 3.0])
    np.testing.assert_allclose(masked_softmax(a, mask), masked_softmax(b, mask))


def test_probabilities_sum_to_one_per_row():
    g = np.random.default_rng(1)
    logits = g.normal(0, 3, (50, 24))
    mask = g.random((50, 24)) < 0.3
    mask[np.arange(50), g.integers(0, 24, 50)] = True  # at least one legal per row
    p = masked_softmax(logits, mask)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(50))
    assert np.all(p[~mask] == 0.0)
    assert np.all(p[mask] > 0.0)


def test_adding_a_constant_changes_nothing():
    # Step 1: only the gaps between scores matter.
    logits = np.array([3.0, 3.0, 1.0])
    mask = np.array([True, True, True])
    np.testing.assert_allclose(masked_softmax(logits, mask), masked_softmax(logits + 10.0, mask))


def test_single_legal_card_gets_probability_one():
    # Step 2: a forced move is 100% no matter its score.
    logits = np.array([-7.0, 4.0, 123.0])
    mask = np.array([True, False, False])
    assert masked_softmax(logits, mask)[0] == 1.0
    assert masked_log_softmax(logits, mask)[0] == 0.0


@pytest.mark.filterwarnings("error")  # any numpy RuntimeWarning (overflow, nan) fails the test
def test_huge_logits_are_numerically_stable():
    logits = np.array([1e4, 1e4 - 1.0, -1e4, 5e4])
    mask = np.array([True, True, True, False])
    p = masked_softmax(logits, mask)
    lp = masked_log_softmax(logits, mask)
    assert np.all(np.isfinite(lp[:3]))
    e = np.exp([0.0, -1.0])
    np.testing.assert_allclose(p[:2], e / e.sum())
    assert p[2] == pytest.approx(0.0, abs=1e-300)
    assert p[3] == 0.0


@pytest.mark.filterwarnings("error")
def test_hugely_negative_logits_are_numerically_stable():
    logits = np.array([-1e4, -1e4 - 1.0, 0.0])
    mask = np.array([True, True, False])
    p = masked_softmax(logits, mask)
    e = np.exp([0.0, -1.0])
    np.testing.assert_allclose(p[:2], e / e.sum())


def test_batched_equals_row_by_row():
    g = np.random.default_rng(2)
    logits = g.normal(0, 2, (8, 24))
    mask = g.random((8, 24)) < 0.4
    mask[:, 0] = True
    batched = masked_log_softmax(logits, mask)
    assert batched.shape == (8, 24)
    for i in range(8):
        np.testing.assert_array_equal(batched[i], masked_log_softmax(logits[i], mask[i]))


def test_row_with_no_legal_entry_raises():
    logits = np.zeros((2, 3))
    mask = np.array([[True, False, False], [False, False, False]])
    with pytest.raises(ValueError):
        masked_log_softmax(logits, mask)


def test_inputs_are_not_modified():
    logits = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, True])
    logits_before, mask_before = logits.copy(), mask.copy()
    masked_log_softmax(logits, mask)
    masked_softmax(logits, mask)
    np.testing.assert_array_equal(logits, logits_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_output_is_float64_even_for_float32_input():
    logits = np.array([1.0, 2.0], dtype=np.float32)
    mask = np.array([True, True])
    assert masked_log_softmax(logits, mask).dtype == np.float64


# ---------------------------------------------------------------------------
# PolicyMLP: construction and initialization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hidden", [(), (64,), (128, 128), (32, 16, 8)])
def test_layer_shapes_follow_hidden_sizes(hidden):
    net = make_net(hidden)
    sizes = [261, *hidden, 24]
    assert len(net.weights) == len(hidden) + 1
    assert len(net.biases) == len(hidden) + 1
    for i, (W, b) in enumerate(zip(net.weights, net.biases)):
        assert W.shape == (sizes[i], sizes[i + 1])
        assert b.shape == (sizes[i + 1],)
        assert W.dtype == np.float64 and b.dtype == np.float64


def test_hidden_is_stored_as_given():
    net = make_net([64, 32])
    assert tuple(net.hidden) == (64, 32)
    assert net.input_dim == 261 and net.output_dim == 24


def test_biases_start_at_zero():
    net = make_net((64, 32))
    for b in net.biases:
        assert np.all(b == 0.0)


def test_hidden_layers_use_he_init():
    net = PolicyMLP(261, (512, 512), 24, random.Random(0))
    for W in net.weights[:-1]:
        fan_in = W.shape[0]
        assert W.std() == pytest.approx(np.sqrt(2.0 / fan_in), rel=0.05)
        assert abs(W.mean()) < 0.01


def test_output_layer_starts_tiny():
    net = PolicyMLP(261, (512,), 24, random.Random(0))
    W_out = net.weights[-1]
    assert W_out.std() == pytest.approx(0.01, rel=0.1)
    assert abs(W_out.mean()) < 0.002


def test_no_hidden_layers_output_layer_is_still_tiny():
    # With hidden=(), the only layer IS the output layer, so it gets std 0.01.
    net = PolicyMLP(261, (), 24, random.Random(0))
    assert net.weights[0].std() == pytest.approx(0.01, rel=0.15)


def test_same_seed_same_weights_different_seed_different_weights():
    a, b, c = make_net(seed=5), make_net(seed=5), make_net(seed=6)
    for Wa, Wb in zip(a.weights, b.weights):
        np.testing.assert_array_equal(Wa, Wb)
    assert not np.array_equal(a.weights[0], c.weights[0])


def test_init_draws_from_the_injected_rng():
    # Decision #7: randomness comes from the injected random.Random.
    # Two nets built from ONE rng in a row must differ (the rng advanced).
    rng = random.Random(3)
    a = PolicyMLP(261, (16,), 24, rng)
    b = PolicyMLP(261, (16,), 24, rng)
    assert not np.array_equal(a.weights[0], b.weights[0])


def test_params_returns_the_real_arrays_in_order():
    net = make_net((16, 8))
    params = net.params()
    assert len(params) == 6
    expected = [net.weights[0], net.biases[0], net.weights[1], net.biases[1], net.weights[2], net.biases[2]]
    for p, e in zip(params, expected):
        assert p is e  # same object, so an optimizer can update them in place


def test_editing_params_in_place_changes_the_output():
    net = make_net((16,))
    x = random_features(1)[0]
    before, _ = net.forward(x)
    net.params()[-1][:] += 1.0  # output bias
    after, _ = net.forward(x)
    np.testing.assert_allclose(after, before + 1.0)


# ---------------------------------------------------------------------------
# PolicyMLP.forward
# ---------------------------------------------------------------------------

def test_forward_single_input_shape_and_dtype():
    net = make_net()
    logits, _ = net.forward(random_features(1)[0])
    assert logits.shape == (24,)
    assert logits.dtype == np.float64


def test_forward_batch_shape():
    net = make_net()
    logits, _ = net.forward(random_features(10))
    assert logits.shape == (10, 24)


def test_forward_batch_equals_one_at_a_time():
    net = make_net((32, 16))
    X = random_features(6)
    batched, _ = net.forward(X)
    for i in range(6):
        single, _ = net.forward(X[i])
        np.testing.assert_allclose(batched[i], single, rtol=1e-12, atol=1e-12)


def test_forward_matches_hand_computed_tiny_net():
    # 2 inputs -> 2 hidden (ReLU) -> 2 outputs, weights set by hand.
    net = PolicyMLP(2, (2,), 2, random.Random(0))
    net.weights[0][:] = [[1.0, -1.0],
                         [2.0, 1.0]]
    net.biases[0][:] = [0.0, -0.5]
    net.weights[1][:] = [[1.0, 0.0],
                         [3.0, -2.0]]
    net.biases[1][:] = [0.1, 0.2]
    # x = [1, 1]:  hidden pre-ReLU = [1+2+0, -1+1-0.5] = [3, -0.5]
    #              after ReLU      = [3, 0]
    #              output          = [3*1 + 0*3 + 0.1, 3*0 + 0*(-2) + 0.2] = [3.1, 0.2]
    logits, _ = net.forward(np.array([1.0, 1.0]))
    np.testing.assert_allclose(logits, [3.1, 0.2])


def test_no_relu_on_the_output_layer():
    # Output scores must be allowed to go negative.
    net = PolicyMLP(1, (1,), 1, random.Random(0))
    net.weights[0][:] = [[1.0]]
    net.biases[0][:] = [0.0]
    net.weights[1][:] = [[-2.0]]
    net.biases[1][:] = [0.0]
    logits, _ = net.forward(np.array([3.0]))
    np.testing.assert_allclose(logits, [-6.0])


def test_no_hidden_layers_is_a_plain_linear_layer():
    net = make_net(())
    X = random_features(4).astype(np.float64)
    logits, _ = net.forward(X)
    np.testing.assert_allclose(logits, X @ net.weights[0] + net.biases[0])


def test_forward_matches_manual_numpy_for_default_net():
    net = make_net((128, 128), seed=9)
    X = random_features(5, seed=9).astype(np.float64)
    h = X
    for W, b in zip(net.weights[:-1], net.biases[:-1]):
        h = np.maximum(h @ W + b, 0.0)
    expected = h @ net.weights[-1] + net.biases[-1]
    logits, _ = net.forward(X)
    np.testing.assert_allclose(logits, expected, rtol=1e-12, atol=1e-12)


def test_forward_accepts_float32_and_does_not_modify_input():
    net = make_net((16,))
    x = random_features(1)[0]  # float32, like the encoder's output
    x_before = x.copy()
    logits, _ = net.forward(x)
    assert logits.dtype == np.float64
    np.testing.assert_array_equal(x, x_before)


def test_forward_is_deterministic():
    net = make_net()
    x = random_features(1)[0]
    a, _ = net.forward(x)
    b, _ = net.forward(x)
    np.testing.assert_array_equal(a, b)


def test_forward_rejects_wrong_input_size():
    net = make_net()
    with pytest.raises(ValueError):
        net.forward(np.zeros(260))
    with pytest.raises(ValueError):
        net.forward(np.zeros((3, 262)))


def test_untrained_net_plays_close_to_uniform():
    # Step 1: the tiny output layer means an untrained net ~ RandomBot.
    net = make_net(seed=4)
    X = random_features(200, seed=4)
    logits, _ = net.forward(X)
    mask = np.zeros((200, 24), dtype=bool)
    mask[:, [0, 5, 9, 14, 20]] = True  # 5 legal cards
    p = masked_softmax(logits, mask)
    legal_p = p[mask].reshape(200, 5)
    assert np.all(np.abs(legal_p - 0.2) < 0.05)


# ===========================================================================
# Step 3b: PolicyMLP.backward(cache, dlogits) -> list of gradients
#
#   cache:   the second thing forward() returned
#   dlogits: d(loss)/d(logits), same shape as the logits from that forward call
#            ((output_dim,) for a single input, (N, output_dim) for a batch)
#   returns: [dW0, db0, dW1, db1, ...]  (same order and shapes as params()),
#            float64, SUMMED over the batch
#   ValueError if dlogits has the wrong shape
#   Must not change the weights, dlogits, or the cache.
#
# The "loss" in these tests is  L = sum(logits * dlogits),  so d(L)/d(logits) is
# exactly dlogits. The gradient check measures dL/dw by nudging each weight.
# ===========================================================================

def tiny_net():
    # The 2 -> 2 -> 2 net from the lessons (Part 1 section 5, backprop lesson 2).
    net = PolicyMLP(2, (2,), 2, random.Random(0))
    net.weights[0][:] = [[1.0, -1.0],
                         [2.0, 1.0]]
    net.biases[0][:] = [0.0, -0.5]
    net.weights[1][:] = [[1.0, 0.0],
                         [3.0, -2.0]]
    net.biases[1][:] = [0.1, 0.2]
    return net


def loss_fn(net, x, dlogits):
    logits, _ = net.forward(x)
    return float(np.sum(logits * dlogits))


def numeric_grad(net, x, dlogits, param, index, eps=1e-6):
    # Nudge one number up and down, measure how the loss changes (central difference).
    old = param[index]
    param[index] = old + eps
    up = loss_fn(net, x, dlogits)
    param[index] = old - eps
    down = loss_fn(net, x, dlogits)
    param[index] = old
    return (up - down) / (2 * eps)


def check_all_gradients(net, x, dlogits, max_per_param=None, seed=0):
    _, cache = net.forward(x)
    grads = net.backward(cache, dlogits)
    g = np.random.default_rng(seed)
    for p_i, (param, grad) in enumerate(zip(net.params(), grads)):
        indices = list(np.ndindex(param.shape))
        if max_per_param is not None and len(indices) > max_per_param:
            picks = g.choice(len(indices), size=max_per_param, replace=False)
            indices = [indices[k] for k in picks]
        for index in indices:
            expected = numeric_grad(net, x, dlogits, param, index)
            assert grad[index] == pytest.approx(expected, rel=1e-5, abs=1e-7), (
                f"param #{p_i} (shape {param.shape}) at {index}: "
                f"backward gave {grad[index]}, nudging measured {expected}"
            )


def test_backward_tiny_net_matches_the_lesson():
    # Backprop lesson 2: x = [1, 1], dlogits = [1, 2]
    net = tiny_net()
    _, cache = net.forward(np.array([1.0, 1.0]))
    dW0, db0, dW1, db1 = net.backward(cache, np.array([1.0, 2.0]))
    np.testing.assert_allclose(dW1, [[3.0, 6.0], [0.0, 0.0]])   # step A
    np.testing.assert_allclose(db1, [1.0, 2.0])
    np.testing.assert_allclose(dW0, [[1.0, 0.0], [1.0, 0.0]])   # step D (after the ReLU gate)
    np.testing.assert_allclose(db0, [1.0, 0.0])


def test_backward_off_neuron_gets_no_gradient():
    # Hidden neuron 1 is off (z = -0.5): nothing flows into or out of it.
    net = tiny_net()
    _, cache = net.forward(np.array([1.0, 1.0]))
    dW0, db0, dW1, db1 = net.backward(cache, np.array([0.7, -1.3]))
    assert np.all(dW1[1, :] == 0.0)   # weights out of the off neuron
    assert np.all(dW0[:, 1] == 0.0)   # weights into the off neuron
    assert db0[1] == 0.0


def test_backward_returns_params_order_shapes_and_dtype():
    net = make_net((16, 8), input_dim=10, output_dim=6)
    _, cache = net.forward(random_features(4, dim=10))
    grads = net.backward(cache, np.ones((4, 6)))
    params = net.params()
    assert isinstance(grads, list)
    assert len(grads) == len(params)
    for p, gr in zip(params, grads):
        assert gr.shape == p.shape
        assert gr.dtype == np.float64


def test_backward_zero_dlogits_gives_zero_gradients():
    net = make_net((16,), input_dim=10, output_dim=6)
    _, cache = net.forward(random_features(3, dim=10))
    for gr in net.backward(cache, np.zeros((3, 6))):
        assert np.all(gr == 0.0)


@pytest.mark.parametrize("hidden", [(), (5,), (6, 4), (4, 4, 4)])
def test_gradient_check_batch(hidden):
    net = make_net(hidden, seed=1, input_dim=7, output_dim=5)
    g = np.random.default_rng(1)
    x = g.normal(0, 1, (3, 7))          # any real numbers work, not only 0/1
    dlogits = g.normal(0, 1, (3, 5))
    check_all_gradients(net, x, dlogits)


@pytest.mark.parametrize("hidden", [(), (5,), (6, 4)])
def test_gradient_check_single_input(hidden):
    net = make_net(hidden, seed=2, input_dim=7, output_dim=5)
    g = np.random.default_rng(2)
    x = g.normal(0, 1, 7)
    dlogits = g.normal(0, 1, 5)
    check_all_gradients(net, x, dlogits)


def test_gradient_check_default_net_on_encoder_like_input():
    # The real shape: 261 -> 128 -> 128 -> 24, 0/1 inputs. Spot-checks 40 numbers per param.
    net = make_net((128, 128), seed=3)
    x = random_features(4, seed=3)
    dlogits = np.random.default_rng(3).normal(0, 1, (4, 24))
    check_all_gradients(net, x, dlogits, max_per_param=40)


def test_gradient_check_after_weights_have_moved():
    # Bigger weights and nonzero biases: more neurons switch on/off in interesting ways.
    net = make_net((6, 5), seed=4, input_dim=7, output_dim=5)
    g = np.random.default_rng(4)
    for p in net.params():
        p[...] = g.normal(0, 0.7, p.shape)
    x = g.normal(0, 1, (5, 7))
    dlogits = g.normal(0, 1, (5, 5))
    check_all_gradients(net, x, dlogits)


def test_batch_gradient_is_the_sum_of_row_gradients():
    net = make_net((6, 4), seed=5, input_dim=7, output_dim=5)
    g = np.random.default_rng(5)
    X = g.normal(0, 1, (3, 7))
    D = g.normal(0, 1, (3, 5))
    _, cache = net.forward(X)
    batch = net.backward(cache, D)
    total = None
    for i in range(3):
        _, c = net.forward(X[i])
        row = net.backward(c, D[i])
        total = row if total is None else [t + r for t, r in zip(total, row)]
    for b, t in zip(batch, total):
        np.testing.assert_allclose(b, t, rtol=1e-12, atol=1e-12)


def test_single_input_equals_batch_of_one():
    net = make_net((8,), seed=6, input_dim=7, output_dim=5)
    g = np.random.default_rng(6)
    x = g.normal(0, 1, 7)
    d = g.normal(0, 1, 5)
    _, c1 = net.forward(x)
    _, c2 = net.forward(x.reshape(1, -1))
    for a, b in zip(net.backward(c1, d), net.backward(c2, d.reshape(1, -1))):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_backward_does_not_change_weights_or_dlogits():
    net = make_net((8, 8), seed=7, input_dim=7, output_dim=5)
    before = [p.copy() for p in net.params()]
    _, cache = net.forward(random_features(3, dim=7))
    dlogits = np.ones((3, 5))
    net.backward(cache, dlogits)
    for p, b in zip(net.params(), before):
        np.testing.assert_array_equal(p, b)
    np.testing.assert_array_equal(dlogits, np.ones((3, 5)))


def test_backward_can_be_called_twice_with_the_same_cache():
    net = make_net((8, 8), seed=8, input_dim=7, output_dim=5)
    _, cache = net.forward(random_features(3, dim=7))
    d = np.random.default_rng(8).normal(0, 1, (3, 5))
    first = net.backward(cache, d)
    second = net.backward(cache, d)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_backward_rejects_wrong_dlogits_shape():
    net = make_net((8,), input_dim=7, output_dim=5)
    _, cache = net.forward(random_features(3, dim=7))
    with pytest.raises(ValueError):
        net.backward(cache, np.zeros((3, 4)))
    with pytest.raises(ValueError):
        net.backward(cache, np.zeros((2, 5)))
    _, cache1 = net.forward(random_features(1, dim=7)[0])
    with pytest.raises(ValueError):
        net.backward(cache1, np.zeros(4))

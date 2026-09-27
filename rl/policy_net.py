import math
import random

import numpy as np


def masked_log_softmax(logits, mask) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any(axis=-1).all():
        raise ValueError("every row needs at least one legal card")

    masked = np.where(mask, logits, -np.inf)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    log_total = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_total


def masked_softmax(logits, mask) -> np.ndarray:
    return np.exp(masked_log_softmax(logits, mask))


class PolicyMLP:
    def __init__(self, input_dim: int, hidden: tuple[int, ...], output_dim: int, rng: random.Random):
        self.input_dim = input_dim
        self.hidden = tuple(hidden)
        self.output_dim = output_dim

        sizes = [input_dim, *self.hidden, output_dim]
        g = np.random.default_rng(rng.randrange(2**32))

        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            if i == len(sizes) - 2:
                std = 0.01
            else:
                std = math.sqrt(2 / fan_in)
            self.weights.append(g.normal(0.0, std, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    def params(self):
        result = []
        for W, b in zip(self.weights, self.biases):
            result.append(W)
            result.append(b)
        return result

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} features, got {x.shape[-1]}")
        single = (x.ndim == 1)
        if single:
            x = x.reshape(1, -1)
        h = x
        layer_inputs = []
        for i in range(len(self.weights)):
            layer_inputs.append(h)
            z = h @ self.weights[i] + self.biases[i]
            if i == len(self.weights) - 1:
                h = z
            else:
                h = np.maximum(z, 0.0)
        if single:
            logits = h[0]
        else:
            logits = h
        return logits, (single, layer_inputs)

    def backward(self, cache, dlogits) -> list[np.ndarray]:
        single, layer_inputs = cache
        dlogits = np.asarray(dlogits, dtype=np.float64)
        n = layer_inputs[0].shape[0]
        expected = (self.output_dim,) if single else (n, self.output_dim)
        if dlogits.shape != expected:
            raise ValueError(f"expected dlogits of shape {expected}, got {dlogits.shape}")
        if single:
            dz = dlogits.reshape(1, -1)
        else:
            dz = dlogits
        gradients = [None] * (2 * len(layer_inputs))
        for i in reversed(range(len(layer_inputs))):
            W = self.weights[i]
            h_in = layer_inputs[i]
            gradients[2 * i] = h_in.T @ dz
            gradients[2 * i + 1] = dz.sum(axis=0)
            if i > 0:
                dh = dz @ W.T
                dz = dh * (h_in > 0)
        return gradients
                
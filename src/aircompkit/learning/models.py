"""NumPy models with a flat parameter vector (FEATURES L4, L7).

A model exposes ``init(rng) -> params``, ``loss_grad(params, x, y) -> (loss, grad)``, and
``evaluate(params, x, y) -> (loss, accuracy)``. ``params`` is a 1-D float array, which is what
devices transmit. Any object with these three methods can be used as a model (L7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aircompkit.core.registry import register


@register("model", "mlp")
@dataclass
class MLP:
    """Fully connected network, ReLU hidden layers, softmax cross-entropy. ``hidden=[]`` gives
    multinomial logistic regression."""

    d_in: int
    n_classes: int
    hidden: list[int] = field(default_factory=lambda: [64])

    def __post_init__(self) -> None:
        dims = [self.d_in, *self.hidden, self.n_classes]
        self._shapes = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        sizes = [a * b + b for a, b in self._shapes]
        self._offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self.size = int(self._offsets[-1])

    def _unpack(self, params):
        out = []
        for (a, b), off in zip(self._shapes, self._offsets[:-1]):
            W = params[off: off + a * b].reshape(a, b)
            out.append((W, params[off + a * b: off + a * b + b]))
        return out

    def init(self, rng: np.random.Generator) -> np.ndarray:
        parts = []
        for a, b in self._shapes:
            parts.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).ravel())
            parts.append(np.zeros(b))
        return np.concatenate(parts)

    def _forward(self, params, x):
        acts = [x]
        layers = self._unpack(params)
        for i, (W, b) in enumerate(layers):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0.0) if i < len(layers) - 1 else z)
        return layers, acts

    @staticmethod
    def _softmax_ce(logits, y):
        z = logits - logits.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean(), logp

    def loss_grad(self, params, x, y):
        layers, acts = self._forward(params, x)
        loss, logp = self._softmax_ce(acts[-1], y)
        delta = np.exp(logp)
        delta[np.arange(len(y)), y] -= 1.0
        delta /= len(y)
        grads = []
        for i in range(len(layers) - 1, -1, -1):
            W, _ = layers[i]
            grads.append(((acts[i].T @ delta).ravel(), delta.sum(axis=0)))
            if i > 0:
                delta = (delta @ W.T) * (acts[i] > 0)
        flat = [g for pair in reversed(grads) for g in pair]
        return float(loss), np.concatenate(flat)

    def evaluate(self, params, x, y, batch: int = 4096):
        losses, correct = 0.0, 0
        for lo in range(0, len(y), batch):
            _, acts = self._forward(params, x[lo: lo + batch])
            loss, _ = self._softmax_ce(acts[-1], y[lo: lo + batch])
            losses += loss * len(acts[-1])
            correct += int((acts[-1].argmax(axis=1) == y[lo: lo + batch]).sum())
        return losses / len(y), correct / len(y)


@register("model", "logreg")
def logreg(d_in: int, n_classes: int) -> MLP:
    return MLP(d_in, n_classes, hidden=[])

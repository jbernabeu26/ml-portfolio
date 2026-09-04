"""
SECTION 2.5 ADDENDUM -- softmax, log-sum-exp trick, multiclass logistic
regression (guided exercise, YOU implement the TODOs)

You already have Categorical/Multinomial from the earlier session
(categorical_multinomial_scratch.py) -- that covered 2.5.1. This file covers
the rest of 2.5 that hasn't been built yet: 2.5.2 (softmax), 2.5.4
(log-sum-exp trick), and 2.5.3 (multiclass logistic regression), applied to
classifying a synthetic pixel into forest/water/urban from 2 band values.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)


# ---------------------------------------------------------------------------
# 2.5.2 -- softmax, naive version (will break on large logits)
# ---------------------------------------------------------------------------

def softmax_naive(logits):
    """
    TODO: S(a)_c = exp(a_c) / sum_c' exp(a_c')
    Implement it literally, no tricks. You'll deliberately break this
    below with large logits -- that's the point of this version.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2.5.4 -- log-sum-exp trick -- the stable version
# ---------------------------------------------------------------------------

def log_sum_exp(logits):
    """
    TODO: lse(a) = m + log(sum(exp(a - m))) where m = max(a).
    This identity holds for ANY m (2.5.4 states it for all m) -- using
    m = max(a) is what keeps every exponent <= 0, preventing overflow.
    """
    raise NotImplementedError


def softmax_stable(logits):
    """
    TODO: p(y=c|x) = exp(a_c - lse(a)), using your log_sum_exp above.
    Should give the SAME result as softmax_naive for small logits, but
    not blow up for large ones.
    """
    raise NotImplementedError


def demo_overflow():
    """
    Provided demo scaffold -- fill in the two calls once your functions
    above exist. Try logits = np.array([1000., 1001., 1002.]) -- naive
    softmax will hit exp overflow (inf/inf -> nan), stable will not.
    """
    logits = np.array([1000.0, 1001.0, 1002.0])
    print("naive: ", "TODO: call softmax_naive(logits)")
    print("stable:", "TODO: call softmax_stable(logits)")


# ---------------------------------------------------------------------------
# 2.5.3 -- multiclass logistic regression, from scratch
# ---------------------------------------------------------------------------

def generate_landcover_pixels(n_per_class=150):
    """
    Provided: 3 classes (forest=0, water=1, urban=2), 2 synthetic band
    features per pixel, roughly separable clusters.
    """
    centers = np.array([[0.2, 0.7], [0.8, 0.1], [0.6, 0.6]])  # forest, water, urban
    X, y = [], []
    for c, center in enumerate(centers):
        pts = center + rng.normal(0, 0.08, size=(n_per_class, 2))
        X.append(pts)
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def one_hot(y, n_classes):
    """
    TODO: convert integer labels y (shape (N,)) into one-hot matrix (N, n_classes).
    """
    raise NotImplementedError


def forward_multiclass(X, W, b):
    """
    TODO: logits = X @ W + b  (X: (N,2), W: (2,C), b: (C,) -> logits (N,C))
    then apply your softmax_stable to EACH ROW (each pixel's logits)
    to get class probabilities, shape (N,C).
    """
    raise NotImplementedError


def cross_entropy_loss(y_onehot, probs):
    """
    TODO: NLL = -mean( sum_c y_onehot[:,c] * log(probs[:,c]) )
    Clip probs away from 0 first.
    """
    raise NotImplementedError


def gradients_multiclass(X, y_onehot, probs):
    """
    TODO: same clean result as the binary case generalizes here:
    d(NLL)/d(logits) = probs - y_onehot  (shape (N,C))
    From there, chain-rule through logits = X @ W + b to get grad_W (2,C)
    and grad_b (C,).
    """
    raise NotImplementedError


def train_multiclass_logistic_regression(X, y, n_classes=3, lr=0.1, n_steps=2000):
    """
    TODO: gradient descent loop, analogous to the binary version in 2.4's
    exercise, but with W shape (2, n_classes) and b shape (n_classes,).
    """
    raise NotImplementedError


if __name__ == "__main__":
    demo_overflow()

    X, y = generate_landcover_pixels()
    # W, b = train_multiclass_logistic_regression(X, y)
    # print("learned W:\n", W, "\nlearned b:", b)

    print("Fill in the TODOs above (softmax pair, then multiclass logreg).")

    # Suggested visual once trained: scatter the 3 pixel clusters colored
    # by true class, overlay a fine grid colored by ARGMAX predicted class
    # (imshow of predicted class over a meshgrid) -- gives you a genuine
    # "classified land-cover map" look, same idea as satellite classification
    # products but on synthetic data.

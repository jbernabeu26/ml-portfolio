"""
SECTION 2.5 ADDENDUM -- softmax, log-sum-exp trick, multiclass logistic
regression

THE PROBLEM: a satellite band-sensor gives 2 numbers per pixel. Each pixel
is actually one of 3 land-cover types -- forest, water, or urban -- but you
only see the 2 band values, not the label. Can those 2 numbers predict
which of the 3 classes a pixel belongs to?

This generalizes section 2.4's binary flood classifier (2 classes: sigmoid,
forward, nll_loss, gradients) to 3+ classes (softmax, forward_multiclass,
cross_entropy_loss, gradients_multiclass) -- same shape of problem, same
kind of solution, one more class. You already covered the single-pixel
distribution side of this (Categorical/Multinomial, 2.5.1) in an earlier
session; this file is the predictive half: 2.5.2 (softmax), 2.5.4
(log-sum-exp), 2.5.3 (multiclass logistic regression).

Order (each step needs the ones before it):
  1. softmax_naive, log_sum_exp, softmax_stable -- standalone, generalize sigmoid
  2. one_hot                                     -- standalone, prep for the loss
  3. forward_multiclass                          -- needs softmax_stable
  4. cross_entropy_loss                          -- needs forward_multiclass's output
  5. gradients_multiclass                        -- needs forward_multiclass's output
  6. train_multiclass_logistic_regression         -- needs #3-#5, solves the problem
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)


# ---------------------------------------------------------------------------
# 2.5.2 -- softmax, naive version (will break on large logits)
# ---------------------------------------------------------------------------

def softmax_naive(logits):
    """
    Why: generalizes sigmoid to 3 classes -- turns 3 raw scores into 3
    probabilities summing to 1. Deliberately fragile: broken on large
    logits below, motivating log_sum_exp/softmax_stable next.
    In: logits (C,) or (N,C), any real numbers.
    Out: same shape, values in (0,1), sums to 1 along the last axis.
    Formula: S(a)_c = exp(a_c) / sum_c' exp(a_c')
    """
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


# ---------------------------------------------------------------------------
# 2.5.4 -- log-sum-exp trick -- the stable version
# ---------------------------------------------------------------------------

def log_sum_exp(logits):
    """
    Why: the numerically stable building block softmax_stable needs next --
    keeps every exponent <= 0, preventing the overflow softmax_naive hits.
    In: logits (C,), any real numbers.
    Out: scalar, log(sum(exp(logits))).
    Formula: lse(a) = m + log(sum(exp(a - m))), m=max(a) (holds for any m).
    """
    m = np.max(logits)
    logsumexp = m + np.log(np.sum(np.exp(logits - m)))
    return logsumexp


def softmax_stable(logits):
    """
    Why: the version forward_multiclass actually uses -- same math as
    softmax_naive, immune to overflow.
    In: logits (C,), any real numbers. Out: (C,), values in (0,1), sums to 1.
    Formula: p(y=c|x) = exp(a_c - lse(a)), using log_sum_exp above.
    """
    lse = log_sum_exp(logits)
    return np.exp(logits - lse)


def demo_overflow():
    """
    Why: proves softmax_naive and softmax_stable diverge exactly where
    expected -- large logits are common once training pushes weights up,
    so this isn't a hypothetical edge case.
    In: none. Out: none (prints).
    """
    logits = np.array([1000.0, 1001.0, 1002.0])
    print("naive: ", softmax_naive(logits))
    print("stable:", softmax_stable(logits))


# ---------------------------------------------------------------------------
# 2.5.3 -- multiclass logistic regression, from scratch
# ---------------------------------------------------------------------------

def generate_landcover_pixels(n_per_class=150):
    """
    Why: the dataset the whole problem is about -- 3 classes, 2 band
    features per pixel, roughly separable clusters.
    In: n_per_class (int). Out: X (3*n_per_class, 2), y (3*n_per_class,) in {0,1,2}.
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
    Why: cross_entropy_loss and gradients_multiclass need labels in this
    form to compare directly against per-class probabilities.
    In: y (N,) ints in {0,...,n_classes-1}, n_classes (int).
    Out: (N, n_classes), row i is all zeros except a 1 at column y[i].
    """
    return np.eye(n_classes)[y]


def forward_multiclass(X, W, b):
    """
    Why: the classifier itself -- turns (band1, band2) into a probability
    per land-cover class. This IS the answer to "can 2 numbers predict the
    class," once W, b are trained.
    In: X (N,2), W (2,C), b (C,). Out: (N,C), each row sums to 1.
    Formula: softmax_stable(X @ W + b), applied per row.
    """
    logits = X @ W + b
    return softmax_stable(logits)


def cross_entropy_loss(y_onehot, probs):
    """
    Why: one scalar saying how wrong forward_multiclass's predictions are --
    generalizes 2.4's binary nll_loss to 3 classes. Needed before improving
    W, b (next: gradients_multiclass).
    In: y_onehot (N,C) from one_hot. probs (N,C) from forward_multiclass.
    Out: scalar.
    Formula: -mean( sum_c y_onehot[:,c] * log(probs[:,c]) ). Clip probs first.
    """
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))


def gradients_multiclass(X, y_onehot, probs):
    """
    Why: tells you which direction to move W, b to reduce cross_entropy_loss
    -- needed by the training loop, next.
    In: X (N,2), y_onehot (N,C), probs (N,C). Out: (grad_W (2,C), grad_b (C,)).
    Derive on paper: d(loss)/d(logits) = probs - y_onehot (N,C) -- same
    clean form as 2.4's (p-y), generalized to C classes. Chain-rule through
    logits = X@W+b: grad_W = X.T @ d_logits / N, grad_b = mean(d_logits, axis=0).
    """
    return (X.T @ (probs - y_onehot)) / X.shape[0], np.mean(probs - y_onehot, axis=0)


def train_multiclass_logistic_regression(X, y, n_classes=3, lr=0.1, n_steps=2000):
    """
    Why: this answers the problem -- runs forward_multiclass/
    cross_entropy_loss/gradients_multiclass together, repeatedly, until W, b
    actually separate the 3 classes.
    In: X (N,2), y (N,) ints, n_classes (int), lr (float), n_steps (int).
    Out: (W (2,n_classes), b (n_classes,)).
    """
    y_onehot = one_hot(y, n_classes)
    W = np.zeros((X.shape[1], n_classes))
    b = np.zeros(n_classes)
    for step in range(n_steps):
        probs = forward_multiclass(X, W, b)
        loss = cross_entropy_loss(y_onehot, probs)
        grad_W, grad_b = gradients_multiclass(X, y_onehot, probs)
        W -= lr * grad_W
        b -= lr * grad_b
        if step % 100 == 0:
            print(f"Step {step}, loss: {loss:.4f}")
    return W, b


if __name__ == "__main__":
    demo_overflow()

    X, y = generate_landcover_pixels()
    W, b = train_multiclass_logistic_regression(X, y)
    print("learned W:\n", W, "\nlearned b:", b)

    # Answering the problem visually: classify every point on a fine grid,
    # color the background by predicted class, overlay the real pixels
    # colored by TRUE class -- a genuine "classified land-cover map."
    band1, band2 = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid_points = np.column_stack([band1.ravel(), band2.ravel()])
    forward_probs = forward_multiclass(grid_points, W, b)
    predicted_classes = np.argmax(forward_probs, axis=1).reshape(band1.shape)

    fig, ax = plt.subplots()
    ax.contourf(band1, band2, predicted_classes, alpha=0.3, cmap="tab10")
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="tab10")
    ax.set_title("Multiclass Logistic Regression Land-Cover Classification")
    ax.set_xlabel("Band 1")
    ax.set_ylabel("Band 2")
    plt.savefig("multiclass_landcover_classification.png")
    plt.show()
"""
SECTION 2.4 -- Bernoulli/Binomial distributions, sigmoid, binary logistic
regression (guided exercise, YOU implement the TODOs)

Theme: "Flood pixel classifier" -- given a pixel's elevation and nearby
rainfall, is it flooded (1) or not (0)? Binary logistic regression is
literally the model 2.4.3 describes: p(y|x) = Ber(y | sigma(w^T x + b)).
"""

import numpy as np
from math import comb
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)


# ---------------------------------------------------------------------------
# 2.4.1 -- Binomial pmf, from scratch (no scipy)
# ---------------------------------------------------------------------------

def binomial_pmf(k, n, theta):
    """
    TODO: Bin(k|n,theta) := C(n,k) * theta^k * (1-theta)^(n-k)
    Use math.comb(n, k) for the binomial coefficient (that's the one
    piece of "library" allowed here -- computing C(n,k) by hand from
    factorials is a numerical-stability trap, not a learning objective).

    Self-check: sum over k=0..n of binomial_pmf(k, n, theta) must equal 1.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2.4.2 -- sigmoid function
# ---------------------------------------------------------------------------

def sigmoid(a):
    """
    TODO: sigma(a) := 1 / (1 + e^(-a))
    Works fine as a one-liner with np.exp -- the point isn't difficulty,
    it's noticing where else this exact function reappears (softmax with
    2 classes reduces to this -- you'll prove that to yourself in 2.5).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2.4.3 -- binary logistic regression, from scratch
# ---------------------------------------------------------------------------

def generate_flood_data(n_pixels=300):
    """
    Provided: synthetic pixels with 2 features (elevation_m, rainfall_mm),
    label = 1 (flooded) if a noisy linear combination crosses a threshold.
    """
    elevation = rng.uniform(0, 100, n_pixels)
    rainfall = rng.uniform(0, 200, n_pixels)
    X = np.column_stack([elevation, rainfall])
    true_w = np.array([-0.08, 0.05])
    true_b = 2.0
    logits = X @ true_w + true_b + rng.normal(0, 0.5, n_pixels)
    y = (logits > 0).astype(int)
    return X, y


def forward(X, w, b):
    """
    TODO: return p(y=1|x) = sigmoid(X @ w + b) for a batch of inputs X.
    Shape: X is (N,2), w is (2,), b is scalar -> return shape (N,).
    """
    raise NotImplementedError


def nll_loss(y_true, y_pred_prob):
    """
    TODO: negative log-likelihood (binary cross-entropy) for the whole batch:
        NLL = -mean( y*log(p) + (1-y)*log(1-p) )
    Clip y_pred_prob away from exactly 0 or 1 first (e.g. np.clip(p, 1e-9, 1-1e-9))
    to avoid log(0).
    """
    raise NotImplementedError


def gradients(X, y_true, y_pred_prob):
    """
    TODO: derive and implement the gradient of NLL w.r.t. w and b.
    Hint (not the full derivation -- work this out on paper first, it's
    a classic interview question): because sigmoid's derivative has a
    uniquely clean form, the gradient of NLL w.r.t. the LOGITS collapses
    to simply (y_pred_prob - y_true). From there, use the chain rule
    through logits = X @ w + b to get d(NLL)/dw and d(NLL)/db.
    Return (grad_w, grad_b) with grad_w shape (2,), grad_b scalar.
    """
    raise NotImplementedError


def train_logistic_regression(X, y, lr=0.001, n_steps=2000):
    """
    TODO: standard gradient descent loop using forward/nll_loss/gradients
    above. Initialize w=zeros(2), b=0. Print loss every 200 steps so you
    can watch it decrease. Return final (w, b).

    Note: features here are on very different scales (elevation 0-100,
    rainfall 0-200) -- if training is unstable or refuses to converge,
    that's not a bug in your gradient code, it's a lesson about feature
    scaling. Try standardizing X first if you get stuck.
    """
    raise NotImplementedError


if __name__ == "__main__":
    X, y = generate_flood_data()

    # w, b = train_logistic_regression(X, y)
    # print("learned w, b:", w, b)

    # Self-check against a library implementation -- your w/b won't be
    # identical (different optimizer, no regularization by default in
    # sklearn) but decision boundary should look similar:
    # from sklearn.linear_model import LogisticRegression
    # ref = LogisticRegression(penalty=None).fit(X, y)
    # print("sklearn coef_, intercept_:", ref.coef_, ref.intercept_)

    print("Fill in sigmoid, forward, nll_loss, gradients, train_logistic_regression.")

    # Suggested visual once trained: scatter X colored by y (flooded/not),
    # overlaid with the decision boundary line where sigmoid(w.x+b)=0.5
    # (i.e. where w[0]*elevation + w[1]*rainfall + b == 0) -- looks like a
    # simple flood-risk map cross-section.

    # Also worth a quick separate plot: binomial_pmf(k, 10, theta) as a bar
    # chart for theta=0.25 and theta=0.9, mirroring the book's own figure.

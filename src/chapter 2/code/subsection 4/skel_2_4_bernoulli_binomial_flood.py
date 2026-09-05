"""
SECTION 2.4 -- Bernoulli/Binomial, sigmoid, binary logistic regression

THE INVESTIGATION: one region's pixels, each with (elevation, rainfall) and
a flooded/not-flooded label. Two questions about this SAME data:

  (A) Statistically: this region floods a fraction theta of the time --
      what's the distribution of "how many flood" if I sample n pixels?
      -> binomial_pmf, using theta estimated straight from the data.

  (B) Predictively: CAN elevation/rainfall tell me WHICH pixels flood?
      -> sigmoid -> forward -> nll_loss -> gradients -> train, one
      classifier, each function needed by the next.

main() runs every function below, in order, on this one dataset. Nothing
is commented out: running this file top to bottom IS the workflow. It
will stop with NotImplementedError at whichever function you haven't
finished yet -- that's expected, not a bug. Implement one, rerun, watch
it get one function further, repeat.
"""

import numpy as np
from math import comb
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)


def generate_flood_data(n_pixels=300):
    """Provided. Out: X (n_pixels,2) [elevation_m, rainfall_mm], y (n_pixels,) in {0,1}."""
    elevation = rng.uniform(0, 100, n_pixels)
    rainfall = rng.uniform(0, 200, n_pixels)
    X = np.column_stack([elevation, rainfall])
    true_w = np.array([-0.08, 0.05])
    true_b = 2.0
    logits = X @ true_w + true_b + rng.normal(0, 0.5, n_pixels)
    y = (logits > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# (A) Statistical question: how many pixels flood, out of a sample of n?
# ---------------------------------------------------------------------------

def binomial_pmf(k, n, theta):
    """
    Why: answers "out of n pixels sampled from this region (flood rate
    theta), what's p(exactly k are flooded)?" -- theta comes from the
    actual data (y.mean()), not a guess.
    In: k, n (ints), theta (float in [0,1]). Out: float, Bin(k|n,theta).
    Formula: C(n,k) * theta^k * (1-theta)^(n-k).
    Checkpoint: sum(binomial_pmf(k,10,0.3) for k in range(11)) ~= 1.0
    """
    # NB: math.comb(n,k) avoids raw factorials overflowing for large n.
    n_factorial = np.prod(np.arange(1, n + 1))
    nk_factorial = np.prod(np.arange(1, (n - k) + 1))
    k_factorial = np.prod(np.arange(1, k + 1))
    n_choose_k = n_factorial / (nk_factorial * k_factorial)
    return n_choose_k * (theta**k) * (1 - theta)**(n - k)


# ---------------------------------------------------------------------------
# (B) Predictive question: build a classifier, elevation/rainfall -> flooded?
# ---------------------------------------------------------------------------

def sigmoid(a):
    """
    Why: squashes any real score into (0,1), usable as a probability --
    needed by forward(), next.
    In: a (float or ndarray). Out: same shape, values in (0,1).
    Formula: 1 / (1 + e^(-a)).
    Checkpoint: sigmoid(0)==0.5, sigmoid(100)~=1, sigmoid(-100)~=0.
    """
    return 1 / (1 + np.exp(-a))


def forward(X, w, b):
    """
    Why: the classifier itself -- turns (elevation, rainfall) into
    p(flooded). train_logistic_regression's job is to find good w, b for it.
    In: X (N,2), w (2,), b scalar. Out: (N,), p(y=1|x) per pixel.
    Formula: sigmoid(X @ w + b).
    Checkpoint: w=zeros(2), b=0 -> every output exactly 0.5.
    """
    return sigmoid(X @ w + b)


def nll_loss(y_true, y_pred_prob):
    """
    Why: one scalar saying how wrong forward()'s predictions are, across
    all N pixels -- needed before improving w, b (next: gradients).

    In:
      y_true      -- ACTUAL label per pixel, 0/1. Shape (N,).
      y_pred_prob -- forward()'s predicted p(flooded). Shape (N,).
    Out: scalar (lower = predictions matched true labels better).

    What to compute: per pixel, the log-probability the model assigned
    to what ACTUALLY happened -- log(p) if y_true=1, log(1-p) if y_true=0.
    y*log(p) + (1-y)*log(1-p) selects the right term per pixel without an
    if-statement. Negate, average over all N. Clip p to [1e-9,1-1e-9] first.

    Checkpoint: y=[1,0], p=[0.9,0.1] -> small loss (~0.1). y=[1,0], p=[0.1,0.9] -> large (~2.4).
    """

    # clip to avoid inf values due to python's flooring
    y_clipped = np.clip(y_pred_prob, 1e-9, 1-1e-9)
    # nll = - sum_n log p(y_n | x_n, theta)
    # note that p(y_n | x_n, theta) is bernoulli pmf here and if we apply the log we obtain a summation
    # in the bernouilli fromula p = y_pred_prob and y = y_true
    return - np.mean((y_true * np.log(y_clipped) + (1-y_true) * np.log(1-y_clipped)))



    raise NotImplementedError


def gradients(X, y_true, y_pred_prob):
    """
    Why: tells you which direction to move w, b to make nll_loss smaller --
    needed by the training loop, next.
    In: X (N,2), y_true (N,), y_pred_prob (N,). Out: (grad_w (2,), grad_b scalar).
    Derive on paper: d(NLL)/d(logits) = y_pred_prob - y_true, shape (N,).
    Chain-rule through logits = X@w+b to get grad_w, grad_b.
    Checkpoint: nudge one entry of w by 1e-5, confirm
    (loss_nudged - loss_orig)/1e-5 ~= corresponding grad_w entry.
    """
    # nll derivation in notes
    # tells how the per-pixel prediction error varies with the logit X@w + b
    return (y_pred_prob-y_true)@X/len(y), np.mean((y_pred_prob-y_true))




    raise NotImplementedError


def train_logistic_regression(X, y, lr=0.001, n_steps=2000):
    """
    Why: answers the predictive question -- runs forward/nll_loss/
    gradients together, repeatedly, until w, b actually separate flooded
    from dry pixels.
    In: X (N,2), y (N,), lr, n_steps. Out: (w (2,), b scalar).
    Loop n_steps times: forward -> nll_loss (print every 200) -> gradients
    -> w,b -= lr*grad.
    Note: elevation/rainfall are on very different scales -- if loss
    explodes or won't move, standardize X first (un-scale w,b after).
    Checkpoint: printed loss decreases monotonically, no NaNs.
    """
    w = np.zeros(2)
    b = 0.0

    for i in range(n_steps):
        y_pred_prob = forward(X, w, b)
        loss_func = nll_loss(y, y_pred_prob)
        if np.mod(i, 200)==0: print(f"NLL: {loss_func:.1}") 
        dw, db = gradients(X, y, y_pred_prob)
        w, b = w - lr*dw, b - lr*db

    return w, b
    raise NotImplementedError


if __name__ == "__main__":
    X, y = generate_flood_data()

    # (A) Statistical
    theta_hat = y.mean()
    print(f"empirical flood rate in this region: {theta_hat:.2f}")
    for k in range(11):
        print(f"  p(exactly {k}/10 pixels flooded) = {binomial_pmf(k, 10, theta_hat):.3f}")

    # (B) Predictive -- every function below is actually called; this will
    # raise NotImplementedError at whichever one you haven't finished yet.
    w0, b0 = np.zeros(2), 0.0
    p0 = forward(X, w0, b0)
    print(f"untrained model's avg predicted p(flood): {p0.mean():.2f} (expect 0.5)")

    loss0 = nll_loss(y, p0)
    #print(f"untrained model's loss: {loss0:.3f}")

    grad_w0, grad_b0 = gradients(X, y, p0)
    print(f"initial gradients: grad_w={grad_w0}, grad_b={grad_b0:.4f}")

    w, b = train_logistic_regression(X, y)
    print("learned w, b:", w, b)

    from sklearn.linear_model import LogisticRegression
    ref = LogisticRegression(penalty=None).fit(X, y)
    print("sklearn coef_, intercept_ (compare w[0]/w[1] slope, not raw values):",
          ref.coef_, ref.intercept_)


    flooded_elevation = X[y == 1, 0]
    non_flooded_elevation = X[y == 0, 0]
    flooded_rainfall = X[y == 1, 1]
    non_flooded_rainfall = X[y == 0, 1]

    fig, ax = plt.subplots()
    ax.scatter(non_flooded_elevation, non_flooded_rainfall, c="tab:blue", label="not flooded", alpha=0.6, s=15)
    ax.scatter(flooded_elevation, flooded_rainfall, c="tab:red", label="flooded", alpha=0.6, s=15)
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    ax.plot(xs, -(w[0] * xs + b) / w[1], "k--", label="decision boundary 1")
    ax.set_xlabel("elevation (m)"); ax.set_ylabel("rainfall (mm)"); ax.legend()
    fig.savefig("flood_classifier_result.png", dpi=150)
    print("saved flood_classifier_result.png")
    
"""
SECTION 2.2 -- Random variables (guided exercise, YOU implement the TODOs)

Theme: "Rainfall Station Explorer"
A synthetic rain gauge station reports daily rainfall (mm) for one year.
You'll treat rainfall-amount as a continuous rv, bucket it into a discrete
"rainfall category" rv, and a second station's "is it monsoon month" as a
binary rv -- then build joint/marginal/conditional distributions and moments
entirely from counting, exactly as 2.2 defines them.

Everything below marked TODO is yours. Data generation is provided so you can
focus on the probability, not the plumbing. Formulas are quoted from 2.2 for
reference -- turning them into working code is the actual exercise.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Data generation (provided -- do not need to touch)
# ---------------------------------------------------------------------------

def generate_station_data(n_days=365):
    """
    Returns:
      rainfall_mm: (n_days,) continuous rv, lognormal-ish, more rain in monsoon months
      is_monsoon:  (n_days,) binary rv, 1 for days 150-240 (roughly 3 months)
    """
    is_monsoon = np.zeros(n_days, dtype=int)
    is_monsoon[150:240] = 1
    base = rng.exponential(scale=2.0, size=n_days)
    monsoon_boost = is_monsoon * rng.exponential(scale=15.0, size=n_days)
    rainfall_mm = base + monsoon_boost
    return rainfall_mm, is_monsoon


def bucket_rainfall(rainfall_mm):
    """
    Discretize into categories: 0=none(<1mm), 1=light(1-5mm), 2=moderate(5-20mm), 3=heavy(>20mm)
    Provided -- this turns your continuous rv into a discrete rv for the joint/marginal work below.
    """
    bins = np.array([1, 5, 20])
    return np.digitize(rainfall_mm, bins)


# ---------------------------------------------------------------------------
# 2.2.1 / 2.2.2 -- discrete & continuous rv basics
# ---------------------------------------------------------------------------

def empirical_pmf(discrete_data, n_categories):
    """
    TODO: p(x) := Pr(X=x). Return an array of length n_categories where
    entry k = (count of discrete_data == k) / len(discrete_data).
    Must satisfy: 0 <= p(x) <= 1 and sum(p) == 1 (assert this at the end).
    """

    count = np.bincount(discrete_data, minlength=n_categories)
    pmf = count / count.sum() 

    print(f"p(x) integrates to {pmf.sum():.2f}")

    return pmf

    raise NotImplementedError
    

def empirical_cdf(continuous_data):
    """
    TODO: P(x) := Pr(X <= x). Return (sorted_values, cdf_values) where
    cdf_values[i] = fraction of continuous_data <= sorted_values[i].
    Hint: sorting the data and using rank/N gives you this directly --
    think about why (this *is* the definition of an empirical cdf).
    """
    continuous_data_sorted = np.sort(continuous_data)
    cdf = np.arange(1, len(continuous_data_sorted) + 1) / len(continuous_data_sorted)

    return continuous_data_sorted, cdf
    
    raise NotImplementedError


def quantile_from_cdf(sorted_values, cdf_values, q):
    """
    TODO: implement the inverse-cdf lookup (quantile function).
    Given q in [0,1], find x_q such that Pr(X <= x_q) ~= q.
    (np.searchsorted on cdf_values is one clean way -- same trick you
    used for Categorical.sample() in the 2.5 code, just inverted.)
    Use this to report the median (q=0.5) and the 25th/75th percentiles.
    """
    percentiles = np.array([0.25, 0.75])
    quartiles = sorted_values[np.searchsorted(cdf_values, percentiles, side="left")]

    print(f"The 25th percentile is {quartiles[0]:.3f}, and the 75th percentile is {quartiles[1]:.3f}.")

    return sorted_values[np.searchsorted(cdf_values, q, side="left")]


    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2.2.5 -- moments
# ---------------------------------------------------------------------------

def mean_variance_mode(discrete_data, n_categories):
    """
    TODO: compute, from the empirical_pmf you already wrote:
      mean  = E[X] = sum_x x * p(x)
      var   = V[X] = E[(X-mu)^2] = sum_x (x-mu)^2 * p(x)
      mode  = argmax_x p(x)
    Return (mean, var, mode). Do NOT use np.mean/np.var on the raw data --
    the point is to compute these FROM the pmf, since that's what
    generalizes to continuous/weighted cases later.
    """

    p = empirical_pmf(discrete_data, n_categories)
    x = np.arange(n_categories)

    mean = np.dot(discrete_data, n_categories)
    var = np.dot(discrete_data - mean, n_categories)
    mode = x[np.argmax(p[x])]

    return mean, var, mode 

    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2.2.3 -- joint / marginal / conditional (sum rule, product rule)
# ---------------------------------------------------------------------------

def joint_pmf(x_categories, y_binary, n_x_categories):
    """
    TODO: build a (n_x_categories, 2) table where entry [x,y] =
    Pr(X=x, Y=y), estimated as counts / total_N.
    """

    monsoon_idx = y_binary==1
    x_y1 = x_categories[monsoon_idx]
    x_y0 = x_categories[~monsoon_idx]
    joint = np.zeros([n_x_categories, 2])

    for x in range(n_x_categories):
        py0 = np.sum(x_y0 == x)/len(x_categories)
        py1 = np.sum(x_y1 == x)/len(x_categories)
        joint[x, 0] = py0
        joint[x, 1] = py1

    return joint

    raise NotImplementedError


def marginal_from_joint(joint, axis):
    """
    TODO: sum rule. p(X=x) = sum_y p(X=x,Y=y). Implement via summing the
    joint table over the given axis. Verify your result matches calling
    empirical_pmf directly on the relevant variable (assert np.allclose).
    """
    # We return the probability of X=x, Y=y (all combinations)
    return joint.sum(axis=axis)


    raise NotImplementedError


def conditional_from_joint(joint, given_axis):
    """
    TODO: product rule / definition of conditional probability:
      p(Y=y|X=x) = p(X=x,Y=y) / p(X=x)
    Return the conditional table. Careful with which axis you're
    conditioning on vs. summing over -- get this backwards and you'll
    silently compute the wrong direction of dependence.
    """
    # We return the probability of X=x marginalizing over Y=y (combinations of X)
    return joint.sum(axis=given_axis)

    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2.2.5.4 -- law of total expectation (self-check via two independent routes)
# ---------------------------------------------------------------------------

def check_law_of_total_expectation(rainfall_mm, is_monsoon):
    """
    TODO: verify E[X] = E[E[X|Y]] two ways:
      route A: mean of ALL rainfall_mm directly
      route B: E[X|Y=0]*p(Y=0) + E[X|Y=1]*p(Y=1)
    Print both numbers -- they should match closely (small sampling noise
    is fine, but they should agree to ~1 decimal place).
    """

    mean1 = rainfall_mm.mean()

    N = len(is_monsoon)
    monsoon_mask = is_monsoon == 1
    monsoon1_cases = is_monsoon.sum()
    monsoon0_cases = N - monsoon1_cases

    pY1 = monsoon1_cases/N
    pY0 = 1 - pY1

    ex0 = rainfall_mm[~monsoon_mask]/monsoon0_cases*pY0
    ex1 = rainfall_mm[monsoon_mask]/monsoon1_cases*pY1
    mean2 = ex0.sum() + ex1.sum()

    return print(f"Route A mean is {mean1:.1f} and route B is  {mean2:.1f} ")

    raise NotImplementedError


if __name__ == "__main__":
    rainfall_mm, is_monsoon = generate_station_data()
    rainfall_cat = bucket_rainfall(rainfall_mm)

    # Once implemented, these should all run without error:
    pmf = empirical_pmf(rainfall_cat, n_categories=4)
    sorted_vals, cdf_vals = empirical_cdf(rainfall_mm)
    median = quantile_from_cdf(sorted_vals, cdf_vals, 0.5)
    mean, var, mode = mean_variance_mode(rainfall_cat, n_categories=4)
    joint = joint_pmf(rainfall_cat, is_monsoon, n_x_categories=4)
    marg_x = marginal_from_joint(joint, axis=1)
    cond_y_given_x = conditional_from_joint(joint, given_axis=0)
    check_law_of_total_expectation(rainfall_mm, is_monsoon)


    # Suggested visual once it all works: 2x2 subplot --
    #   (1) bar chart of empirical_pmf(rainfall_cat)
    #   (2) step plot of empirical_cdf(rainfall_mm) with median/quartiles marked
    #   (3) heatmap (imshow) of the joint_pmf table
    #   (4) two overlaid histograms of rainfall_mm split by is_monsoon,
    #       with a vertical line at each conditional mean



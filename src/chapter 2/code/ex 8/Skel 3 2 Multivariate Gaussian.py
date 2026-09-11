"""
SECTION 3.2 -- Multivariate Gaussian: conditioning, Mahalanobis distance,
missing-value imputation

THE INVESTIGATION: two nearby weather stations, A and B, report correlated
temperatures (same regional weather affects both). Historical data lets
you learn their joint relationship -- then two practical questions:

  (A) Outlier detection: today's readings include a few sensor glitches --
      can you flag them automatically? A single reading being "unusual"
      isn't just about how far it is from the average -- it also depends
      on how A and B normally relate to each other. Mahalanobis distance
      is a "distance" that accounts for this: it measures how many
      standard deviations away a point is, AFTER accounting for the
      variables' correlation and different scales -- not a straight-line
      distance, a correlation-aware one.

  (B) Missing data: station B's sensor fails on some days. Can station A's
      reading (which you DO have) tell you what B's reading probably was?
      This uses GAUSSIAN CONDITIONING -- given the joint distribution of
      (A,B) and an observed value of A, what's the best-guess distribution
      for B? This exact formula, applied recursively over time, is the
      Kalman filter's measurement-update step -- what you're building here
      IS that equation, just not yet wrapped in a time loop.

Order (each step needs the ones before it):
  1. joint_mean_cov          -- standalone, generalizes 2.6's gaussian_mle to 2D
  2. mahalanobis_distance     -- needs joint_mean_cov's output
  3. detect_outliers          -- needs mahalanobis_distance
  4. gaussian_conditional_1d  -- needs joint_mean_cov's output (the Kalman-update formula)
  5. impute_missing           -- needs gaussian_conditional_1d
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(31)

TRUE_MU = np.array([15.0, 16.0])
TRUE_SIGMA = np.array([[4.0, 3.0], [3.0, 5.0]])


def generate_historical_data(n=500):
    """Provided. Out: (n,2) array, clean joint readings [stationA, stationB]."""
    return rng.multivariate_normal(TRUE_MU, TRUE_SIGMA, n)


def generate_test_data_with_outliers(n=50, n_outliers=5):
    """
    Provided. Out: (data (n,2) with a few corrupted rows, true_outlier_idx
    (n_outliers,) the actual indices that were corrupted -- for checking
    your detector against ground truth).
    """
    data = rng.multivariate_normal(TRUE_MU, TRUE_SIGMA, n)
    outlier_idx = rng.choice(n, n_outliers, replace=False)
    data[outlier_idx] += rng.normal(0, 1, (n_outliers, 2)) * 15
    return data, outlier_idx


def generate_data_with_missing_B(n=300, n_missing=30):
    """
    Provided. Out: (data_with_nan (n,2) with stationB set to NaN on some
    rows, missing_idx (n_missing,), true_B_values (n_missing,) -- the
    real values that were hidden, for checking your imputation against).
    """
    data = rng.multivariate_normal(TRUE_MU, TRUE_SIGMA, n)
    missing_idx = rng.choice(n, n_missing, replace=False)
    true_B_values = data[missing_idx, 1].copy()
    data_with_nan = data.copy()
    data_with_nan[missing_idx, 1] = np.nan
    return data_with_nan, missing_idx, true_B_values


def joint_mean_cov(data):
    """
    Why: fits THIS pair of stations' actual mean vector and covariance
    matrix from historical data -- generalizes 2.6's gaussian_mle (single
    variable) to 2 variables at once. Needed by everything below.
    In: data (N,2), each row one day's [stationA, stationB] reading.
    Out: (mu (2,), Sigma (2,2)).
    Formula: mu = mean over rows. Sigma[i,j] = mean((data[:,i]-mu[i]) * (data[:,j]-mu[j]))
    -- same idea as covariance() from 3.1, computed for all 4 (i,j) pairs
    (2x2, with Sigma[0,1]==Sigma[1,0] by construction).
    Checkpoint: on generate_historical_data(n=5000), mu should land close
    to [15,16], Sigma close to [[4,3],[3,5]].
    """
    # This derivation is done handwritten in /Theory Notes

    mu = np.mean(data, axis=0)
    centered_data = data - mu
    Sigma = np.mean(centered_data[:, :, None] * centered_data[:, None, :], axis=0)
    return mu, Sigma


def mahalanobis_distance(x, mu, Sigma):
    """
    Why: "how unusual is this point," accounting for correlation and
    different scales between the two variables -- needed by
    detect_outliers next.
    In: x (2,) or (N,2) -- one point or a batch. mu (2,). Sigma (2,2).
    Out: scalar (if x is (2,)) or (N,) (if x is (N,2)).
    Formula: d(x) = sqrt( (x-mu)^T @ inv(Sigma) @ (x-mu) ). Use
    np.linalg.inv(Sigma) -- for a 2x2 matrix this is stable and simple,
    no numerical trap here (unlike the erf/factorial cases earlier).
    Checkpoint: mahalanobis_distance(mu, mu, Sigma) == 0 (a point exactly
    at the mean is zero distance away, by definition).
    """

    sigma_inv = np.linalg.inv(Sigma)
    diff = x - mu

    dist = np.sqrt(np.einsum('...i,ij,...j', diff, sigma_inv, diff))
    return dist

def detect_outliers(data, mu, Sigma, threshold=3.0):
    """
    Why: the actual answer to investigation (A) -- flags which rows are
    unusual enough to be sensor glitches, not normal joint variation.
    In: data (N,2), mu (2,), Sigma (2,2), threshold (float).
    Out: (N,) boolean array, True where mahalanobis_distance > threshold.
    Checkpoint: on generate_test_data_with_outliers(), fitting mu,Sigma
    from a SEPARATE generate_historical_data() call first, the detected
    indices should match true_outlier_idx closely (exact match expected
    at threshold=3.0 with this synthetic data).
    """

    distances = mahalanobis_distance(data, mu, Sigma)
    return distances > threshold



def gaussian_conditional_1d(mu, Sigma, observed_value, observed_idx):
    """
    Why: THE Kalman-update formula -- given a joint (mu,Sigma) over 2
    variables and an observed value for ONE of them, returns the updated
    belief (mean and variance) for the OTHER, unobserved one. Needed by
    impute_missing next.
    In: mu (2,), Sigma (2,2), observed_value (float, the known reading),
    observed_idx (int, 0 or 1 -- WHICH of the two variables was observed).
    Out: (cond_mean (float), cond_var (float)) for the OTHER variable.
    Formula, with o=observed_idx, m=1-o (the missing one):
      cond_mean = mu[m] + Sigma[m,o]/Sigma[o,o] * (observed_value - mu[o])
      cond_var  = Sigma[m,m] - Sigma[m,o]**2 / Sigma[o,o]
    Checkpoint: if observed_value == mu[observed_idx] exactly (an
    "unsurprising" observation), cond_mean should equal mu[m] exactly
    (no new information shifts the estimate) and cond_var should be
    LESS than Sigma[m,m] (observing anything correlated always reduces
    uncertainty about the other variable, even an unsurprising value).
    """
    # Extract the relevant indices
    o = observed_idx
    m = 1 - o

    # Derived in /Theory Notes. Note that here Sigma indexes the elements of the covariance matrix a, b, c, d used in the note.

    # Compute the conditional mean and variance
    cond_mean = mu[m] + Sigma[m, o] / Sigma[o, o] * (observed_value - mu[o])
    cond_var = Sigma[m, m] - Sigma[m, o]**2 / Sigma[o, o]

    return cond_mean, cond_var


def impute_missing(data_with_nan, mu, Sigma):
    """
    Why: the actual answer to investigation (B) -- fills in station B's
    missing readings using station A's observed values on those same days.
    In: data_with_nan (N,2) with np.nan marking missing stationB entries,
    mu (2,), Sigma (2,2).
    Out: (N,2) array, same as input but NaNs replaced by their conditional
    mean estimate (column 0, stationA, is assumed never missing here).
    What to do: for each row with data_with_nan[i,1] is NaN, call
    gaussian_conditional_1d with observed_value=data_with_nan[i,0],
    observed_idx=0, and use the returned cond_mean to fill that entry.
    Checkpoint: on generate_data_with_missing_B(), compare imputed values
    to true_B_values (mean absolute error) -- should be clearly LOWER
    than the error from just filling every missing value with mu[1]
    (the naive "ignore station A" baseline).
    """
    # Create a copy of the data to avoid modifying the original
    imputed_data = data_with_nan.copy()

    # Iterate over each row
    for i in range(imputed_data.shape[0]):
        # Check if the value in column 1 (station B) is missing
        if np.isnan(imputed_data[i, 1]):
            # Call the conditional distribution function
            cond_mean, _ = gaussian_conditional_1d(mu, Sigma, observed_value=imputed_data[i, 0], observed_idx=0)
            # Fill the missing value with the conditional mean
            imputed_data[i, 1] = cond_mean

    return imputed_data


if __name__ == "__main__":
    historical = generate_historical_data()
    mu, Sigma = joint_mean_cov(historical)
    print(f"fitted mu={mu}, Sigma=\n{Sigma}")

    d0 = mahalanobis_distance(mu, mu, Sigma)
    print(f"mahalanobis_distance(mu, mu, Sigma) = {d0:.4f} (expect 0)")

    test_data, true_outlier_idx = generate_test_data_with_outliers()
    detected = detect_outliers(test_data, mu, Sigma)
    print(f"true outlier rows: {sorted(true_outlier_idx)}")
    print(f"detected rows:     {sorted(np.where(detected)[0])}")

    cond_mean, cond_var = gaussian_conditional_1d(mu, Sigma, observed_value=mu[0], observed_idx=0)
    print(f"conditional on an unsurprising A reading: mean={cond_mean:.3f} (expect ~= mu[1]={mu[1]:.3f}), "
          f"var={cond_var:.3f} (expect < Sigma[1,1]={Sigma[1,1]:.3f})")

    data_missing, missing_idx, true_B_values = generate_data_with_missing_B()
    imputed = impute_missing(data_missing, mu, Sigma)
    imputed_error = np.mean(np.abs(imputed[missing_idx, 1] - true_B_values))
    naive_error = np.mean(np.abs(mu[1] - true_B_values))
    print(f"imputation MAE: {imputed_error:.3f}  vs naive mean-fill MAE: {naive_error:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(test_data[:, 0], test_data[:, 1], c=detected, cmap="coolwarm", edgecolor="k", s=30)
    ax.scatter(*mu, marker="*", s=200, c="black", label="fitted mean")
    ax.set_xlabel("station A temp"); ax.set_ylabel("station B temp")
    ax.set_title("Outlier detection via Mahalanobis distance (red=flagged)")
    ax.legend()
    fig.savefig("multivariate_gaussian_result.png", dpi=150)
    print("saved multivariate_gaussian_result.png")
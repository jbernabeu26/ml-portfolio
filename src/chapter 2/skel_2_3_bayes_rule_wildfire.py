"""
SECTION 2.3 -- Bayes' rule (guided exercise, YOU implement the TODOs)

Theme: "Wildfire smoke detector" -- structurally identical to the book's
COVID-testing example (2.3.1): H = fire present/absent (hidden), Y = sensor
reading (thermal anomaly detected: yes/no). Same formula, different domain.

Then a stretch goal: run several sensor passes in a row and update your
belief sequentially -- this previews the exact "posterior becomes the next
prior" logic you'll need for the Kalman filter later.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

# ---------------------------------------------------------------------------
# Problem setup (provided -- these are the "known statistics" of your sensor)
# ---------------------------------------------------------------------------

PRIOR_FIRE = 0.02          # p(H=1): base rate of fire in a given region/day -- rare
SENSITIVITY = 0.90         # p(Y=1|H=1): true positive rate of the thermal sensor
SPECIFICITY = 0.95         # p(Y=0|H=0): true negative rate


# ---------------------------------------------------------------------------
# 2.3 core formula
# ---------------------------------------------------------------------------

def bayes_posterior(prior_h1, p_y1_given_h1, p_y1_given_h0, observed_y):
    """
    TODO: implement Bayes' rule exactly as in 2.3:
        p(H=1|Y=y) = p(H=1) * p(Y=y|H=1) / p(Y=y)
    where p(Y=y) is the marginal likelihood:
        p(Y=1) = p(Y=1|H=1)*p(H=1) + p(Y=1|H=0)*p(H=0)
        p(Y=0) = 1 - p(Y=1)   (since Y is binary)

    Args:
        prior_h1: p(H=1) before seeing data
        p_y1_given_h1: sensitivity
        p_y1_given_h0: false positive rate (you'll derive this from specificity)
        observed_y: 0 or 1, what the sensor actually reported

    Returns: posterior p(H=1 | Y=observed_y)

    Self-check: with PRIOR_FIRE=0.02, SENSITIVITY=0.90, SPECIFICITY=0.95,
    and observed_y=1, you should get a posterior noticeably LESS than
    0.90 despite the sensor's "90% accuracy" -- this is the classic
    base-rate effect the COVID example in 2.3.1 demonstrates. If your
    number comes back close to 0.90, you've made the same mistake most
    people make on first exposure to this -- check your marginal likelihood.
    """
    # Recall (1 - p_y1_given_h1) = p_y0_given_h1 because p(Y =y | H=h) marginalized over H needs to sum 1.
    pY1 = p_y1_given_h1*prior_h1 + p_y1_given_h0*(1-prior_h1)
    if observed_y:
        
        pXYy = p_y1_given_h1*prior_h1/pY1
    else:
        pXYy = (1-p_y1_given_h1)*prior_h1/(1-pY1)

    return pXYy

    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stretch: sequential updating (posterior becomes next prior)
# ---------------------------------------------------------------------------

def sequential_update(initial_prior, readings, p_y1_given_h1, p_y1_given_h0):
    """
    TODO: given a sequence of sensor readings (0s and 1s from repeated,
    independent passes over the same patch of ground), apply bayes_posterior
    repeatedly -- each pass's posterior becomes the next pass's prior.

    Return an array of posteriors, one after each reading (so length ==
    len(readings)), so you can plot how belief evolves over time.

    This is the same "prior_alpha -> posterior_alpha becomes next prior"
    pattern from the Dirichlet code, and the same pattern the Kalman
    filter uses every timestep -- worth noticing the recurrence.
    """
    posteriors = np.zeros([np.size(readings)])
    posteriors[0] = initial_prior
    for idx in range(1, np.size(readings)):
        posteriors[idx] = bayes_posterior(posteriors[idx-1], p_y1_given_h1, p_y1_given_h0, 1)
        
    return posteriors
    raise NotImplementedError


def simulate_sensor_passes(true_fire_state, n_passes, p_y1_given_h1, p_y1_given_h0):
    """
    Provided: simulate what a real sensor would report over n_passes,
    given the ground truth. Returns an array of 0/1 readings.
    """
    if true_fire_state:
        return rng.random(n_passes) < p_y1_given_h1
    else:
        return rng.random(n_passes) < p_y1_given_h0


if __name__ == "__main__":

    p_y1_given_h0 = 1 - SPECIFICITY  # false positive rate
    # Single-reading sanity check
    posterior_if_positive = bayes_posterior(PRIOR_FIRE, SENSITIVITY, p_y1_given_h0, observed_y=1)
    print("Posterior fire probability after ONE positive reading:", posterior_if_positive)

    # Sequential demo: simulate a real fire, see how many consecutive
    # positive-leaning readings it takes before your belief crosses 0.95
    true_readings = simulate_sensor_passes(True, n_passes=10, p_y1_given_h1=SENSITIVITY, p_y1_given_h0=p_y1_given_h0)
    posteriors = sequential_update(PRIOR_FIRE, true_readings, SENSITIVITY, p_y1_given_h0)
    plt.plot(range(1, len(posteriors)+1), posteriors, marker='o')
    plt.axhline(0.95, color='red', linestyle='--', label='95% confidence')
    plt.xlabel("sensor pass #"); plt.ylabel("p(fire | readings so far)")
    plt.legend(); plt.title("Belief about wildfire presence, updated pass by pass")
    plt.show()

    print("Fill in bayes_posterior and sequential_update, then uncomment the block above.")

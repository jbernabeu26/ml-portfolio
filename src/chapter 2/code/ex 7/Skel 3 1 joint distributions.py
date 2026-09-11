"""
SECTION 3.1 -- Joint distributions: covariance, correlation, Simpson's paradox

THE INVESTIGATION: soil moisture and vegetation health look correlated
across a whole region -- but does that correlation hold within EVERY
biome individually? And does "correlated" even mean "related" in every
case?

NDVI (Normalized Difference Vegetation Index): a number derived from
satellite imagery, roughly -1 to 1, that measures how green/healthy
vegetation is at a spot on the ground. Live plants strongly reflect
near-infrared light and absorb red light for photosynthesis; NDVI is built
from that ratio specifically to detect vegetation health this way. Higher
NDVI = denser, healthier vegetation; near 0 = bare soil/water/no plants.

Three biomes (desert, grassland, forest) each show their own soil-moisture-
to-NDVI relationship (more moisture -> healthier vegetation, WITHIN a
biome); pooling them together can tell a different, misleading story
(Simpson's paradox) -- and a separate, uncorrelated-but-clearly-dependent
example shows correlation itself can miss real structure even without
multiple groups involved.

Order (each step needs the ones before it):
  1. covariance      -- standalone
  2. correlation      -- needs covariance
  3. simpsons_paradox_check -- needs correlation, run per-biome AND pooled
  4. uncorrelated_but_dependent_demo -- needs correlation, standalone dataset
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(23)


def generate_biome_data(n_per_biome=100):
    """
    Provided. 3 biomes, each with soil_moisture and ndvi POSITIVELY
    correlated WITHIN the biome -- but biome BASELINES run in the OPPOSITE
    direction (higher-moisture biomes have lower baseline NDVI), so pooling
    all 3 together flips the sign entirely. This is the real mechanism
    behind Simpson's paradox: within-group trend and between-group trend
    can point opposite ways.
    Out: dict with keys 'desert','grassland','forest', each a
    (soil_moisture (n,), ndvi (n,)) tuple.
    """
    biomes = {}
    configs = {
        "desert":    dict(moisture_base=0.15, ndvi_base=0.75, slope=0.5),
        "grassland": dict(moisture_base=0.45, ndvi_base=0.50, slope=0.5),
        "forest":    dict(moisture_base=0.75, ndvi_base=0.25, slope=0.5),
    }
    for name, cfg in configs.items():
        moisture = cfg["moisture_base"] + rng.normal(0, 0.04, n_per_biome)
        ndvi = cfg["ndvi_base"] + cfg["slope"] * (moisture - cfg["moisture_base"]) + rng.normal(0, 0.03, n_per_biome)
        biomes[name] = (moisture, ndvi)
    return biomes


def covariance(x, y):
    """
    Why: measures whether two variables move together -- needed by
    correlation next.
    In: x (N,), y (N,), paired observations. Out: scalar.
    Formula: Cov[X,Y] = mean((x-mean(x)) * (y-mean(y))).
    Checkpoint: covariance(x,x) == variance of x (compare to np.var(x)).
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return np.mean((x - x_mean) * (y - y_mean))


def correlation(x, y):
    """
    Why: covariance's units depend on x,y's own scales, making it hard to
    compare across variable pairs. Correlation rescales it to [-1,1] --
    needed by simpsons_paradox_check and the independence demo next.
    In: x (N,), y (N,). Out: scalar in [-1,1].
    Formula: corr(X,Y) = Cov[X,Y] / (std(x) * std(y)).
    Checkpoint: compare against np.corrcoef(x,y)[0,1] -- must match closely.
    """

    cov = covariance(x, y)
    std_x = np.std(x)
    std_y = np.std(y)
    return cov / (std_x * std_y)


def simpsons_paradox_check(biomes):
    """
    Why: the actual investigation -- does each biome's WITHIN-group
    correlation agree with the POOLED (all-biomes-combined) correlation?
    In: biomes (dict from generate_biome_data). Out: (per_biome_corrs dict,
    pooled_corr float).
    What to compute: correlation() called once per biome (3 calls), then
    once more on ALL moisture/ndvi values concatenated together.
    Checkpoint: per-biome correlations should all be moderately positive
    (~0.5-0.65, matching the slope=0.5 built into each biome); pooled
    correlation should be strongly NEGATIVE (~-0.95) -- the sign genuinely
    flips, not just weakens.
    """
    print("Simpson's paradox check: per-biome correlations vs pooled correlation")
    per_biome_corrs = {}
    moisture_all = []
    ndvi_all = []
    for name, (moisture, ndvi) in biomes.items():
        per_biome_corrs[name] = correlation(moisture, ndvi)
        moisture_all.append(moisture)
        ndvi_all.append(ndvi)

    pooled_corr = correlation(np.concatenate(moisture_all), np.concatenate(ndvi_all))

    return per_biome_corrs, pooled_corr

def uncorrelated_but_dependent_demo(n=500):
    """
    Why: correlation only detects LINEAR relationships -- this shows a
    case where two variables are obviously, deterministically related,
    yet correlation reports ~0. "Uncorrelated" does not mean "independent."
    In: n (int). 
    Out: (x (n,), y (n,), corr (float)).
    What to build: x uniform on [-1,1]; y = x**2 + small noise (a clear,
    deterministic, symmetric curve -- y is entirely determined by x).
    Call your own correlation() on (x,y).
    Checkpoint: |corr| should come out small (<0.1) despite y being a
    deterministic function of x -- plot (x,y) to see the parabola directly.
    """

    x = rng.uniform(-1, 1, n)
    noise = rng.normal(0, 0.05, n)
    y = x**2 + noise    
    corr = correlation(x, y)

    return x, y, corr

if __name__ == "__main__":
    biomes = generate_biome_data()

    cov_check = covariance(biomes["forest"][0], biomes["forest"][0])
    print(f"covariance(forest_moisture, forest_moisture) = {cov_check:.5f}  (compare to np.var: {np.var(biomes['forest'][0]):.5f})")

    corr_check = correlation(biomes["forest"][0], biomes["forest"][1])
    print(f"correlation(forest_moisture, forest_ndvi) = {corr_check:.3f}")

    per_biome_corrs, pooled_corr = simpsons_paradox_check(biomes)
    print("per-biome correlations:", per_biome_corrs)
    print(f"pooled correlation (all biomes combined): {pooled_corr:.3f}")

    x_dep, y_dep, corr_dep = uncorrelated_but_dependent_demo()
    print(f"uncorrelated-but-dependent demo: correlation = {corr_dep:.3f} (expect near 0, despite y=x^2 exactly)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    colors = {"desert": "tab:orange", "grassland": "tab:green", "forest": "tab:blue"}
    for name, (m, n) in biomes.items():
        ax.scatter(m, n, s=10, alpha=0.5, label=f"{name} (r={per_biome_corrs[name]:.2f})", c=colors[name])
    ax.set_xlabel("soil moisture"); ax.set_ylabel("NDVI")
    ax.set_title(f"Simpson's paradox: pooled r={pooled_corr:.2f}, but every biome is positive")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(x_dep, y_dep, s=8, alpha=0.4)
    ax.set_xlabel("x"); ax.set_ylabel("y = x^2 + noise")
    ax.set_title(f"correlation={corr_dep:.3f}, but y is fully determined by x")
    plt.show()
    
    fig.tight_layout()
    fig.savefig("joint_distributions_result.png", dpi=150)
    print("saved joint_distributions_result.png")
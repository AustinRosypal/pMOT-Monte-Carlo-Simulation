"""Velocity quadrature and reproducible speed draws for the fine-grid scan."""

import numpy as np

from diagnose_mot_strange_velocity_sampling import STRATUM_EDGES_M_PER_S


FINE_GRID = np.arange(0., 80.125, .125)
DRAWS_PER_STRATUM = 2
PRODUCTION_LAUNCH_SEED = 2026110602
PRODUCTION_SPEED_SEED = 2026110609


def strata_for_grid(grid, vapor):
    """Compute trapezoidal flux weights and split positive nodes into strata."""
    if grid[0] != 0 or grid[-1] != STRATUM_EDGES_M_PER_S[-1]:
        raise ValueError("Velocity grid does not cover the specified strata")
    spacing = np.diff(grid)
    widths = np.r_[spacing[0] / 2, (spacing[1:] + spacing[:-1]) / 2, spacing[-1] / 2]
    weights = (widths * grid**3 * np.exp(-grid**2 / vapor.thermal_scale_m2_per_s2)
               * vapor.loading_rate_prefactor)
    ids = np.searchsorted(STRATUM_EDGES_M_PER_S[1:], grid, side="left")
    strata = []
    for group in range(len(STRATUM_EDGES_M_PER_S) - 1):
        nodes = np.flatnonzero((ids == group) & (grid > 0))
        if not len(nodes):
            raise ValueError(f"Empty velocity stratum {group}")
        strata.append((nodes, weights[nodes]))
    return strata


def draw_nodes(strata, seed, index, disc, point, draws_per_stratum=DRAWS_PER_STRATUM):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, disc, point]))
    return np.array([rng.choice(nodes, size=draws_per_stratum, p=weights / weights.sum())
                     for nodes, weights in strata])


def contribution(mask, stratum_masses, area):
    """Unbiased estimate of the discrete velocity quadrature at one launch point."""
    return float(area * np.dot(stratum_masses, np.asarray(mask, dtype=float).mean(axis=1)))

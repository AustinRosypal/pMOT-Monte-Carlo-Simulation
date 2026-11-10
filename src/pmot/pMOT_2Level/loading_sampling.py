"""Independent disc clusters and continuous Maxwell-flux speed strata."""

from types import SimpleNamespace

import numpy as np
from scipy.special import gammainc, gammaincinv
from scipy.stats import t

from ..launch_geometry import sample_incident_disc_full_sphere, sample_disc_points
from ..loading import DEFAULT_VAPOR_LOADING_PARAMETERS as VAPOR
from ..mot_strange.cell_launch_sampling import disc_area_inside_cell_m2
from .loading_scan import CELL_HALF_M, DISC_DISTANCE_M, DISC_RADIUS_M, speed_support_upper_m_s


PILOT_SEED = 2026110901
PRODUCTION_SEED = 2026110902
SPEED_EDGES = (0., 4., 8., 12., 16., 20., 24., 28., 34., 42., 55., 75., speed_support_upper_m_s())


def speed_draws(rng, points, draws):
    """Sample exactly the v^3 exp(-v²/u²) kernel, conditional on each stratum."""
    thermal = VAPOR.thermal_scale_m2_per_s2
    cdf = gammainc(2., np.square(SPEED_EDGES) / thermal)
    masses = np.diff(cdf) * .5 * thermal**2 * VAPOR.loading_rate_prefactor
    uniform = rng.random((points, len(masses), draws))
    quantile = cdf[:-1, None] + uniform * np.diff(cdf)[:, None]
    return np.sqrt(thermal * gammaincinv(2., quantile)), masses


def sampled_disc(seed, disc_index, points, draws):
    rng = np.random.default_rng(np.random.SeedSequence([seed, disc_index, 0]))
    speeds_rng = np.random.default_rng(np.random.SeedSequence([seed, disc_index, 1]))
    disc = sample_incident_disc_full_sphere(disc_index, DISC_DISTANCE_M, rng)
    area = disc_area_inside_cell_m2(disc, DISC_RADIUS_M, SimpleNamespace(cell_half_length_m=CELL_HALF_M))
    accepted = []
    while len(accepted) < points:
        point = sample_disc_points(disc, 1, DISC_RADIUS_M, False, rng)[0]
        if np.max(np.abs(point.initial_position_m)) < CELL_HALF_M:
            accepted.append(point.initial_position_m)
    speed, mass = speed_draws(speeds_rng, points, draws)
    return np.array(accepted), np.array(disc.incident_unit_vector), area, speed, mass


def disc_rate(mask, area, masses):
    """Reduce per-point, per-stratum, per-draw indicators to a disc rate."""
    return float(area * np.dot(np.asarray(mask, dtype=float).mean(axis=(0, 2)), masses))


def interval(samples):
    values = np.asarray(samples, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite independent direction discs')
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(t.ppf(.975, len(values)-1) * sd / np.sqrt(len(values)))
    return {'mean': mean, 'sd': sd, 'half_width': half if mean > 0 else None,
            'relative_half_width': half / mean if mean > 0 else None,
            'discs_nonzero': int(np.count_nonzero(values))}

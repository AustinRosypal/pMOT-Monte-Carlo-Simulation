"""Area-stratified variant for rare capture in the thin optical overlap."""

from types import SimpleNamespace

import numpy as np

from ..launch_geometry import sample_incident_disc_full_sphere
from ..mot_strange.cell_launch_sampling import disc_area_inside_cell_m2
from .loading_sampling import SPEED_EDGES, interval, speed_draws
from .loading_scan import CELL_HALF_M, DISC_DISTANCE_M, DISC_RADIUS_M


PILOT_SEED = 2026110913
PRODUCTION_SEED = 2026110914
IMPACT_EDGES_M = (0., .0002, .0005, .001, .002, .004, .008, DISC_RADIUS_M)
POINT_ALLOCATION = {11: (2, 2, 2, 2, 1, 1, 1), 22: (4, 4, 4, 4, 2, 2, 2)}


def sampled_disc(seed, disc_index, points, draws):
    """Sample every annulus and return the in-cell area weight of each point."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, disc_index, 0]))
    speed_rng = np.random.default_rng(np.random.SeedSequence([seed, disc_index, 1]))
    disc = sample_incident_disc_full_sphere(disc_index, DISC_DISTANCE_M, rng)
    lengths = SimpleNamespace(cell_half_length_m=CELL_HALF_M)
    areas = np.diff([0., *(disc_area_inside_cell_m2(disc, r, lengths) for r in IMPACT_EDGES_M[1:])])
    if np.any(areas <= 0):
        raise ValueError('Empty impact-parameter annulus in a launch disc')
    allocation = POINT_ALLOCATION[points]
    origin, u, v = map(np.asarray, (disc.center_position_m, disc.basis_u, disc.basis_v))
    positions, weights = [], []
    for inner, outer, area, n in zip(IMPACT_EDGES_M[:-1], IMPACT_EDGES_M[1:], areas, allocation, strict=True):
        accepted = 0
        for _ in range(10000):
            if accepted == n:
                break
            phi = 2*np.pi*rng.random()
            impact = np.sqrt(rng.uniform(inner*inner, outer*outer))
            r = origin + impact*(np.cos(phi)*u+np.sin(phi)*v)
            if np.max(np.abs(r)) < CELL_HALF_M:
                positions.append(r)
                weights.append(area/n)
                accepted += 1
        if accepted != n:
            raise RuntimeError('Annular rejection sampler did not fill the required point count')
    speeds, masses = speed_draws(speed_rng, points, draws)
    return np.asarray(positions), np.asarray(disc.incident_unit_vector), np.asarray(weights), speeds, masses


def disc_rate(mask, area, masses):
    """Sum the flux-weighted probability in each clipped impact annulus."""
    return float(area @ (np.asarray(mask, dtype=float).mean(axis=2) @ masses))

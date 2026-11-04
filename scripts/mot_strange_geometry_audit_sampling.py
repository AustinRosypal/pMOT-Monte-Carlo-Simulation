"""Prospective paired geometry sampling, separate from frozen physical source.

This module runs no trajectories and certifies no aperture convergence.
"""
from dataclasses import replace

import numpy as np

from pmot.mot_strange.cell_launch_sampling import disc_area_inside_cell_m2
from pmot.mot_strange.surrogate_effective_field import inside_surrogate_cell


def enlarged_disc_mixture(disc, baseline_positions_m, baseline_radius_m,
                          larger_radius_m, config, rng):
    """Pair a larger disc with existing uniform baseline samples.

    The marginal larger-disc distribution is uniform over its cell intersection
    when the baseline points are themselves uniform. Conditional on saved
    baseline positions, reused points are atoms of that conditional distribution;
    averaging over the original sampling preserves the unbiased area estimator.
    Reuse decisions must be independent of the original capture outcomes.
    """
    if not (0 < baseline_radius_m <= larger_radius_m < np.inf):
        raise ValueError('require finite nested positive radii')
    positions = np.asarray(baseline_positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError('baseline positions must be a nonempty N by 3 array')
    center = np.asarray(disc.center_position_m)
    offsets = positions-center
    if (not np.all(np.isfinite(positions))
        or np.any(np.abs(offsets @ np.asarray(disc.outward_unit_vector)) > 1e-12)
        or np.any(np.linalg.norm(offsets, axis=1) > baseline_radius_m+1e-12)
        or not all(inside_surrogate_cell(p, config) for p in positions)):
        raise ValueError('baseline positions must lie in the clipped baseline disc')
    area_small = disc_area_inside_cell_m2(disc, baseline_radius_m, config)
    area_large = disc_area_inside_cell_m2(disc, larger_radius_m, config)
    if area_large < area_small-1e-12*area_small:
        raise ValueError('nested clipped areas are inconsistent')
    probability = min(1., area_small/area_large)
    reuse = rng.random(len(positions)) < probability
    result = positions.copy()
    proposals = 0
    for index in np.flatnonzero(~reuse):
        while True:
            proposals += 1
            if proposals > max(10000, 1000*len(positions)):
                raise RuntimeError('annulus rejection sampler exhausted its proposal limit')
            angle = rng.uniform(0., 2*np.pi)
            radius = np.sqrt(rng.uniform(baseline_radius_m**2, larger_radius_m**2))
            candidate = center+radius*(np.cos(angle)*np.asarray(disc.basis_u)
                                      +np.sin(angle)*np.asarray(disc.basis_v))
            if inside_surrogate_cell(candidate, config):
                result[index] = candidate
                break
    return dict(positions_m=result, reuse_baseline= reuse,
        source_point_indices=np.where(reuse, np.arange(len(positions)), -1),
        baseline_area_m2=area_small, larger_area_m2=area_large,
        reuse_probability=probability, annulus_proposal_count=proposals)


def disc_at_distance(disc, distance_m):
    """Keep the sampled normal and basis, and move the plane center upstream.

    Generate new conditional points on this disc; do not translate old accepted
    points and assume that rejection alone preserves the new marginal density.
    """
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError('distance must be finite and positive')
    return replace(disc, center_position_m=tuple(distance_m*np.asarray(disc.outward_unit_vector)))

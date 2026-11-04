"""Uniform launch points and deterministic areas for disc/cell intersections."""

from dataclasses import replace

import numpy as np
from scipy.integrate import quad

from ..launch_geometry import DiscSample, sample_disc_points
from .surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG, SurrogateEffectiveFieldConfig,
    inside_surrogate_cell,
)


def disc_area_inside_cell_m2(
    disc: DiscSample, radius_m: float,
    config: SurrogateEffectiveFieldConfig | None = None,
) -> float:
    """Integrate the radial boundary of a disc clipped by six cell faces.

    The disc center must be strictly inside the cube. Angular breakpoints
    include every possible active-face switch and circle/face intersection.
    This avoids estimating area from a random rejection count.
    """
    cfg = config or DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("disc radius must be finite and positive")
    center = np.asarray(disc.center_position_m)
    if np.any(np.abs(center) >= cfg.cell_half_length_m):
        raise ValueError("disc center must be strictly inside the cell")
    basis = np.column_stack((disc.basis_u, disc.basis_v))
    normals = np.vstack((basis, -basis))
    distances = np.r_[cfg.cell_half_length_m - center,
                      cfg.cell_half_length_m + center]
    breaks = [0.0, 2.0 * np.pi]

    def add_roots(a, b, value=0.0):
        amplitude = np.hypot(a, b)
        if amplitude == 0 or abs(value) > amplitude:
            return
        phase = np.arctan2(b, a)
        angle = np.arccos(np.clip(value / amplitude, -1.0, 1.0))
        breaks.extend(((phase - angle) % (2*np.pi),
                       (phase + angle) % (2*np.pi)))

    for i, (a, b) in enumerate(normals):
        add_roots(a, b, distances[i] / radius_m)
        for j in range(i):
            coefficients = distances[i] * normals[j] - distances[j] * normals[i]
            add_roots(*coefficients)

    def radial_area(angle):
        projections = normals @ np.asarray((np.cos(angle), np.sin(angle)))
        positive = projections > 0
        maximum_radius = min(radius_m, float(np.min(
            distances[positive] / projections[positive]
        )))
        return 0.5 * maximum_radius**2

    area = np.pi * radius_m**2
    result, error = quad(radial_area, 0.0, 2.0*np.pi,
                         points=np.unique(breaks), epsabs=area*1e-11,
                         epsrel=1e-11, limit=400)
    if error > area * 1e-8:
        raise RuntimeError("cell-clipped disc area did not converge")
    return min(area, max(0.0, float(result)))


def sample_disc_points_inside_cell(
    disc: DiscSample, count: int, radius_m: float, rng: np.random.Generator,
    config: SurrogateEffectiveFieldConfig | None = None,
):
    """Retry uniform-area proposals until exactly count in-cell points exist.

    Returns (accepted_points, allowed_area_m2, proposal_count). Directions are
    never rejected; all accepted velocities remain parallel to the disc normal.
    """
    if count <= 0:
        raise ValueError("point count must be positive")
    cfg = config or DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    area = disc_area_inside_cell_m2(disc, radius_m, cfg)
    accepted = []
    proposed = 0
    while len(accepted) < count:
        if proposed >= max(10000, 1000 * count):
            raise RuntimeError("cell launch rejection sampler exhausted its proposal limit")
        candidate = sample_disc_points(disc, 1, radius_m, False, rng)[0]
        proposed += 1
        if inside_surrogate_cell(candidate.initial_position_m, cfg):
            accepted.append(replace(candidate, point_index=len(accepted)))
    return accepted, area, proposed

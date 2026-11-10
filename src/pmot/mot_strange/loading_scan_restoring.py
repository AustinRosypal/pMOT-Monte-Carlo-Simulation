"""Check whether a localized endpoint belongs to a stable off-core pocket."""

import numpy as np
from scipy.optimize import root

from . import accelerated_surrogate as solver
from .loading_scan_retention import CORE_RADIUS_M, OFF_CORE, observe


NEIGHBOURHOOD_M = 1e-3
ROOT_RESIDUAL_M_PER_S2 = 1e-4
CHECK_TIME_S = .2


def stability_spectra(center, data):
    """Return the phase-space Jacobian spectra at two difference scales."""
    axis = np.array([0., 0., 1.])
    equilibrium = np.r_[center, np.zeros(3)]

    def derivative(state):
        return np.r_[state[3:], solver.acceleration(state[:3], state[3:], axis, data)]

    spectra = []
    for refinement in (1., 2.):
        steps = np.r_[np.full(3, 1e-7), np.full(3, 1e-4)] / refinement
        jacobian = np.column_stack([
            (derivative(equilibrium + delta) - derivative(equilibrium - delta)) / (2 * step)
            for delta, step in zip(np.diag(steps), steps, strict=True)
        ])
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("nonfinite local Jacobian")
        spectra.append(np.linalg.eigvals(jacobian))
    return spectra


def stable_spectra(spectra):
    abscissae = [float(max(spectrum.real)) for spectrum in spectra]
    uncertainty = abs(abscissae[0] - abscissae[1])
    return max(abscissae) + 2 * uncertainty < 0


def check_off_core_pocket(observation, data):
    """Check a nearby equilibrium, local stability, and return under perturbation.

    The result is an affirmative witness when all checks pass. An unsuccessful
    search leaves the trajectory in the finite-window localized category.
    """
    if observation[0] != OFF_CORE:
        raise ValueError("off-core pocket check requires a localized observation")
    end, velocity, anchor = observation[2], observation[3], observation[5]
    axis = np.array([0., 0., 1.])
    zero = np.zeros(3)
    wall = data[11][15]
    evidence = {"status": "no_nearby_equilibrium"}

    def accel(position, velocity):
        return solver.acceleration(position, velocity, axis, data)

    # The endpoint and the measured window must fit in the tested pocket.
    if observation[8] > NEIGHBOURHOOD_M:
        evidence["status"] = "window_wider_than_tested_neighbourhood"
        return evidence
    def local_residual(offset):
        return accel(end + NEIGHBOURHOOD_M * offset, zero)

    result = root(local_residual, zero)
    center = end + NEIGHBOURHOOD_M * result.x
    residual = np.linalg.norm(accel(center, zero))
    evidence.update(center_m=center.tolist(), acceleration_residual_m_per_s2=float(residual))
    if (
        not np.isfinite(residual) or residual > ROOT_RESIDUAL_M_PER_S2
        or np.linalg.norm(center - end) >= NEIGHBOURHOOD_M
        or np.linalg.norm(center - anchor) + observation[8] >= NEIGHBOURHOOD_M
        or np.linalg.norm(center) <= CORE_RADIUS_M + NEIGHBOURHOOD_M
        or np.max(np.abs(center)) >= wall - NEIGHBOURHOOD_M
        or np.linalg.norm(solver.field(center, data)) <= data[10][3]
    ):
        return evidence

    try:
        spectra = stability_spectra(center, data)
    except (ValueError, np.linalg.LinAlgError) as error:
        evidence["status"] = "local_jacobian_failed"
        evidence["local_jacobian_error"] = str(error)
        return evidence
    evidence["spectral_abscissae_per_s"] = [float(max(spectrum.real)) for spectrum in spectra]
    if not stable_spectra(spectra):
        evidence["status"] = "no_converged_local_stability"
        return evidence

    starts = [(end, velocity)]
    starts.extend(
        (center + shift, zero)
        for shift in np.concatenate((np.eye(3), -np.eye(3))) * NEIGHBOURHOOD_M
    )
    returns = []
    for position, speed in starts:
        for refinement in (1., 2.):
            try:
                trajectory = observe(position, speed, data, CHECK_TIME_S, .05, refinement)
            except (ValueError, FloatingPointError) as error:
                evidence["status"] = "return_probe_failed"
                evidence["return_probe_error"] = str(error)
                return evidence
            if (
                trajectory[0] != OFF_CORE
                or np.linalg.norm(trajectory[5] - center) + trajectory[8] > NEIGHBOURHOOD_M / 2
            ):
                evidence["status"] = "no_converged_return"
                return evidence
            returns.append({
                "refinement": refinement,
                "start_position_m": position.tolist(),
                "start_velocity_m_per_s": speed.tolist(),
                "endpoint_m": trajectory[2].tolist(),
                "window_radius_bound_m": float(
                    np.linalg.norm(trajectory[5] - center) + trajectory[8]
                ),
            })
    evidence["return_probes"] = returns
    evidence["status"] = "stable_pocket_and_return_verified"
    return evidence

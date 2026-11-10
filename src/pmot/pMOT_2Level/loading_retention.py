"""Dual-resolution trajectory classification and off-core equilibrium witnesses."""

import numpy as np
from scipy.optimize import root

from .loading_dynamics import ESCAPED, LOCALIZED_OFF_CORE, NUMERICAL, UNRESOLVED, local, shoot
from .loading_scan import CELL_HALF_M, CORE_M


VERIFIED_OFF_CORE = 5
STEP_CAP_S = 5e-6
HORIZON_S = 2.
AXIS = np.array([0., 0., 1.])
ZERO = np.zeros(3)


def pocket_witness(end, velocity, data):
    """Require a nearby force zero, stable 6D spectra, and perturbed return."""
    def accel(r, v):
        return local(r, v, AXIS, *data)[0]

    length = .001
    # The solver's trial may leave the local basin; reject distant zeros.
    solution = root(lambda z: accel(end + length*z, ZERO), ZERO)
    center = end + length * solution.x
    residual = float(np.linalg.norm(accel(center, ZERO)))
    evidence = {'center_m': center.tolist() if np.isfinite(center).all() else None,
                'residual_m_s2': residual if np.isfinite(residual) else None, 'status': 'no_local_force_zero'}
    if (not np.isfinite(residual) or residual > 1e-4 or np.linalg.norm(center-end) > .0002
            or np.linalg.norm(center) <= CORE_M + length or np.max(np.abs(center)) >= CELL_HALF_M - length):
        return evidence
    # When the differential Stark vector vanishes the fallback axis can make
    # a tensor branch basis-dependent; demand a consistent force at the zero.
    if any(np.linalg.norm(local(center, ZERO, axis, *data)[0]) > 1e-4 for axis in np.eye(3)):
        evidence['status'] = 'fallback_axis_dependent'
        return evidence

    def derivative(state):
        return np.r_[state[3:], accel(state[:3], state[3:])]

    phase = np.r_[center, ZERO]
    spectra = []
    for refinement in (1, 2):
        h = np.array([1e-7]*3 + [1e-4]*3) / refinement
        jac = np.column_stack([(derivative(phase+d) - derivative(phase-d))/(2*size)
                               for d, size in zip(np.diag(h), h, strict=True)])
        if not np.isfinite(jac).all():
            evidence['status'] = 'nonfinite_jacobian'
            return evidence
        spectra.append(float(max(np.linalg.eigvals(jac).real)))
    evidence['spectral_abscissae_s_inv'] = spectra
    if max(spectra) + 2*abs(spectra[0]-spectra[1]) >= 0.:
        evidence['status'] = 'no_converged_local_stability'
        return evidence
    evidence['status'] = 'no_converged_perturbed_return'
    offsets = np.r_[np.eye(3), -np.eye(3)]
    starts = [(end, velocity)] + [(center+.00025*d, ZERO) for d in offsets]
    starts += [(center, .05*d) for d in offsets]
    for r, v in starts:
        for step in (STEP_CAP_S, STEP_CAP_S/2):
            label, tail_r, _, _ = shoot(r, v, step, .5, data)
            if label != LOCALIZED_OFF_CORE or np.linalg.norm(tail_r-center) > .0001:
                return evidence
    evidence['status'] = 'verified'
    return evidence


def classify(r, v, data):
    """Require matching full-horizon classifications at consecutive step caps."""
    coarse, fine = [shoot(r, v, STEP_CAP_S / scale, HORIZON_S, data) for scale in (1, 2)]
    attempts = [coarse[0], fine[0]]
    if coarse[0] != fine[0]:
        coarse, fine = [shoot(r, v, STEP_CAP_S / scale, HORIZON_S, data) for scale in (4, 8)]
        attempts.extend((coarse[0], fine[0]))
    if coarse[0] != fine[0] or fine[0] == NUMERICAL:
        return UNRESOLVED, attempts, None
    if fine[0] == LOCALIZED_OFF_CORE:
        if np.linalg.norm(coarse[1] - fine[1]) > .0001:
            return LOCALIZED_OFF_CORE, attempts, {'status': 'off_core_endpoints_disagree'}
        witness = pocket_witness(fine[1], fine[2], data)
        if witness['status'] == 'verified':
            return VERIFIED_OFF_CORE, attempts, witness
        return LOCALIZED_OFF_CORE, attempts, witness
    return fine[0], attempts, None

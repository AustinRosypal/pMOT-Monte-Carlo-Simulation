"""Bound the off-core diagnostic search after an invalid out-of-cell probe."""

from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

from pmot.mot_strange import loading_scan_restoring as restoring


_root = restoring.root
_check = restoring.check_off_core_pocket


class NoAdmissibleSearchBox(Exception):
    pass


def check_pocket(observation, data):
    """Preserve the original witness gates and recover an out-of-domain root probe."""
    endpoint = observation[2]
    wall = data[11][15]
    scale = restoring.NEIGHBOURHOOD_M
    fallback = {}

    def root_with_local_fallback(fun, initial):
        failed_offset = None

        def tracked(offset):
            nonlocal failed_offset
            try:
                return fun(offset)
            except ValueError:
                failed_offset = np.asarray(offset).copy()
                raise

        try:
            return _root(tracked, initial)
        except ValueError as error:
            if (str(error) != "no unique ground steady state" or failed_offset is None
                    or np.linalg.norm(failed_offset) <= 1.
                    or np.max(np.abs(endpoint + scale * failed_offset)) <= wall):
                raise
            fallback["rejected_probe_position_m"] = (endpoint + scale * failed_offset).tolist()
            fallback["rejected_probe_error"] = str(error)
            # Any center accepted by the existing gate lies at least scale
            # inside the wall and at most scale from the trajectory endpoint.
            margin = wall - scale
            low = np.maximum(-1., (-margin - endpoint) / scale)
            high = np.minimum(1., (margin - endpoint) / scale)
            if np.any(high <= low):
                raise NoAdmissibleSearchBox from error
            bounded = least_squares(fun, np.clip(initial, low, high), bounds=(low, high),
                                    ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=150)
            fallback["bounded_solver_status"] = int(bounded.status)
            fallback["bounded_solver_evaluations"] = int(bounded.nfev)
            return bounded

    with patch.object(restoring, "root", root_with_local_fallback):
        try:
            result = _check(observation, data)
        except NoAdmissibleSearchBox:
            result = {"status": "no_nearby_equilibrium"}
    if fallback:
        result["equilibrium_search_fallback"] = fallback
    return result

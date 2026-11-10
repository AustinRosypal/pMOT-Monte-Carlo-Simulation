"""Finite-window capture observations for the symmetric surrogate campaign."""

import numpy as np
from numba import njit

from ..mot_multilevel.adaptive import embedded_step
from . import accelerated_surrogate as solver


WALL, CORE, OFF_CORE, OTHER = 0, 1, 2, 3
CORE_RADIUS_M = .002
OFF_CORE_EXCURSION_M = .001
OBSERVATION_WINDOW_S = .100
CANDIDATE_HORIZON_S = .400
POSITION_TOLERANCE_M = 1e-5
ADAPTIVE_PAIRS = ((1., 2.), (4., 8.))
FIXED_PAIRS_S = ((5e-6, 2.5e-6), (1.25e-6, .625e-6), (.3125e-6, .15625e-6))


@njit(cache=False)
def observe(r0, v0, data, horizon, window, refinement, fixed_step=0.):
    """Observe every accepted state and the exact start and end of the final window.

    Returns a category, actual duration, endpoint position and velocity,
    time of the first core event, window start position, minimum/maximum
    radius and maximum excursion from the window start. Wall is terminal.
    """
    if not (0 < window < horizon and refinement >= 1 and fixed_step >= 0):
        raise ValueError("invalid retention observation controls")
    y = np.concatenate((r0, v0))
    axis = np.array([0., 0., 1.])
    t = 0.
    h = 5e-6
    window_start = horizon - window
    anchor = np.zeros(3)
    observing = False
    min_radius, max_radius, excursion = np.inf, 0., 0.
    inside = np.linalg.norm(r0) <= CORE_RADIUS_M
    entries = int(inside)
    since = 0. if inside else -1.
    first_core = -1.
    max_steps = int(np.ceil(horizon / fixed_step)) + 4 if fixed_step > 0 else 2_000_000
    for _ in range(max_steps):
        r, v = y[:3], y[3:]
        if solver.escaped(r, v, data):
            return WALL, t, r, v, first_core, anchor, min_radius, max_radius, excursion
        radius = np.linalg.norm(r)
        now_inside = radius <= CORE_RADIUS_M
        if now_inside and not inside:
            entries += 1
            since = t
        elif not now_inside:
            since = -1.
        inside = now_inside
        if first_core < 0 and (entries >= 2 or (since >= 0 and t - since >= .005)):
            first_core = t
        if t >= window_start:
            if not observing:
                anchor = r.copy()
                observing = True
            min_radius = min(min_radius, radius)
            max_radius = max(max_radius, radius)
            excursion = max(excursion, np.linalg.norm(r - anchor))
        if t >= horizon:
            category = OTHER
            if first_core >= 0 and max_radius <= CORE_RADIUS_M:
                category = CORE
            elif min_radius > CORE_RADIUS_M and excursion <= OFF_CORE_EXCURSION_M:
                category = OFF_CORE
            return category, t, r, v, first_core, anchor, min_radius, max_radius, excursion

        limit = horizon if observing else window_start
        if fixed_step > 0:
            step = min(fixed_step, limit - t)
            r_next, v_next, axis = solver.step(r, v, axis, step, data)
            y = np.concatenate((r_next, v_next))
        else:
            speed = np.linalg.norm(v)
            event_cap = max(.0001, abs(radius - CORE_RADIUS_M) / 2) / max(speed, 1e-12)
            cap = (.0001 if radius < .003 else .0005) / refinement
            step = min(h, cap, event_cap / refinement, limit - t,
                       solver.wall_step_limit(r, v, data) / refinement)
            candidate, error = embedded_step(y, axis, step, data, solver.acceleration)
            norm = 0.
            for i in range(6):
                absolute = 1e-8 if i < 3 else 1e-5
                scale = (absolute + 1e-5 * max(abs(y[i]), abs(candidate[i]))) / refinement**2
                norm = max(norm, abs(error[i]) / scale)
            if not np.isfinite(norm):
                raise ValueError("nonfinite adaptive trajectory error")
            factor = 5. if norm == 0 else min(5., max(.2, .9 * norm**(-.2)))
            h = step * factor
            if norm > 1.:
                if h < 1e-14:
                    raise ValueError("adaptive trajectory step underflow")
                continue
            y = candidate
            b = solver.field(y[:3], data)
            magnitude = np.linalg.norm(b)
            if magnitude > data[10][3]:
                axis = b / magnitude
        next_t = limit if step == limit - t else t + step
        if next_t <= t or not np.all(np.isfinite(y)):
            raise ValueError("retention trajectory did not advance to a finite state")
        t = next_t
    return -1, t, y[:3], y[3:], first_core, anchor, min_radius, max_radius, excursion


def agreed_category(a, b):
    """Check the discrete category and the spatial evidence at finite horizons."""
    if a[0] < 0 or a[0] != b[0]:
        return None
    if a[0] == WALL:
        return WALL
    if a[1] != b[1] or np.linalg.norm(a[2] - b[2]) > POSITION_TOLERANCE_M:
        return None
    if np.linalg.norm(a[5] - b[5]) > POSITION_TOLERANCE_M:
        return None
    if np.max(np.abs(np.asarray(a[6:]) - np.asarray(b[6:]))) > POSITION_TOLERANCE_M:
        return None
    return a[0]


def observe_grid(r, direction, speeds, data, *, horizon, window=OBSERVATION_WINDOW_S):
    """Resolve a direct velocity mask and retain every adaptive/fixed attempt."""
    observations = {}
    fixed_nodes = set()
    codes = np.full(len(speeds), -1, dtype=np.int8)

    def attempt(index, method, resolution):
        speed = speeds[index]
        key = (index, method, resolution)
        if key not in observations:
            try:
                observations[key] = observe(
                    r, speed * direction, data, horizon, window,
                    resolution if method == "adaptive" else 1.,
                    resolution if method == "fixed" else 0.,
                )
            except (ValueError, FloatingPointError) as error:
                observations[key] = f"{type(error).__name__}: {error}"
        return observations[key]

    def agree(index, method, a, b):
        first = attempt(index, method, a)
        second = attempt(index, method, b)
        if isinstance(first, str) or isinstance(second, str):
            return None
        return agreed_category(first, second)

    def resolve_fixed(index):
        fixed_nodes.add(index)
        for a, b in FIXED_PAIRS_S:
            category = agree(index, "fixed", a, b)
            if category is not None:
                return category
        return -1

    for index in range(len(speeds)):
        for a, b in ADAPTIVE_PAIRS:
            category = agree(index, "adaptive", a, b)
            if category is not None:
                codes[index] = category
                break
        # A disagreement at the initial adaptive pair still receives fixed proof.
        if (codes[index] < 0 or speeds[index] == 0
                or (index, "adaptive", ADAPTIVE_PAIRS[1][0]) in observations):
            codes[index] = resolve_fixed(index)
    while True:
        capture_labels = np.where(np.isin(codes, (CORE, OFF_CORE)), codes, 0)
        edges = np.flatnonzero(capture_labels[:-1] != capture_labels[1:])
        neighbours = {
            index for edge in edges for index in range(max(0, edge - 1), min(len(speeds), edge + 3))
        } - fixed_nodes
        if not neighbours:
            break
        for index in sorted(neighbours):
            codes[index] = resolve_fixed(index)
    return codes, observations

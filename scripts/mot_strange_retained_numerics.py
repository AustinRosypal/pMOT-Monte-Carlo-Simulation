"""Prospective fixed-step observation revision; frozen force/step stay unchanged.

This module is not installed in the failed campaign and does not reinterpret
any saved result. Each propagated endpoint is observed, including the actual
final endpoint after a floating-point remainder step.
"""
import numpy as np
from numba import njit
from pmot.mot_strange import accelerated_surrogate as frozen


@njit(cache=False)
def propagate(r0, v0, dt, duration, data, trace_times, step, escaped, field):
    if not (dt > 0 and np.isfinite(dt) and duration >= 0 and np.isfinite(duration)):
        raise ValueError('finite positive step and nonnegative duration required')
    r, v, axis = r0.copy(), v0.copy(), np.array([0., 0., 1.])
    was_inside = np.linalg.norm(r) <= .002
    entries = int(was_inside)
    since = 0. if was_inside else -1.
    first, maximum, t = -1., 0., 0.
    trace = np.empty((len(trace_times), 13))
    observed = 0
    while True:
        radius = np.linalg.norm(r)
        inside = radius <= .002
        if inside and not was_inside:
            entries += 1
            since = t
        elif not inside:
            since = -1.
        was_inside = inside
        if first < 0 and (entries >= 2 or (since >= 0 and t-since >= .005)):
            first = t
        if t >= duration-.025:
            maximum = max(maximum, radius)
        while observed < len(trace_times) and t >= trace_times[observed]:
            trace[observed, 0] = t
            trace[observed, 1:4] = r
            trace[observed, 4:7] = v
            trace[observed, 7:10] = axis
            trace[observed, 10] = np.linalg.norm(field(r, data))
            trace[observed, 11] = first
            trace[observed, 12] = maximum
            observed += 1
        if escaped(r, v, data):
            return 0, t, r, v, first, maximum, trace[:observed]
        if t >= duration:
            return -1, t, r, v, first, maximum, trace[:observed]
        h = min(dt, duration-t)
        next_t = min(t+dt, duration)
        if next_t <= t:
            raise ValueError('fixed trajectory time did not advance')
        r, v, axis = step(r, v, axis, h, data)
        t = next_t
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise ValueError('nonfinite surrogate trajectory')


def classify(r, v, dt, duration, data, early_exit=False):
    if early_exit:
        raise ValueError('Retained observation cannot exit at a core event')
    return propagate(r, v, dt, duration, data, np.empty(0),
                     frozen.step, frozen.escaped, frozen.field)[:6]


def traced(r, v, dt, duration, data, trace_times):
    return propagate(r, v, dt, duration, data, np.asarray(trace_times),
                     frozen.step, frozen.escaped, frozen.field)

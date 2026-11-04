"""Surrogate environment and trajectories using the public multilevel kernel.

The physical Section-12 stimulated coefficients, population solve and net
photon-momentum force are imported, never duplicated or interpolated here.
"""
import numpy as np
from numba import njit

from ..configuration import RB87_MASS_KG
from ..mot_multilevel.accelerated import prepare as prepare_atomic, observable_with_field
from ..mot_multilevel.adaptive import integrate_local_environment
from .surrogate_effective_field import calibrated_surrogate_field_config


def prepare(config=None, field_config=None):
    cfg = field_config or calibrated_surrogate_field_config()
    field = np.array([cfg.wavelength_m, cfg.forward_waist_position_m,
        cfg.forward_rayleigh_range_m, cfg.return_waist_position_m,
        cfg.return_rayleigh_range_m, cfg.x_path_power_w, cfg.y_path_power_w,
        cfg.z_path_power_w, cfg.helicity_positive_x, cfg.helicity_negative_x,
        cfg.helicity_positive_y, cfg.helicity_negative_y,
        cfg.helicity_positive_z, cfg.helicity_negative_z,
        cfg.kappa_t_per_w_per_m2, cfg.cell_half_length_m])
    return (*prepare_atomic(config, external_field=True), field)


@njit(cache=True)
def field(position, data):
    p = data[11]
    out = np.zeros(3)
    if np.max(np.abs(position)) > p[15]:
        return out
    radius2 = np.dot(position, position)
    for a in range(3):
        q = position[a]
        rho2 = max(0.0, radius2-q*q)
        fden = (q-p[1])**2+p[2]**2
        rden = (q-p[3])**2+p[4]**2
        forward = 2*p[5+a]*p[2]/(p[0]*fden)*np.exp(-2*np.pi*p[2]*rho2/(p[0]*fden))
        returned = 2*p[5+a]*p[4]/(p[0]*rden)*np.exp(-2*np.pi*p[4]*rho2/(p[0]*rden))
        out[a] = p[14]*(p[8+2*a]*forward-p[9+2*a]*returned)
    return out


@njit(cache=True)
def observable(r, v, axis, data):
    return observable_with_field(r, v, axis, data[:11], field(r, data))


@njit(cache=True)
def acceleration(r, v, axis, data):
    force, _, _, _ = observable(r, v, axis, data)
    out = force/RB87_MASS_KG
    out[2] -= data[10][4]
    return out


@njit(cache=True)
def escaped(r, v, data):
    return np.max(np.abs(r)) > data[11][15]


@njit(cache=True)
def wall_step_limit(r, v, data):
    # Resolve approach to a physical wall before terminating an accepted step.
    distance = max(1e-7, data[11][15]-np.max(np.abs(r)))
    return distance/(2*max(1e-8, np.linalg.norm(v)))


@njit(cache=True)
def integrate(r0, v0, duration, data, refinement=1., retain=False):
    return integrate_local_environment(r0, v0, duration, data, acceleration,
        field, escaped, refinement, retain, wall_step_limit)


@njit(cache=True)
def step(r, v, axis, dt, data):
    a1 = acceleration(r, v, axis, data)
    v2 = v+dt/2*a1
    a2 = acceleration(r+dt/2*v, v2, axis, data)
    v3 = v+dt/2*a2
    a3 = acceleration(r+dt/2*v2, v3, axis, data)
    v4 = v+dt*a3
    a4 = acceleration(r+dt*v3, v4, axis, data)
    rn = r+dt/6*(v+2*v2+2*v3+v4)
    vn = v+dt/6*(a1+2*a2+2*a3+a4)
    b = field(rn, data)
    mag = np.linalg.norm(b)
    return rn, vn, axis if mag <= data[10][3] else b/mag


@njit(cache=True)
def classify(r0, v0, dt, duration, data, early_exit=True):
    """Central-core capture, wall escape, or explicitly indeterminate timeout."""
    r, v, axis = r0.copy(), v0.copy(), np.array([0., 0., 1.])
    was_inside = np.linalg.norm(r) <= .002
    entries = int(was_inside)
    since = 0. if was_inside else -1.
    first = -1.
    maximum = 0.
    t = 0.
    for _ in range(int(np.ceil(duration/dt))+1):
        radius = np.linalg.norm(r)
        if escaped(r, v, data):
            return 0, t, r, v, first, maximum
        inside = radius <= .002
        if inside and not was_inside:
            entries += 1
            since = t
        elif not inside:
            since = -1.
        was_inside = inside
        trapped = entries >= 2 or (since >= 0 and t-since >= .005)
        if trapped and first < 0:
            first = t
        if trapped and early_exit:
            return 1, t, r, v, first, maximum
        if t >= duration-.025:
            maximum = max(maximum, radius)
        if t >= duration-1e-15:
            break
        r, v, axis = step(r, v, axis, min(dt, duration-t), data)
        t = min(t+dt, duration)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise ValueError('nonfinite surrogate trajectory')
    return -1, t, r, v, first, maximum

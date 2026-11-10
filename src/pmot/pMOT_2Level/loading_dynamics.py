"""Compiled Cartesian specialization of the effective two-level pMOT force."""

from math import exp, pi, sqrt

import numpy as np
from numba import njit

from ..configuration import (
    HBAR_J_S, PLANCK_CONSTANT_J_S, RB87_MASS_KG, SPEED_OF_LIGHT_M_PER_S,
    VACUUM_PERMITTIVITY_F_PER_M,
)
from .loading_scan import CELL_HALF_M, CORE_M, FINAL_SPEED_M_S, cooling_components, simple_config


ESCAPED, CORE, LOCALIZED_OFF_CORE, UNRESOLVED, NUMERICAL = range(5)


def prepare(trapping, table, powers):
    """Pack validated symmetric circular beams; no real field or repump input."""
    cooling = cooling_components()
    simple = simple_config()
    geometry = np.empty((6, 4))
    for i, beam in enumerate(trapping):
        axis, sign = i//2, (-1)**i
        assert beam.direction == tuple(np.eye(3)[axis] * sign)
        assert beam.helicity == cooling[i].circular_polarization
        assert beam.helicity in ('sigma+', 'sigma-')
        geometry[i] = (beam.waist_position_m[axis], beam.waist_radius_m**2,
                       1 / beam.rayleigh_range_m**2,
                       2 / pi * powers[axis] * beam.power_per_incident_path_watt)
    wave = cooling[0].wavelength_m
    radius = cooling[0].intensity_beam.beam_radius_m
    for beam in cooling:
        assert beam.intensity_beam.reference_position_m == (0., 0., 0.)
        assert beam.wavelength_m == wave and beam.intensity_beam.beam_radius_m == radius
        assert beam.detuning_hz == simple.cooling_detuning_hz
    params = (
        trapping[0].wavelength_m * 1e9, 1 / wave, radius**2,
        (wave / (pi*radius**2))**2,
        2 * cooling[0].intensity_beam.power_w / (pi * simple.saturation_intensity_w_per_m2),
        simple.cooling_detuning_hz, 2 / simple.linewidth_hz,
        HBAR_J_S * (2*pi / wave) * simple.maximum_scattering_rate_per_s / RB87_MASS_KG,
        simple.gravity_acceleration_m_per_s2[2],
    )
    # Store -2 Δα/(h c ε0), so values multiplied by local intensity give Hz.
    shifts = np.column_stack((table.scalar_si, table.vector_si, table.tensor_si))
    shifts *= -2 / (PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M)
    return params, geometry, table.wavelengths_nm, shifts


@njit(cache=True, inline='always')
def _r2(r, axis):
    return r[(axis+1)%3]**2 + r[(axis+2)%3]**2


@njit(cache=True)
def local(r, v, previous_axis, params, geometry, wave_nodes, coeff):
    """Return acceleration and the continued axis, exactly as in the reference."""
    wave, inv_cooling, waist2, rayleigh_inv2, sat_scale, delta, inv_width, a_scale, gravity = params
    vector = np.zeros(3)
    tensor = np.zeros(3)
    sat = np.zeros(3)
    total_scale = 0.
    scalar = 0.
    for axis in range(3):
        transverse2 = _r2(r, axis)
        cool_w2 = waist2 * (1 + r[axis]**2 * rayleigh_inv2)
        sat[axis] = sat_scale / cool_w2 * exp(-2 * transverse2 / cool_w2)
        helicity = 1 if axis != 2 else -1
        for j in range(2):
            i = 2*axis + j
            sign = 1 if j == 0 else -1
            waist_pos, w02, inv_zr2, power_scale = geometry[i]
            w2 = w02 * (1 + (r[axis] - waist_pos)**2 * inv_zr2)
            intensity = power_scale / w2 * exp(-2 * transverse2 / w2)
            seen = wave / (1 - sign * v[axis] / SPEED_OF_LIGHT_M_PER_S)
            if not wave_nodes[0] <= seen <= wave_nodes[-1]:
                return np.full(3, np.nan), previous_axis
            hi = min(len(wave_nodes)-1, max(1, np.searchsorted(wave_nodes, seen)))
            fraction = (seen-wave_nodes[hi-1]) / (wave_nodes[hi]-wave_nodes[hi-1])
            scalar += intensity * (coeff[hi-1, 0] + fraction * (coeff[hi, 0]-coeff[hi-1, 0]))
            vj = intensity * (coeff[hi-1, 1] + fraction * (coeff[hi, 1]-coeff[hi-1, 1]))
            tensor[axis] += intensity * (coeff[hi-1, 2] + fraction * (coeff[hi, 2]-coeff[hi-1, 2]))
            vector[axis] -= vj * helicity * sign
            total_scale += abs(vj)
    vector_mag = sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    axis_hat = previous_axis
    if vector_mag > 1e-12 * total_scale:
        axis_hat = vector / vector_mag
    common = scalar
    projected = 0.
    for axis in range(3):
        common += .5 * tensor[axis] * (1.5*(1-axis_hat[axis]**2) - 1)
        projected += axis_hat[axis] * vector[axis]
    base = 1 + 2 * (sat[0] + sat[1] + sat[2])
    accel = np.empty(3)
    for axis in range(3):
        helicity = 1 if axis != 2 else -1
        angle = helicity * axis_hat[axis]
        w_pi = .5 * (1 - angle**2)
        w_plus = .25 * (1 + angle)**2
        w_minus = .25 * (1 - angle)**2
        dv = v[axis] * inv_cooling
        d_p = delta - common - dv
        d_m = delta - common + dv
        a_p = (w_minus / (base + (inv_width*(d_p+projected))**2)
               + w_pi / (base + (inv_width*d_p)**2)
               + w_plus / (base + (inv_width*(d_p-projected))**2))
        a_m = (w_plus / (base + (inv_width*(d_m+projected))**2)
               + w_pi / (base + (inv_width*d_m)**2)
               + w_minus / (base + (inv_width*(d_m-projected))**2))
        accel[axis] = a_scale * sat[axis] * (a_p-a_m)
    accel[2] += gravity
    return accel, axis_hat


@njit(cache=True)
def step(r, v, dt, previous_axis, data):
    a1, n1 = local(r, v, previous_axis, *data)
    v2 = v + .5*dt*a1
    a2, n2 = local(r + .5*dt*v, v2, n1, *data)
    v3 = v + .5*dt*a2
    a3, n3 = local(r + .5*dt*v2, v3, n2, *data)
    v4 = v + dt*a3
    a4, n4 = local(r + dt*v3, v4, n3, *data)
    rn = r + (dt/6)*(v + 2*v2 + 2*v3 + v4)
    vn = v + (dt/6)*(a1 + 2*a2 + 2*a3 + a4)
    _, nn = local(rn, vn, n4, *data)
    return rn, vn, nn


@njit(cache=True)
def resolved_step_size(r, v, ceiling, data):
    """Limit displacement when a step could cross a narrow trap envelope.

    Test the entire straight-line predictor and pad its transverse distance by
    the maximal acceleration displacement. Use a lower bound on the waist
    and on the axial envelope scale along that padded segment.
    """
    params, geometry, _, _ = data
    accel_bound = params[7] + abs(params[8])
    pad = .5 * accel_bound * ceiling**2
    h = ceiling
    for axis in range(3):
        i, j = (axis+1)%3, (axis+2)%3
        vt2 = v[i]**2 + v[j]**2
        closest = min(ceiling, max(0., -(r[i]*v[i]+r[j]*v[j]) / vt2)) if vt2 else 0.
        transverse = sqrt((r[i]+closest*v[i])**2 + (r[j]+closest*v[j])**2)
        for beam in range(2*axis, 2*axis+2):
            origin, w02, inv_zr2, _ = geometry[beam]
            x0 = r[axis] - origin
            x1 = x0 + ceiling*v[axis]
            far = max(abs(x0), abs(x1)) + pad
            near = 0. if x0*x1 <= 0 else max(0., min(abs(x0), abs(x1))-pad)
            if transverse > 4*sqrt(w02*(1 + far**2*inv_zr2)) + pad:
                continue
            waist = sqrt(w02*(1 + near**2*inv_zr2))
            axial_scale = sqrt(near**2 + 1/inv_zr2)
            # Bound |v|*h + a*h^2/2 using |v| + a*ceiling/2.
            h = min(h, .35*waist/(sqrt(vt2)+.5*accel_bound*ceiling),
                    .35*axial_scale/(abs(v[axis])+.5*accel_bound*ceiling))
    return h


@njit(cache=True)
def shoot(r, v, dt, duration, data, previous_axis=np.array([0., 0., 1.])):
    """Track terminal wall loss and retention within a fixed final window."""
    start = duration - min(.1, duration/2)
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    max_final_r2 = 0.
    min_final_r2 = np.inf
    max_final_v2 = 0.
    entries = 0
    residence = 0.
    ever_core = False
    was_core = False
    elapsed = 0.
    while elapsed < duration:
        ceiling = min(dt, duration-elapsed)
        if elapsed < start:
            ceiling = min(ceiling, start-elapsed)
        h = resolved_step_size(r, v, ceiling, data)
        r, v, previous_axis = step(r, v, h, previous_axis, data)
        elapsed += h
        if not np.isfinite(v).all() or not np.isfinite(r).all():
            return NUMERICAL, r, v, elapsed
        if np.max(np.abs(r)) > CELL_HALF_M:
            return ESCAPED, r, v, elapsed
        r2 = r[0]**2 + r[1]**2 + r[2]**2
        now_core = r2 <= CORE_M**2
        if now_core:
            if not was_core:
                entries += 1
            residence += h
            ever_core = ever_core or residence >= .005 or entries >= 2
        else:
            residence = 0
        was_core = now_core
        if elapsed >= start:
            low = np.minimum(low, r)
            high = np.maximum(high, r)
            max_final_r2 = max(max_final_r2, r2)
            min_final_r2 = min(min_final_r2, r2)
            max_final_v2 = max(max_final_v2, v[0]**2+v[1]**2+v[2]**2)
    if max_final_v2 <= FINAL_SPEED_M_S**2:
        if ever_core and max_final_r2 <= CORE_M**2:
            return CORE, r, v, duration
        if min_final_r2 > CORE_M**2 and np.linalg.norm(high-low) <= .0005:
            return LOCALIZED_OFF_CORE, r, v, duration
    return UNRESOLVED, r, v, duration

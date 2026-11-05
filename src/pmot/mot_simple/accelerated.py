"""Optional compiled implementation of the reference two-level RK4 dynamics.

No fast-math, field linearization, force interpolation, or diffusion is used.
Complete elliptic integrals are evaluated by the convergent arithmetic-
geometric mean; tests compare against SciPy and the reference trajectories.
"""

from __future__ import annotations

from math import exp, pi, sqrt

import numpy as np
from numba import njit

from ..capture_statistics import TrajectoryClassification
from ..configuration import HBAR_J_S, RB87_MASS_KG, VACUUM_PERMEABILITY_H_PER_M


@njit(cache=True)
def elliptic_ke(m):
    if m < 0.0 or m >= 1.0:
        return np.nan, np.nan
    a, b = 1.0, sqrt(1.0 - m)
    correction, factor = 0.5 * m, 1.0
    for _ in range(30):
        c = 0.5 * (a - b)
        correction += factor * c * c
        next_a = 0.5 * (a + b)
        b = sqrt(a * b)
        a = next_a
        factor *= 2.0
        if abs(c) <= 2.0e-16 * a:
            break
    k = pi / (2.0 * a)
    return k, k * (1.0 - correction)


@njit(cache=True)
def magnetic_field(position, coil):
    x, y, z = position
    radius, half_separation, mu_ni = coil
    rho = sqrt(x*x + y*y)
    if abs(rho - radius) <= 1e-15 and (
        abs(z-half_separation) <= 1e-15 or abs(z+half_separation) <= 1e-15
    ):
        return np.full(3, np.nan)
    result = np.zeros(3)
    if rho <= 1.0e-18:
        result[2] = mu_ni * radius**2 / 2.0 * (
            1.0 / (radius**2 + (z+half_separation)**2)**1.5
            - 1.0 / (radius**2 + (z-half_separation)**2)**1.5
        )
        return result
    brho, bz = 0.0, 0.0
    for side in range(2):
        zz = z + half_separation if side == 0 else z - half_separation
        sign = 1.0 if side == 0 else -1.0
        alpha2 = (radius-rho)**2 + zz**2
        beta2 = (radius+rho)**2 + zz**2
        beta = sqrt(beta2)
        k, e = elliptic_ke(4.0 * radius * rho / beta2)
        bz += sign / beta * (k + (radius**2-rho**2-zz**2) / alpha2 * e)
        brho += sign * zz / beta * (-k + (radius**2+rho**2+zz**2)/alpha2*e)
    result[0] = mu_ni / (2*pi) * brho * x / (rho*rho)
    result[1] = mu_ni / (2*pi) * brho * y / (rho*rho)
    result[2] = mu_ni / (2*pi) * bz
    return result


def pack_parameters(beams, coil, simple):
    data = np.asarray([
        (*b.direction, *b.intensity_beam.reference_position_m,
         b.intensity_beam.power_w, b.intensity_beam.beam_radius_m,
         b.wavelength_m, b.polarization_sign, b.detuning_hz)
        for b in beams
    ], dtype=np.float64).reshape((-1, 11))
    coil_data = np.asarray([
        coil.radius_m, coil.half_separation_m,
        VACUUM_PERMEABILITY_H_PER_M * coil.turns_per_coil * coil.current_a,
    ])
    gravity = simple.gravity_acceleration_m_per_s2 if simple.include_gravity else (0., 0., 0.)
    atom = np.asarray([
        simple.linewidth_hz, simple.saturation_intensity_w_per_m2,
        simple.effective_magnetic_moment_hz_per_t,
        simple.maximum_scattering_rate_per_s, *gravity,
    ])
    return data, coil_data, atom


@njit(cache=True)
def acceleration(position, velocity, beams, coil, atom):
    field = magnetic_field(position, coil)
    saturation = np.empty(len(beams))
    total = 0.0
    for j in range(len(beams)):
        axial, norm2 = 0.0, 0.0
        for q in range(3):
            relative = position[q] - beams[j, q+3]
            axial += relative * beams[j, q]
            norm2 += relative * relative
        radial2 = max(0.0, norm2-axial*axial)
        waist2 = beams[j, 7]**2
        rayleigh = pi * waist2 / beams[j, 8]
        local_waist2 = waist2 * (1.0+(axial/rayleigh)**2)
        saturation[j] = 2*beams[j, 6] / (pi*local_waist2*atom[1]) * exp(-2*radial2/local_waist2)
        total += saturation[j]
    acc = atom[4:7].copy()
    for j in range(len(beams)):
        doppler, parallel_field = 0.0, 0.0
        for q in range(3):
            doppler += velocity[q] * beams[j, q] / beams[j, 8]
            parallel_field += field[q] * beams[j, q]
        detuning = beams[j, 10] - doppler - beams[j, 9]*atom[2]*parallel_field
        rate = atom[3]*saturation[j] / (1.0+total+(2*detuning/atom[0])**2)
        momentum_per_mass = HBAR_J_S*2*pi / (beams[j, 8]*RB87_MASS_KG)
        for q in range(3):
            acc[q] += momentum_per_mass * rate * beams[j, q]
    return acc


@njit(cache=True)
def rk4(position, velocity, dt, beams, coil, atom):
    a1 = acceleration(position, velocity, beams, coil, atom)
    v2 = velocity + 0.5*dt*a1
    a2 = acceleration(position + 0.5*dt*velocity, v2, beams, coil, atom)
    v3 = velocity + 0.5*dt*a2
    a3 = acceleration(position + 0.5*dt*v2, v3, beams, coil, atom)
    v4 = velocity + dt*a3
    a4 = acceleration(position + dt*v3, v4, beams, coil, atom)
    return (position + dt/6*(velocity+2*v2+2*v3+v4),
            velocity + dt/6*(a1+2*a2+2*a3+a4))


@njit(cache=True, nogil=True)
def classify_numeric(position, velocity, beams, coil, atom, controls):
    dt, duration, core, escape, required_entries, residence = controls
    position, velocity = position.copy(), velocity.copy()
    radius = sqrt(np.dot(position, position))
    minimum = radius
    was_inside = radius <= core
    entered = was_inside
    entries = int(was_inside)
    inside_since = 0.0 if was_inside else -1.0
    elapsed = 0.0
    code = 0  # timeout; 1 entries; 2 residence; 3 escaped; 4 nonfinite
    for _ in range(int(np.ceil(duration/dt))+1):
        radius = sqrt(np.dot(position, position))
        minimum = min(minimum, radius)
        inside = radius <= core
        entered = entered or inside
        if inside and not was_inside:
            entries += 1
            inside_since = elapsed
        elif not inside:
            inside_since = -1.0
        was_inside = inside
        if entries >= required_entries:
            code = 1
            break
        if inside_since >= 0.0 and elapsed-inside_since >= residence:
            code = 2
            break
        if radius >= escape and np.dot(position, velocity)/max(radius, 1e-15) > 0.0:
            code = 3
            break
        if elapsed >= duration-1e-15:
            break
        step = min(dt, duration-elapsed)
        position, velocity = rk4(position, velocity, step, beams, coil, atom)
        elapsed += step
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            code = 4
            break
    # The final allowed iteration can advance a tiny remainder to the requested
    # deadline and then fall through without another event-loop evaluation.
    # Include that saved endpoint in the reported minimum, without changing
    # the trajectory or any terminal classification.
    if np.all(np.isfinite(position)):
        minimum = min(minimum, sqrt(np.dot(position, position)))
    return code, entered, entries, elapsed, minimum, position, velocity


def classify_initial_conditions_batch(beams, positions, velocities, coil, simple, search):
    positions, velocities = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or velocities.shape != positions.shape:
        raise ValueError("positions and velocities must both have shape (N,3)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("initial conditions must be finite")
    if search.time_step_s <= 0 or search.max_simulation_time_s < 0 or search.bounded_core_residence_s < 0:
        raise ValueError("invalid timestep, duration, or residence time")
    data, coil_data, atom = pack_parameters(beams, coil, simple)
    controls = np.asarray([
        search.time_step_s, search.max_simulation_time_s,
        search.trap_core_radius_m, search.escape_radius_m,
        search.required_core_entries, search.bounded_core_residence_s,
    ])
    reasons = ("timeout", "two_core_entries", "bounded_core_residence", "escaped", "non_finite")
    results = []
    for position, velocity in zip(positions, velocities, strict=True):
        code, entered, entries, elapsed, minimum, final_r, final_v = classify_numeric(
            position, velocity, data, coil_data, atom, controls,
        )
        results.append(TrajectoryClassification(
            trapped=code in (1, 2), termination_reason=reasons[code],
            entered_trap_core=bool(entered), core_entry_count=int(entries),
            elapsed_time_s=float(elapsed), minimum_radius_m=float(minimum),
            final_radius_m=float(np.linalg.norm(final_r)),
            final_position_m=tuple(float(x) for x in final_r),
            final_velocity_m_per_s=tuple(float(x) for x in final_v),
        ))
    return tuple(results)


def classify_trajectory(beams, point, speed, coil, simple, search):
    if not np.isfinite(speed) or speed < 0:
        raise ValueError("incident speed must be finite and nonnegative")
    return classify_initial_conditions_batch(
        beams, [point.initial_position_m],
        [float(speed)*np.asarray(point.incident_unit_vector)], coil, simple, search,
    )[0]

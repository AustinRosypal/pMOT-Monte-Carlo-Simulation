"""Compiled Section-12 equations; checked against the reference implementation.

No force interpolation, saturated scattering closure, or archived imports.
Excited populations are eliminated algebraically from the stationary 24-state
system, leaving an equivalent normalized 8-state linear solve. All original
24 equations and photon-flow balance are checked on every evaluation.
Only circular traveling polarizations are supported by this optimization.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..configuration import (HBAR_J_S, RB87_MASS_KG,
    SPEED_OF_LIGHT_M_PER_S, VACUUM_PERMITTIVITY_F_PER_M,
    VACUUM_PERMEABILITY_H_PER_M)
from ..magnetic_fields import default_anti_helmholtz_config
from .configuration import default_multilevel_mot_config
from .rate_equations import build_rate_equation_model
from .simulation import build_multilevel_mot_beams


def prepare(config=None, coil=None):
    cfg = config or default_multilevel_mot_config()
    coil = coil or default_anti_helmholtz_config()
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=cfg)
    directions = np.array([b.direction for b in beams])
    optical = np.empty((len(beams), 5))
    strengths = np.zeros((len(beams), model.transition_count))
    offsets = np.zeros_like(strengths)
    for i, b in enumerate(beams):
        if b.circular_polarization not in ('sigma+', 'sigma-'):
            raise ValueError('compiled solver requires transverse circular polarization')
        optical[i] = (b.power_w, b.beam_radius_m**2,
                      (np.pi*b.beam_radius_m**2/b.wavelength_m)**2,
                      2*np.pi/b.wavelength_m,
                      1 if b.circular_polarization == 'sigma+' else -1)
        if b.family == 'cooling':
            mask = (model.transition_ground_f == 2) & np.isin(
                model.transition_excited_f, cfg.enabled_cooling_excited_manifolds)
            omega = model.structure.cooling_reference_angular_frequency_rad_per_s + cfg.cooling_detuning_rad_per_s
        else:
            mask = (model.transition_ground_f == 1) & np.isin(
                model.transition_excited_f, cfg.enabled_repump_excited_manifolds)
            omega = model.structure.repump_reference_angular_frequency_rad_per_s + cfg.repump_detuning_rad_per_s
        strengths[i, mask] = (2*model.transition_dipole_c_m[mask]**2 /
            (SPEED_OF_LIGHT_M_PER_S*VACUUM_PERMITTIVITY_F_PER_M*HBAR_J_S**2))
        offsets[i] = omega - model.transition_angular_frequency_rad_per_s
    controls = np.array([coil.radius_m, coil.half_separation_m,
        VACUUM_PERMEABILITY_H_PER_M*coil.turns_per_coil*coil.current_a/(2*np.pi),
        cfg.magnetic_field_epsilon_t, 9.80665 if cfg.include_gravity else 0.0,
        cfg.steady_state_relative_tolerance, cfg.population_absolute_tolerance])
    return (directions, optical, strengths, offsets,
            model.transition_ground.copy(), model.transition_excited.copy(),
            model.transition_q.copy(),
            model.transition_zeeman_coefficient_rad_per_s_per_t.copy(),
            model.excited_decay_rates_per_s.copy(),
            model.spontaneous_decay_matrix_per_s.copy(), controls)


@njit(cache=True)
def elliptic_ke(m):
    """Complete elliptic integrals via the arithmetic-geometric mean."""
    if m < 0 or m >= 1:
        raise ValueError('coil singularity or invalid elliptic parameter')
    a, b, correction, factor = 1.0, np.sqrt(1-m), 0.5*m, 1.0
    for _ in range(30):
        c = (a-b)*0.5
        correction += factor*c*c
        an = (a+b)*0.5
        b = np.sqrt(a*b)
        a = an
        factor *= 2
        if abs(c) < 2e-16:
            break
    k = np.pi/(2*a)
    return k, k*(1-correction)


@njit(cache=True)
def magnetic_field(position, controls):
    x, y, z = position
    radius, half, prefactor = controls[:3]
    rho = np.sqrt(x*x+y*y)
    br, bz = 0.0, 0.0
    for sign in (1.0, -1.0):
        zz = z + sign*half
        if rho <= 1e-18:
            bz += sign*prefactor*np.pi*radius**2/(radius**2+zz**2)**1.5
        else:
            alpha2 = (radius-rho)**2+zz**2
            beta2 = (radius+rho)**2+zz**2
            beta = np.sqrt(beta2)
            k, e = elliptic_ke(4*radius*rho/beta2)
            bz += sign*prefactor/beta*(k+(radius**2-rho**2-zz**2)/alpha2*e)
            br += sign*prefactor/rho*zz/beta*(-k+(radius**2+rho**2+zz**2)/alpha2*e)
    out = np.array([0.0, 0.0, bz])
    if rho > 1e-18:
        out[0], out[1] = br*x/rho, br*y/rho
    return out


@njit(cache=True)
def observable(position, velocity, previous_axis, data):
    directions, optical, strengths, offsets, gi, ei, qs, zc, gamma, decay, ctrl = data
    field = magnetic_field(position, ctrl)
    magnitude = np.sqrt(np.sum(field*field))
    axis = previous_axis if magnitude <= ctrl[3] else field/magnitude
    bw = np.zeros_like(strengths)
    w = np.zeros((16, 8))
    for b in range(len(directions)):
        axial = np.dot(position, directions[b])
        radial2 = max(0.0, np.dot(position, position)-axial*axial)
        waist2 = optical[b, 1]*(1+axial*axial/optical[b, 2])
        intensity = 2*optical[b, 0]/(np.pi*waist2)*np.exp(-2*radial2/waist2)
        cosine = np.dot(directions[b], axis)
        doppler = optical[b, 3]*np.dot(directions[b], velocity)
        for t in range(len(gi)):
            if strengths[b, t] == 0:
                continue
            weight = ((1+optical[b, 4]*qs[t]*cosine)**2/4 if qs[t] != 0
                      else (1-cosine*cosine)/2)
            det = offsets[b, t]-doppler-zc[t]*magnitude
            value = gamma[ei[t]]*strengths[b, t]*intensity*max(0.0, weight)/(gamma[ei[t]]**2+4*det*det)
            bw[b, t] = value
            w[ei[t], gi[t]] += value
    de = gamma + np.sum(w, axis=1)
    ratios = w/de.reshape((16, 1))
    # Off-diagonal ground generator avoids subtracting nearly equal large rates.
    reduced = np.zeros((8, 8))
    for g in range(8):
        for h in range(8):
            if g != h:
                for e in range(16):
                    reduced[g, h] += (w[e, g]+decay[g, e])*ratios[e, h]
                reduced[h, h] -= reduced[g, h]
    scale = np.max(np.abs(reduced))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('no unique ground steady state')
    system = reduced/scale
    for g in range(8):
        system[7, g] = 1+np.sum(ratios[:, g])
    rhs = np.zeros(8)
    rhs[7] = 1
    pg = np.linalg.solve(system, rhs)
    pe = ratios @ pg
    populations = np.concatenate((pg, pe))
    if np.min(populations) < -ctrl[6] or not np.all(np.isfinite(populations)):
        raise ValueError('invalid stationary populations')
    populations = np.maximum(populations, 0)
    populations /= np.sum(populations)
    pg, pe = populations[:8], populations[8:]
    rg = (w.T+decay) @ pe - np.sum(w, axis=0)*pg
    re = w @ pg - de*pe
    original_scale = max(np.max(de), np.max(np.sum(w, axis=0)))
    if max(np.max(np.abs(rg)), np.max(np.abs(re))) > ctrl[5]*original_scale:
        raise ValueError('24-state steady-state residual exceeds tolerance')
    force, rates = np.zeros(3), np.zeros(len(directions))
    for b in range(len(directions)):
        for t in range(len(gi)):
            rates[b] += bw[b, t]*(pg[gi[t]]-pe[ei[t]])
        force += HBAR_J_S*optical[b, 3]*directions[b]*rates[b]
    spontaneous = np.dot(gamma, pe)
    if abs(np.sum(rates)-spontaneous) > max(1e-7, ctrl[5]*max(spontaneous, 1))+10*ctrl[5]*spontaneous:
        raise ValueError('net photon flow does not balance spontaneous decay')
    return force, populations, rates, axis


@njit(cache=True)
def acceleration(r, v, axis, data):
    force, _, _, _ = observable(r, v, axis, data)
    out = force/RB87_MASS_KG
    out[2] -= data[-1][4]
    return out


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
    field = magnetic_field(rn, data[-1])
    mag = np.sqrt(np.dot(field, field))
    return rn, vn, axis if mag <= data[-1][3] else field/mag


@njit(cache=True)
def classify(r0, v0, dt, duration, data, early_exit=True):
    """Return (code, time, r, v, first_trapped_time); 1 trapped, 0 escaped, -1 timeout.

    With early_exit=False, continue beyond the early criterion for a bounded
    trajectory audit. Completion alone is not recast as trapped.
    """
    r, v, axis = r0.copy(), v0.copy(), np.array([0., 0., 1.])
    was_inside = np.sqrt(np.dot(r, r)) <= .002
    entries = int(was_inside)
    inside_since = 0.0 if was_inside else -1.0
    first_trapped = -1.0
    t = 0.0
    for _ in range(int(np.ceil(duration/dt))+1):
        radius = np.sqrt(np.dot(r, r))
        inside = radius <= .002
        if inside and not was_inside:
            entries += 1
            inside_since = t
        elif not inside:
            inside_since = -1.0
        was_inside = inside
        trapped = entries >= 2 or (inside_since >= 0 and t-inside_since >= .005)
        if trapped and first_trapped < 0:
            first_trapped = t
        if trapped and early_exit:
            return 1, t, r, v, first_trapped
        if radius >= .03 and np.dot(r, v) > 0:
            return 0, t, r, v, first_trapped
        if t >= duration-1e-15:
            break
        h = min(dt, duration-t)
        r, v, axis = step(r, v, axis, h, data)
        t += h
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise ValueError('nonfinite trajectory')
    return -1, t, r, v, first_trapped


@njit(cache=True)
def retention_audit(r0, v0, dt, duration, data):
    """Continue to duration; report maximum radius over the final 25 ms."""
    r,v,axis=r0.copy(),v0.copy(),np.array([0.,0.,1.])
    maximum_radius=0.
    t=0.
    for _ in range(int(np.ceil(duration/dt))+1):
        radius=np.sqrt(np.dot(r,r))
        if t>=duration-.025:
            maximum_radius=max(maximum_radius,radius)
        if radius>=.03 and np.dot(r,v)>0:
            return False,t,r,v,maximum_radius
        if t>=duration-1e-15:
            break
        h=min(dt,duration-t)
        r,v,axis=step(r,v,axis,h,data)
        t+=h
    return maximum_radius<=.002,t,r,v,maximum_radius

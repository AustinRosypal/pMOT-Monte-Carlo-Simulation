"""Vectorized trajectory classification for two-level capture audits.

The production threshold search normally follows one launch speed at a time.
Audit scans, in contrast, evaluate many speeds for the *same* launch ray.  This
module advances those independent trajectories in one NumPy batch while using
the identical RK4 force law, terminal-event ordering, and SI-unit conventions
as :mod:`pmot.mot_simple.sampling`.

It is intentionally an optimization layer rather than a different physical
model.  Scalar/batch agreement is regression-tested before the batch path is
used for expensive finite-island scans.
"""

from __future__ import annotations

from math import pi
from typing import Sequence

import numpy as np

from ..capture_statistics import TrajectoryClassification
from ..configuration import HBAR_J_S, RB87_MASS_KG
from ..launch_geometry import PointSample
from ..magnetic_fields import anti_helmholtz_field_t
from .configuration import SimpleMOTConfig
from .sampling import CaptureSearchConfig
from .simulation import SimpleMOTBeam


def _beam_arrays(beams: Sequence[SimpleMOTBeam]) -> dict[str, np.ndarray]:
    if not beams:
        raise ValueError("at least one beam is required")
    directions = np.asarray([beam.direction for beam in beams], dtype=float)
    return {
        "directions": directions,
        "reference_positions": np.asarray(
            [beam.intensity_beam.reference_position_m for beam in beams], dtype=float
        ),
        "powers": np.asarray(
            [beam.intensity_beam.power_w for beam in beams], dtype=float
        ),
        "wavelengths": np.asarray(
            [beam.wavelength_m for beam in beams], dtype=float
        ),
        "waist_radii": np.asarray(
            [beam.intensity_beam.beam_radius_m for beam in beams], dtype=float
        ),
        "polarization_signs": np.asarray(
            [beam.polarization_sign for beam in beams], dtype=float
        ),
        "detunings": np.asarray([beam.detuning_hz for beam in beams], dtype=float),
    }


def _acceleration_batch(
    positions_m: np.ndarray,
    velocities_m_per_s: np.ndarray,
    beam_data: dict[str, np.ndarray],
    coil_config,
    simple_config: SimpleMOTConfig,
) -> np.ndarray:
    """Evaluate the exact simple-MOT acceleration for an ``(N, 3)`` batch."""

    positions = np.asarray(positions_m, dtype=float)
    velocities = np.asarray(velocities_m_per_s, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities_m_per_s must match positions_m")

    bx, by, bz = anti_helmholtz_field_t(
        positions[:, 0], positions[:, 1], positions[:, 2], coil_config
    )
    magnetic_field = np.column_stack((bx, by, bz))

    directions = beam_data["directions"]
    wavelengths = beam_data["wavelengths"]
    relative = positions[:, None, :] - beam_data["reference_positions"][None, :, :]
    axial = np.einsum("nbc,bc->nb", relative, directions)
    radial_squared = np.maximum(
        0.0, np.einsum("nbc,nbc->nb", relative, relative) - axial**2
    )
    waist_squared_0 = beam_data["waist_radii"] ** 2
    rayleigh_ranges = pi * waist_squared_0 / wavelengths
    local_waist_squared = waist_squared_0[None, :] * (
        1.0 + (axial / rayleigh_ranges[None, :]) ** 2
    )
    intensities = (
        2.0
        * beam_data["powers"][None, :]
        / (pi * local_waist_squared)
        * np.exp(-2.0 * radial_squared / local_waist_squared)
    )
    saturations = intensities / simple_config.saturation_intensity_w_per_m2
    total_saturation = np.sum(saturations, axis=1, keepdims=True)

    doppler_hz = (
        velocities @ directions.T
    ) / wavelengths[None, :]
    zeeman_hz = (
        simple_config.effective_magnetic_moment_hz_per_t
        * (magnetic_field @ directions.T)
        * beam_data["polarization_signs"][None, :]
    )
    effective_detuning_hz = (
        beam_data["detunings"][None, :] - doppler_hz - zeeman_hz
    )
    rates_per_s = (
        0.5
        * simple_config.linewidth_hz
        * saturations
        / (
            1.0
            + total_saturation
            + (2.0 * effective_detuning_hz / simple_config.linewidth_hz) ** 2
        )
    )
    photon_momenta = HBAR_J_S * 2.0 * pi / wavelengths
    forces_n = (rates_per_s * photon_momenta[None, :]) @ directions
    acceleration = forces_n / RB87_MASS_KG
    if simple_config.include_gravity:
        acceleration = acceleration + np.asarray(
            simple_config.gravity_acceleration_m_per_s2, dtype=float
        )[None, :]
    return acceleration


def _rk4_batch(
    positions_m: np.ndarray,
    velocities_m_per_s: np.ndarray,
    time_step_s: float,
    beam_data: dict[str, np.ndarray],
    coil_config,
    simple_config: SimpleMOTConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance independent simple-MOT trajectories by one shared RK4 step."""

    dt = float(time_step_s)
    k1_r = velocities_m_per_s
    k1_v = _acceleration_batch(
        positions_m, velocities_m_per_s, beam_data, coil_config, simple_config
    )
    k2_r = velocities_m_per_s + 0.5 * dt * k1_v
    k2_v = _acceleration_batch(
        positions_m + 0.5 * dt * k1_r,
        velocities_m_per_s + 0.5 * dt * k1_v,
        beam_data,
        coil_config,
        simple_config,
    )
    k3_r = velocities_m_per_s + 0.5 * dt * k2_v
    k3_v = _acceleration_batch(
        positions_m + 0.5 * dt * k2_r,
        velocities_m_per_s + 0.5 * dt * k2_v,
        beam_data,
        coil_config,
        simple_config,
    )
    k4_r = velocities_m_per_s + dt * k3_v
    k4_v = _acceleration_batch(
        positions_m + dt * k3_r,
        velocities_m_per_s + dt * k3_v,
        beam_data,
        coil_config,
        simple_config,
    )
    return (
        positions_m + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r),
        velocities_m_per_s
        + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v),
    )


def classify_initial_conditions_batch(
    beams: Sequence[SimpleMOTBeam],
    initial_positions_m: Sequence[Sequence[float]] | np.ndarray,
    initial_velocities_m_per_s: Sequence[Sequence[float]] | np.ndarray,
    coil_config,
    simple_config: SimpleMOTConfig,
    search_config: CaptureSearchConfig,
) -> tuple[TrajectoryClassification, ...]:
    """Classify arbitrary independent initial states in one RK4 loop.

    Advancing a larger collection changes only NumPy scheduling: every row has
    its own position, velocity, capture history, and terminal result.  The
    shared clock is valid because all rows use the same fixed timestep and
    stopping duration.  Result order is exactly the input-row order.
    """

    initial_positions = np.asarray(initial_positions_m, dtype=float)
    initial_velocities = np.asarray(initial_velocities_m_per_s, dtype=float)
    if initial_positions.ndim != 2 or initial_positions.shape[1] != 3:
        raise ValueError("initial_positions_m must have shape (N, 3)")
    if initial_velocities.shape != initial_positions.shape:
        raise ValueError(
            "initial_velocities_m_per_s must match initial_positions_m"
        )
    if np.any(~np.isfinite(initial_positions)) or np.any(
        ~np.isfinite(initial_velocities)
    ):
        raise ValueError("initial positions and velocities must be finite")
    if search_config.time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive")
    if search_config.max_simulation_time_s < 0.0:
        raise ValueError("max_simulation_time_s must be non-negative")
    if search_config.bounded_core_residence_s < 0.0:
        raise ValueError("bounded_core_residence_s must be non-negative")
    if len(initial_positions) == 0:
        return ()

    count = len(initial_positions)
    positions = np.array(initial_positions, dtype=float, copy=True)
    velocities = np.array(initial_velocities, dtype=float, copy=True)
    beam_data = _beam_arrays(beams)

    radii = np.linalg.norm(positions, axis=1)
    minimum_radii = radii.copy()
    was_inside = radii <= search_config.trap_core_radius_m
    entered_core = was_inside.copy()
    core_entry_count = was_inside.astype(int)
    inside_since = np.where(was_inside, 0.0, np.nan)
    active = np.ones(count, dtype=bool)
    results: list[TrajectoryClassification | None] = [None] * count
    elapsed_time_s = 0.0
    max_steps = int(
        np.ceil(search_config.max_simulation_time_s / search_config.time_step_s)
    )

    def finish(indices: np.ndarray, trapped: bool, reason: str) -> None:
        for index in indices.tolist():
            radius = float(np.linalg.norm(positions[index]))
            results[index] = TrajectoryClassification(
                trapped=trapped,
                termination_reason=reason,
                entered_trap_core=bool(entered_core[index]),
                core_entry_count=int(core_entry_count[index]),
                elapsed_time_s=float(elapsed_time_s),
                minimum_radius_m=float(minimum_radii[index]),
                final_radius_m=radius,
                final_position_m=tuple(float(value) for value in positions[index]),
                final_velocity_m_per_s=tuple(float(value) for value in velocities[index]),
            )
        active[indices] = False

    for _ in range(max_steps + 1):
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            break
        active_positions = positions[active_indices]
        active_velocities = velocities[active_indices]
        radii = np.linalg.norm(active_positions, axis=1)
        minimum_radii[active_indices] = np.minimum(
            minimum_radii[active_indices], radii
        )
        inside = radii <= search_config.trap_core_radius_m
        entered_core[active_indices] |= inside
        newly_inside = inside & ~was_inside[active_indices]
        core_entry_count[active_indices[newly_inside]] += 1
        inside_since[active_indices[newly_inside]] = elapsed_time_s
        inside_since[active_indices[~inside]] = np.nan
        was_inside[active_indices] = inside
        radial_velocity = np.einsum(
            "nc,nc->n", active_positions, active_velocities
        ) / np.maximum(radii, 1.0e-15)

        trapped_entries = active_indices[
            core_entry_count[active_indices] >= search_config.required_core_entries
        ]
        if len(trapped_entries):
            finish(trapped_entries, True, "two_core_entries")

        remaining = np.flatnonzero(active)
        if len(remaining):
            residence = (
                ~np.isnan(inside_since[remaining])
                & (
                    elapsed_time_s - inside_since[remaining]
                    >= search_config.bounded_core_residence_s
                )
            )
            if np.any(residence):
                finish(remaining[residence], True, "bounded_core_residence")

        remaining = np.flatnonzero(active)
        if len(remaining):
            remaining_positions = positions[remaining]
            remaining_velocities = velocities[remaining]
            remaining_radii = np.linalg.norm(remaining_positions, axis=1)
            remaining_radial_velocity = np.einsum(
                "nc,nc->n", remaining_positions, remaining_velocities
            ) / np.maximum(remaining_radii, 1.0e-15)
            escaped = (
                (remaining_radii >= search_config.escape_radius_m)
                & (remaining_radial_velocity > 0.0)
            )
            if np.any(escaped):
                finish(remaining[escaped], False, "escaped")

        remaining = np.flatnonzero(active)
        if len(remaining) == 0:
            break
        if elapsed_time_s >= search_config.max_simulation_time_s - 1.0e-15:
            finish(remaining, False, "timeout")
            break

        step_time_s = min(
            search_config.time_step_s,
            search_config.max_simulation_time_s - elapsed_time_s,
        )
        next_positions, next_velocities = _rk4_batch(
            positions[remaining],
            velocities[remaining],
            step_time_s,
            beam_data,
            coil_config,
            simple_config,
        )
        positions[remaining] = next_positions
        velocities[remaining] = next_velocities
        elapsed_time_s += step_time_s

        finite = np.all(np.isfinite(next_positions), axis=1) & np.all(
            np.isfinite(next_velocities), axis=1
        )
        if not np.all(finite):
            finish(remaining[~finite], False, "non_finite")

    # Mirror the scalar classifier's fall-through timeout.  Floating-point
    # accumulation can leave the shared clock a few femtoseconds below the
    # requested endpoint on the final loop iteration.
    remaining = np.flatnonzero(active)
    if len(remaining):
        finish(remaining, False, "timeout")
    unresolved = [index for index, result in enumerate(results) if result is None]
    if unresolved:
        raise RuntimeError(
            "batched classifier failed to terminate every trajectory; "
            f"unresolved_count={len(unresolved)}, elapsed_time_s={elapsed_time_s:.17g}"
        )
    return tuple(result for result in results if result is not None)


def classify_trajectory_batch(
    beams: Sequence[SimpleMOTBeam],
    point: PointSample,
    incident_speeds_m_per_s: Sequence[float] | np.ndarray,
    coil_config,
    simple_config: SimpleMOTConfig,
    search_config: CaptureSearchConfig,
) -> tuple[TrajectoryClassification, ...]:
    """Classify many launch speeds for one ray using one vectorized RK4 loop."""

    speeds = np.asarray(incident_speeds_m_per_s, dtype=float)
    if speeds.ndim != 1:
        raise ValueError("incident_speeds_m_per_s must be one-dimensional")
    if np.any(~np.isfinite(speeds)) or np.any(speeds < 0.0):
        raise ValueError("incident speeds must be finite and non-negative")
    count = len(speeds)
    positions = np.repeat(
        np.asarray(point.initial_position_m, dtype=float)[None, :], count, axis=0
    )
    velocities = speeds[:, None] * np.asarray(
        point.incident_unit_vector, dtype=float
    )[None, :]
    return classify_initial_conditions_batch(
        beams,
        positions,
        velocities,
        coil_config,
        simple_config,
        search_config,
    )


__all__ = ["classify_initial_conditions_batch", "classify_trajectory_batch"]

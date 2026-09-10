"""Run two provisional pMOT trajectories with a local fictitious-field axis.

At every trajectory sample the six trapping components are evaluated at the
current position and velocity, their stretched-transition-equivalent vector
fields are summed, and the normalized sum is supplied to the unchanged
24-state cooling/repump rate kernel.  The output independently reconstructs
the beamwise fields and all local spherical-polarization weights for QA.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from math import pi
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...configuration import PLANCK_CONSTANT_J_S
from ...configuration import SPEED_OF_LIGHT_M_PER_S
from ...configuration import VACUUM_PERMITTIVITY_F_PER_M
from ...mot_multilevel.coupling import beam_polarization_vector
from ...mot_multilevel.polarization import polarization_weights
from ...mot_multilevel.polarization import propagation_frame_polarization
from ...mot_multilevel.rate_equations import RateEquationAtomState
from ...mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from ..polarizability import interpolate_differential_polarizability_arrays
from ..trajectory_plotting import plot_pmot_trajectory_diagnostics
from ..trapping_beams import helicity_sign
from ..vector_only_trajectories import build_vector_only_trajectory_context
from ..vector_only_trajectories import inward_launch_state
from ..vector_only_trajectories import simulate_vector_only_pmot_trajectory
from ..vector_only_trajectories import vector_only_trajectory_dataframe


CAMPAIGN_NAME = "temporary_local_field_axis_two_trajectories_20260909"
DURATION_S = 25.0e-3
PRIMARY_TIME_STEP_S = 2.5e-6
COARSE_TIME_STEP_S = 5.0e-6
CORE_RADIUS_M = 2.0e-3
OBLIQUE_UNIT_VECTOR = np.asarray((2.0, 3.0, 6.0)) / 7.0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _reference_transition_index(model) -> int:
    for index, transition in enumerate(model.structure.absorption_transitions):
        if (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
        ) == (2, 2, 3, 3):
            return index
    raise RuntimeError("the stretched cycling transition is absent")


def _cases() -> dict[str, RateEquationAtomState]:
    return {
        "axial_x": inward_launch_state(),
        "oblique_236": RateEquationAtomState(
            tuple(15.0e-3 * OBLIQUE_UNIT_VECTOR),
            tuple(-17.0 * OBLIQUE_UNIT_VECTOR),
        ),
    }


def _safe_label(value: str) -> str:
    return (
        value.replace("horizontal_", "")
        .replace("vertical_", "")
        .replace("_trapping", "")
        .replace(" ", "_")
    )


def reconstruct_component_fields_t(record, context):
    """Reconstruct all six transition-equivalent fields at every sample."""

    wavelengths_nm = np.asarray(record.atom_frame_wavelengths_nm, dtype=float)
    intensities = np.asarray(
        record.trapping_component_intensities_w_per_m2,
        dtype=float,
    )
    _, alpha_vector, _ = interpolate_differential_polarizability_arrays(
        wavelengths_nm,
        context.polarizability_table,
    )
    field_squared = 2.0 * intensities / (
        SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M
    )
    directions = np.asarray(
        [beam.direction for beam in context.trapping_beams],
        dtype=float,
    )
    signs = np.asarray(
        [helicity_sign(beam.helicity) for beam in context.trapping_beams],
        dtype=float,
    )
    vector_energy = (
        -alpha_vector[..., None]
        * field_squared[..., None]
        * signs[None, :, None]
        * directions[None, :, :]
    )
    reference_index = _reference_transition_index(context.model)
    magnetic_moment = (
        context.model.transition_zeeman_coefficient[reference_index]
        * PLANCK_CONSTANT_J_S
        / (2.0 * pi)
    )
    component_fields = vector_energy / magnetic_moment
    component_field_sum = np.sum(component_fields, axis=1)
    # This order matches ac_stark.py exactly: sum vector energies, then divide.
    reconstructed_net = np.sum(vector_energy, axis=1) / magnetic_moment
    recorded_net = np.asarray(record.effective_fields_t, dtype=float)
    maximum_record_error_t = float(
        np.max(np.linalg.norm(reconstructed_net - recorded_net, axis=1))
    )
    maximum_component_sum_roundoff_t = float(
        np.max(np.linalg.norm(component_field_sum - reconstructed_net, axis=1))
    )
    if not np.allclose(
        reconstructed_net,
        recorded_net,
        rtol=2.0e-12,
        atol=2.0e-14,
    ):
        raise RuntimeError(
            "beamwise effective-field reconstruction disagrees with trajectory"
        )
    return (
        component_fields,
        maximum_record_error_t,
        maximum_component_sum_roundoff_t,
    )


def polarization_history(record, context):
    """Return all 18 beam projections ordered sigma+, pi, sigma-."""

    axes = np.asarray(record.quantization_axes, dtype=float)
    beam_records = []
    lab_polarizations = []
    for beam in context.cooling_repump_beams:
        beam_records.append(
            {
                "label": (
                    f"{beam.family}_{_safe_label(beam.axis_name)}_"
                    f"{beam.propagation_sense}"
                ),
                "family": beam.family,
            }
        )
        lab_polarizations.append(beam_polarization_vector(beam))
    for beam in context.trapping_beams:
        beam_records.append(
            {
                "label": f"trapping_{_safe_label(beam.label)}",
                "family": "trapping",
            }
        )
        lab_polarizations.append(
            propagation_frame_polarization(beam.direction, beam.helicity)
        )
    weights = np.empty((len(axes), len(beam_records), 3), dtype=float)
    for time_index, axis in enumerate(axes):
        axis_tuple = tuple(float(value) for value in axis)
        for beam_index, polarization in enumerate(lab_polarizations):
            values = polarization_weights(polarization, axis_tuple)
            weights[time_index, beam_index] = (
                values[+1],
                values[0],
                values[-1],
            )
    maximum_sum_error = float(np.max(np.abs(np.sum(weights, axis=2) - 1.0)))
    if maximum_sum_error > 2.0e-14:
        raise RuntimeError("local spherical polarization weights do not sum to one")
    return weights, beam_records, maximum_sum_error


def _trajectory_frame(record, context) -> pd.DataFrame:
    frame = vector_only_trajectory_dataframe(record)
    rates = np.asarray(record.rate_equation.beam_scattering_rates_per_s)
    for index, beam in enumerate(context.cooling_repump_beams):
        label = (
            f"{beam.family}_{_safe_label(beam.axis_name)}_"
            f"{beam.propagation_sense}_absorption_rate_per_s"
        )
        frame[label] = rates[:, index]
    field = np.asarray(record.effective_fields_t)
    frame["effective_field_proxy_magnitude_t"] = np.linalg.norm(field, axis=1)
    return frame


def _field_frame(record, context, fields_t) -> pd.DataFrame:
    data = {"time_s": record.rate_equation.times_s}
    for index, beam in enumerate(context.trapping_beams):
        label = f"trapping_{_safe_label(beam.label)}"
        for component, axis_name in enumerate("xyz"):
            data[f"{label}_beq_{axis_name}_t"] = fields_t[:, index, component]
    return pd.DataFrame(data)


def _polarization_frame(record, beam_records, weights) -> pd.DataFrame:
    data = {"time_s": record.rate_equation.times_s}
    names = ("sigma_plus_fraction", "pi_fraction", "sigma_minus_fraction")
    for beam_index, beam in enumerate(beam_records):
        for component, name in enumerate(names):
            data[f"{beam['label']}_{name}"] = weights[:, beam_index, component]
    return pd.DataFrame(data)


def _summary(
    record,
    fields_t,
    field_error_t,
    component_sum_roundoff_t,
    weight_error,
    wall_s,
):
    base = record.rate_equation
    times = np.asarray(base.times_s)
    position = np.asarray(base.positions_m)
    velocity = np.asarray(base.velocities_m_per_s)
    force = np.asarray(base.forces_n)
    radius = np.linalg.norm(position, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    net_field = np.asarray(record.effective_fields_t, dtype=float)
    field_magnitude = np.linalg.norm(net_field, axis=1)
    axes = np.asarray(record.quantization_axes)
    angle_steps = np.degrees(
        np.arccos(
            np.clip(np.sum(axes[1:] * axes[:-1], axis=1), -1.0, 1.0)
        )
    )
    inside = np.flatnonzero(radius <= CORE_RADIUS_M)
    peak_index = int(np.argmax(field_magnitude))
    return {
        "termination_reason": base.termination_reason,
        "wall_time_s": wall_s,
        "sample_count": len(times),
        "elapsed_time_ms": 1.0e3 * times[-1],
        "initial_position_mm": (1.0e3 * position[0]).tolist(),
        "initial_velocity_m_per_s": velocity[0].tolist(),
        "final_position_mm": (1.0e3 * position[-1]).tolist(),
        "final_velocity_m_per_s": velocity[-1].tolist(),
        "final_radius_mm": 1.0e3 * radius[-1],
        "minimum_radius_mm": 1.0e3 * np.min(radius),
        "final_speed_m_per_s": speed[-1],
        "maximum_speed_m_per_s": np.max(speed),
        "first_core_entry_time_ms": (
            None if not len(inside) else 1.0e3 * times[inside[0]]
        ),
        "capture": asdict(record.capture),
        "initial_effective_field_proxy_g": (1.0e4 * net_field[0]).tolist(),
        "final_effective_field_proxy_g": (1.0e4 * net_field[-1]).tolist(),
        "minimum_effective_field_proxy_g": 1.0e4 * np.min(field_magnitude),
        "maximum_sampled_effective_field_proxy_g": 1.0e4 * np.max(field_magnitude),
        "peak_field_time_ms": 1.0e3 * times[peak_index],
        "peak_field_position_mm": (1.0e3 * position[peak_index]).tolist(),
        "maximum_interstep_axis_angle_deg": (
            0.0 if not len(angle_steps) else np.max(angle_steps)
        ),
        "axis_steps_over_30_deg": int(np.count_nonzero(angle_steps > 30.0)),
        "initial_quantization_axis": axes[0].tolist(),
        "final_quantization_axis": axes[-1].tolist(),
        "reference_vector_shift_range_mhz": [
            1.0e-6 * np.min(record.reference_vector_shift_hz),
            1.0e-6 * np.max(record.reference_vector_shift_hz),
        ],
        "maximum_radiation_force_n": np.max(np.linalg.norm(force, axis=1)),
        "mean_ground_weighted_absorption_rate_per_s": np.mean(
            base.total_scattering_rates_per_s
        ),
        "beamwise_field_sum_maximum_error_t": field_error_t,
        "sum_fields_vs_sum_energy_then_divide_maximum_roundoff_t": (
            component_sum_roundoff_t
        ),
        "polarization_normalization_maximum_error": weight_error,
    }


def _plot_field_axis_and_polarization(
    record,
    context,
    weights,
    beam_records,
    title,
    path,
):
    times_ms = 1.0e3 * np.asarray(record.rate_equation.times_s)
    field_g = 1.0e4 * np.asarray(record.effective_fields_t)
    axes_history = np.asarray(record.quantization_axes)
    figure, panels = plt.subplots(2, 2, figsize=(14.5, 9.0))
    colors = ("#2563eb", "#f97316", "#16a34a")
    for component, (axis_name, color) in enumerate(zip("xyz", colors)):
        panels[0, 0].plot(
            times_ms,
            field_g[:, component],
            color=color,
            label=rf"$B_{{{axis_name}}}$",
        )
        panels[0, 1].plot(
            times_ms,
            axes_history[:, component],
            color=color,
            label=rf"$n_{{{axis_name}}}$",
        )
    panels[0, 0].set_yscale("symlog", linthresh=1.0e-3)
    panels[0, 0].set(
        title="Net transition-equivalent field",
        ylabel="Field proxy [G; symmetric log scale]",
    )
    panels[0, 1].set(
        title="Normalized local quantization-axis proxy",
        ylabel="Axis component",
        ylim=(-1.05, 1.05),
    )

    field_magnitude_g = np.linalg.norm(field_g, axis=1)
    panels[1, 0].semilogy(
        times_ms,
        np.maximum(field_magnitude_g, 1.0e-15),
        color="#111827",
    )
    panels[1, 0].set(
        title="Fictitious-field magnitude",
        ylabel="|B proxy| [G]",
    )

    cooling_indices = [
        index
        for index, beam in enumerate(beam_records)
        if beam["family"] == "cooling"
    ]
    mean_weights = np.mean(weights[:, cooling_indices, :], axis=0)
    image = panels[1, 1].imshow(
        mean_weights,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        aspect="auto",
    )
    panels[1, 1].set_xticks((0, 1, 2), (r"$\sigma^+$", r"$\pi$", r"$\sigma^-$"))
    panels[1, 1].set_yticks(
        np.arange(len(cooling_indices)),
        [beam_records[index]["label"] for index in cooling_indices],
    )
    panels[1, 1].set_title(
        "Time-averaged cooling polarization fractions\n"
        "(repump components have identical geometry)"
    )
    for row in range(mean_weights.shape[0]):
        for column in range(3):
            panels[1, 1].text(
                column,
                row,
                f"{mean_weights[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if mean_weights[row, column] < 0.45 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=panels[1, 1], label="Fraction")
    for panel in panels.flat:
        if panel is not panels[1, 1]:
            panel.set_xlabel("Time [ms]")
            panel.grid(alpha=0.22)
            if panel is not panels[1, 0]:
                panel.legend(frameon=False, fontsize=8)
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_comparison(records, path):
    figure, panels = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=False)
    colors = {"axial_x": "#2563eb", "oblique_236": "#dc2626"}
    labels = {"axial_x": "axial +x launch", "oblique_236": "oblique (2,3,6)/7 launch"}
    for name, record in records.items():
        times_ms = 1.0e3 * np.asarray(record.rate_equation.times_s)
        position = np.asarray(record.rate_equation.positions_m)
        velocity = np.asarray(record.rate_equation.velocities_m_per_s)
        field_g = 1.0e4 * np.linalg.norm(
            np.asarray(record.effective_fields_t), axis=1
        )
        rates = np.asarray(record.rate_equation.total_scattering_rates_per_s)
        panels[0, 0].plot(
            times_ms,
            1.0e3 * np.linalg.norm(position, axis=1),
            color=colors[name],
            label=labels[name],
        )
        panels[0, 1].plot(
            times_ms,
            np.linalg.norm(velocity, axis=1),
            color=colors[name],
            label=labels[name],
        )
        panels[1, 0].semilogy(
            times_ms,
            np.maximum(field_g, 1.0e-15),
            color=colors[name],
            label=labels[name],
        )
        panels[1, 1].plot(
            times_ms,
            rates,
            color=colors[name],
            label=labels[name],
        )
    panels[0, 0].axhspan(0.0, 2.0, color="#16a34a", alpha=0.10, label="2 mm core")
    panels[0, 0].set(title="Distance from origin", ylabel="Radius [mm]")
    panels[0, 1].set(title="Speed", ylabel="Speed [m/s]")
    panels[1, 0].set(title="Transition-equivalent field", ylabel="|B proxy| [G]")
    panels[1, 1].set(
        title="Inherited ground-weighted absorption rate",
        ylabel=r"Rate [s$^{-1}$]",
    )
    for panel in panels.flat:
        panel.set_xlabel("Time [ms]")
        panel.grid(alpha=0.22)
        panel.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Two local-effective-field-axis pMOT trajectory diagnostics\n"
        "25 ms, 2.5 us step, 17 m/s inward launches, gravity on, diffusion off"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _coarse_summary(record):
    position = np.asarray(record.rate_equation.positions_m)
    velocity = np.asarray(record.rate_equation.velocities_m_per_s)
    radius = np.linalg.norm(position, axis=1)
    return {
        "capture": asdict(record.capture),
        "final_position_m": position[-1].tolist(),
        "final_velocity_m_per_s": velocity[-1].tolist(),
        "minimum_radius_m": np.min(radius),
    }


def _write_readme(path: Path, summary: dict) -> None:
    axial = summary["trajectories"]["axial_x"]
    oblique = summary["trajectories"]["oblique_236"]
    text = f"""# Two local-effective-field-axis pMOT trajectories

This is a provisional vector-only stretched-transition diagnostic. At each
time sample the six 1529-nm component fields are reconstructed and summed,
the normalized sum supplies the local quantization-axis proxy, and all 12
cooling/repump polarizations are projected into that spherical basis before
the inherited 24-state rate solve.

- trapping scale: `{summary['trapping_power_mw_per_path']:.9f} mW/path`
- central transition-equivalent gradient target: `20 G/cm`
- primary timestep: `{summary['primary_time_step_us']:.3f} us`
- duration: `{summary['duration_ms']:.3f} ms`
- gravity: on; recoil diffusion: off

The axial launch is classified `{axial['capture']['classification']}` with a
minimum radius of `{axial['minimum_radius_mm']:.6f} mm`. The nonsymmetric
three-dimensional launch is classified `{oblique['capture']['classification']}`
with a minimum radius of `{oblique['minimum_radius_mm']:.6f} mm`.

The axial ray crosses the ideal 2.234-micrometre waist. Consequently its
sampled field proxy reaches `{axial['maximum_sampled_effective_field_proxy_g']:.6g} G`
and the local-axis history contains large jumps. That peak is not timestep
resolved and must not be interpreted as a quantitative physical field.

The CSV files retain SI trajectory quantities, all six reconstructed beamwise
field vectors, and all sigma+/pi/sigma- fractions for the 12 cooling/repump
and six trapping components at every stored sample.

These runs omit scalar/tensor shifts by the explicitly imposed ideal-magic
assumption, conservative 1529-nm force, trap-light scattering/heating/loss,
coherent interference, measured Jones transformations, and nonadiabatic
dynamics at the fictitious-field zero. A heuristic core-residence result is
not evidence of a dynamically stable three-dimensional pMOT.
"""
    path.write_text(text, encoding="utf-8")


def run() -> dict:
    root = _project_root() / "outputs" / "diagnostics" / "pmot" / CAMPAIGN_NAME
    data_root = root / "data"
    figure_root = root / "figures"
    data_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    context = build_vector_only_trajectory_context(
        trapping_power_w_per_path=None,
        target_gradient_g_per_cm=20.0,
    )
    cases = _cases()
    primary_config = RateEquationTrajectoryConfig(
        time_step_s=PRIMARY_TIME_STEP_S,
        include_diffusion=False,
        escape_radius_m=30.0e-3,
    )
    coarse_config = RateEquationTrajectoryConfig(
        time_step_s=COARSE_TIME_STEP_S,
        include_diffusion=False,
        escape_radius_m=30.0e-3,
    )
    records = {}
    summaries = {}
    coarse = {}
    for case_index, (name, initial_state) in enumerate(cases.items(), start=1):
        print(f"[trajectory {case_index}/2] {name}: primary 2.5 us run", flush=True)
        start = perf_counter()

        def progress(done, total, elapsed):
            if done == total or done % max(1, total // 10) == 0:
                print(
                    f"[{name}] {done}/{total} ({100.0*done/total:.0f}%), "
                    f"t={1.0e3*elapsed:.3f} ms",
                    flush=True,
                )

        record = simulate_vector_only_pmot_trajectory(
            initial_state,
            duration_s=DURATION_S,
            context=context,
            trajectory_config=primary_config,
            progress_callback=progress,
        )
        wall_s = perf_counter() - start
        (
            fields_t,
            field_error_t,
            component_sum_roundoff_t,
        ) = reconstruct_component_fields_t(record, context)
        weights, beam_records, weight_error = polarization_history(record, context)
        _trajectory_frame(record, context).to_csv(
            data_root / f"{name}_trajectory.csv",
            index=False,
        )
        _field_frame(record, context, fields_t).to_csv(
            data_root / f"{name}_beamwise_effective_fields.csv",
            index=False,
        )
        _polarization_frame(record, beam_records, weights).to_csv(
            data_root / f"{name}_polarization_weights.csv",
            index=False,
        )
        plot_pmot_trajectory_diagnostics(
            record,
            list(context.cooling_repump_beams),
            list(context.trapping_beams),
            path=figure_root / f"{name}_trajectory_and_beams.png",
            title=f"{name}: local-effective-field-axis pMOT diagnostic",
        )
        plt.close("all")
        _plot_field_axis_and_polarization(
            record,
            context,
            weights,
            beam_records,
            f"{name}: beamwise field sum, local axis, and spherical projections",
            figure_root / f"{name}_field_axis_and_polarization.png",
        )
        records[name] = record
        summaries[name] = _summary(
            record,
            fields_t,
            field_error_t,
            component_sum_roundoff_t,
            weight_error,
            wall_s,
        )

        print(f"[trajectory {case_index}/2] {name}: coarse 5 us QA rerun", flush=True)
        coarse_record = simulate_vector_only_pmot_trajectory(
            initial_state,
            duration_s=DURATION_S,
            context=context,
            trajectory_config=coarse_config,
        )
        coarse[name] = _coarse_summary(coarse_record)
        summaries[name]["coarse_5us_comparison"] = {
            "capture_classification": coarse_record.capture.classification,
            "classification_matches_primary": (
                coarse_record.capture.classification
                == record.capture.classification
            ),
            "final_position_difference_mm": 1.0e3
            * float(
                np.linalg.norm(
                    np.asarray(coarse_record.rate_equation.positions_m[-1])
                    - np.asarray(record.rate_equation.positions_m[-1])
                )
            ),
            "final_velocity_difference_m_per_s": float(
                np.linalg.norm(
                    np.asarray(coarse_record.rate_equation.velocities_m_per_s[-1])
                    - np.asarray(record.rate_equation.velocities_m_per_s[-1])
                )
            ),
            "minimum_radius_difference_mm": 1.0e3
            * abs(
                coarse[name]["minimum_radius_m"]
                - 1.0e-3 * summaries[name]["minimum_radius_mm"]
            ),
        }
        print(
            f"[{name}] result={record.capture.classification}, "
            f"min radius={summaries[name]['minimum_radius_mm']:.6f} mm",
            flush=True,
        )

    _plot_comparison(
        records,
        figure_root / "two_trajectory_comparison.png",
    )
    summary = {
        "status": "PROVISIONAL_VECTOR_ONLY_TRAJECTORY_DIAGNOSTIC",
        "model": (
            "stretched-transition-equivalent local field axis with ideal scalar/tensor cancellation"
        ),
        "trapping_power_mw_per_path": 1.0e3 * context.trapping_power_w_per_path,
        "trapping_power_source": context.trapping_power_source,
        "retro_power_fraction": context.apparatus.trapping_laser.retro_power_fraction,
        "target_central_gradient_g_per_cm": context.target_gradient_g_per_cm,
        "cooling_power_mw_per_component": 1.0e3
        * context.apparatus.mot_light.cooling.power_w_per_beam,
        "repump_power_mw_per_component": 1.0e3
        * context.apparatus.mot_light.repump.power_w_per_beam,
        "cooling_detuning_mhz": 1.0e-6
        * context.apparatus.mot_light.cooling.detuning_hz,
        "duration_ms": 1.0e3 * DURATION_S,
        "primary_time_step_us": 1.0e6 * PRIMARY_TIME_STEP_S,
        "coarse_qa_time_step_us": 1.0e6 * COARSE_TIME_STEP_S,
        "gravity_enabled": context.multilevel_config.include_gravity,
        "recoil_diffusion_enabled": False,
        "external_magnetic_field_t": [0.0, 0.0, 0.0],
        "path_helicities_xyz": ["sigma+", "sigma+", "sigma-"],
        "trajectories": summaries,
        "physics_omitted": [
            "level-resolved 24-state Stark Hamiltonian",
            "scalar/tensor shifts outside the imposed ideal-magic approximation",
            "conservative Stark-gradient force",
            "1529-nm scattering, heating, and loss",
            "coherent standing-wave interference",
            "measured window/mirror Jones transformations",
            "nonadiabatic dynamics at the fictitious-field zero",
        ],
    }
    summary = _json_ready(summary)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(root / "README.md", summary)
    print(f"outputs: {root.resolve()}", flush=True)
    return summary


if __name__ == "__main__":
    run()

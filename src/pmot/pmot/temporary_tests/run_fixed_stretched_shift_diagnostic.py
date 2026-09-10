"""Run the isolated fixed-stretched-transition pMOT diagnostic campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...configuration import PLANCK_CONSTANT_J_S
from ...configuration import SPEED_OF_LIGHT_M_PER_S
from ...configuration import VACUUM_PERMITTIVITY_F_PER_M
from ...mot_multilevel.configuration import default_multilevel_mot_config
from ...mot_multilevel.rate_equations import build_rate_equation_model
from ...mot_multilevel.polarization import propagation_frame_polarization
from ..ac_stark import ProvisionalStarkConfig
from ..ac_stark import build_physics_trapping_beams
from ..ac_stark import provisional_power_for_target_gradient_w_per_path
from ..configuration import build_pmot_cooling_and_repump_beams
from ..configuration import default_pmot_apparatus_config
from ..polarizability import interpolate_differential_polarizability_arrays
from ..polarizability import load_differential_polarizability_table
from ..trapping_beams import helicity_sign
from .fixed_stretched_shift import ANGULAR_FREQUENCY_EFFECTIVE_DETUNING_EQUATION
from .fixed_stretched_shift import ORDINARY_FREQUENCY_EFFECTIVE_DETUNING_EQUATION
from .fixed_stretched_shift import STRETCHED_TRANSITION_LABEL
from .fixed_stretched_shift import cooling_effective_detunings_hz
from .fixed_stretched_shift import evaluate_fixed_stretched_transition_shift


CAMPAIGN_NAME = "temporary_stretched_transition_1529nm_20260908"
POSITION_EXTENT_M = 2.0e-3
VELOCITY_EXTENT_M_PER_S = 25.0
LINEOUT_SAMPLE_COUNT = 401
REFERENCE_POSITION_M = (0.35e-3, -0.20e-3, 0.10e-3)
REFERENCE_VELOCITY_M_PER_S = (17.0, -8.0, 4.0)
REFERENCE_QUANTIZATION_AXIS = (0.0, 0.0, 1.0)
AXES = {
    "x": np.asarray((1.0, 0.0, 0.0)),
    "y": np.asarray((0.0, 1.0, 0.0)),
    "z": np.asarray((0.0, 0.0, 1.0)),
}
AXIS_NAMES = {
    "x": "horizontal_x",
    "y": "horizontal_y",
    "z": "vertical_z",
}
COLORS = {
    "scalar": "#2563eb",
    "vector": "#dc2626",
    "tensor": "#16a34a",
    "total": "#111827",
    "residual": "#9333ea",
    "fixed": "#d97706",
    "centered": "#0f766e",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _short_beam_label(beam) -> str:
    axis = {
        "horizontal_x": "x",
        "horizontal_y": "y",
        "vertical_z": "z",
    }[beam.axis_name]
    direction = "+k" if beam.propagation_sense == "incident" else "-k"
    return f"{axis} {direction} ({beam.helicity})"


def _build_context(power_w_per_path: float | None):
    apparatus = default_pmot_apparatus_config()
    rate_config = replace(
        default_multilevel_mot_config(),
        repumper_enabled=True,
        repump_power_w_per_beam=apparatus.mot_light.repump.power_w_per_beam,
    )
    model = build_rate_equation_model(rate_config.natural_linewidth_rad_per_s)
    table = load_differential_polarizability_table()
    if power_w_per_path is None:
        selected_power = provisional_power_for_target_gradient_w_per_path(
            model,
            apparatus.trapping_laser,
            target_gradient_g_per_cm=20.0,
            polarizability_table=table,
        )
        power_source = (
            "historical stretched-reference 20 G/cm vector-gradient proxy; "
            "demonstration scale only"
        )
    else:
        if not np.isfinite(power_w_per_path) or power_w_per_path <= 0.0:
            raise ValueError("power_w_per_path must be finite and positive")
        selected_power = float(power_w_per_path)
        power_source = "explicit command-line demonstration power"
    stark = ProvisionalStarkConfig.uniform_power(
        selected_power,
        incident_helicities_by_axis=("sigma+", "sigma+", "sigma-"),
        retro_helicities_by_axis=("sigma+", "sigma+", "sigma-"),
    )
    trapping_beams = build_physics_trapping_beams(apparatus.trapping_laser, stark)
    cooling_repump_beams = build_pmot_cooling_and_repump_beams(apparatus, rate_config)
    return {
        "apparatus": apparatus,
        "rate_config": rate_config,
        "model": model,
        "table": table,
        "power_w_per_path": selected_power,
        "power_source": power_source,
        "stark": stark,
        "trapping_beams": trapping_beams,
        "cooling_repump_beams": cooling_repump_beams,
    }


def _evaluate(context, position, velocity, axis):
    return evaluate_fixed_stretched_transition_shift(
        context["trapping_beams"],
        position,
        velocity,
        axis,
        context["apparatus"].trapping_laser,
        context["stark"],
        context["table"],
    )


def _doppler_scan(context, data_root: Path, figure_root: Path):
    speeds = np.linspace(
        -VELOCITY_EXTENT_M_PER_S,
        VELOCITY_EXTENT_M_PER_S,
        LINEOUT_SAMPLE_COUNT,
    )
    direction = np.asarray((1.0, -2.0, 0.5), dtype=float)
    direction /= np.linalg.norm(direction)
    beams = context["trapping_beams"]
    beam_directions = np.asarray([beam.direction for beam in beams])
    lab_wavelengths_m = np.asarray([beam.wavelength_m for beam in beams])
    projected = speeds[:, None] * (beam_directions @ direction)[None, :]
    doppler_hz = -projected / lab_wavelengths_m[None, :]
    wavelengths_nm = 1.0e9 * lab_wavelengths_m[None, :] / (
        1.0 - projected / SPEED_OF_LIGHT_M_PER_S
    )
    wavelength_offsets_fm = 1.0e6 * (
        wavelengths_nm - 1.0e9 * lab_wavelengths_m[None, :]
    )
    columns = {"signed_speed_along_scan_m_per_s": speeds}
    for index, beam in enumerate(beams):
        tag = beam.label
        columns[f"{tag}_projected_speed_m_per_s"] = projected[:, index]
        columns[f"{tag}_doppler_shift_hz"] = doppler_hz[:, index]
        columns[f"{tag}_atom_frame_wavelength_nm"] = wavelengths_nm[:, index]
    frame = pd.DataFrame(columns)
    csv_path = data_root / "six_component_doppler_scan.csv"
    frame.to_csv(csv_path, index=False)

    figure, axes = plt.subplots(2, 1, figsize=(11.2, 8.2), sharex=True)
    palette = plt.get_cmap("tab10")
    for index, beam in enumerate(beams):
        label = _short_beam_label(beam)
        color = palette(index)
        axes[0].plot(speeds, doppler_hz[:, index] / 1.0e6, color=color, label=label)
        axes[1].plot(speeds, wavelength_offsets_fm[:, index], color=color, label=label)
    axes[0].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[0].set_ylabel("Atom-frame frequency shift [MHz]")
    axes[1].set_ylabel(r"Atom-frame wavelength shift [fm]")
    axes[1].set_xlabel(
        r"Signed speed along $(1,-2,0.5)/\sqrt{5.25}$ [m/s]"
    )
    axes[0].legend(ncol=3, frameon=False, loc="upper center")
    figure.suptitle(
        "Six-component 1529-nm Doppler evaluation\n"
        r"Each curve uses its own $\hat{k}_j\!\cdot\!\mathbf{v}$; first-order nonrelativistic model",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure_path = _save_figure(
        figure,
        figure_root / "01_six_component_doppler_shift.png",
    )
    return frame, csv_path, figure_path


def _reference_component_ledger(context, data_root: Path, figure_root: Path):
    observable = _evaluate(
        context,
        REFERENCE_POSITION_M,
        REFERENCE_VELOCITY_M_PER_S,
        REFERENCE_QUANTIZATION_AXIS,
    )
    power = context["power_w_per_path"]
    rows = []
    for index, beam in enumerate(context["trapping_beams"]):
        epsilon = np.asarray(
            propagation_frame_polarization(beam.direction, beam.helicity),
            dtype=complex,
        )
        rows.append(
            {
                "beam_index": index,
                "beam_label": beam.label,
                "short_label": _short_beam_label(beam),
                "axis_name": beam.axis_name,
                "propagation_sense": beam.propagation_sense,
                "helicity": beam.helicity,
                "helicity_sign": helicity_sign(beam.helicity),
                "khat_x": beam.direction[0],
                "khat_y": beam.direction[1],
                "khat_z": beam.direction[2],
                "projected_speed_m_per_s": observable.projected_speeds_m_per_s[index],
                "lab_frequency_hz": observable.lab_frequencies_hz[index],
                "atom_frame_frequency_hz": observable.atom_frame_frequencies_hz[index],
                "trapping_doppler_shift_hz": observable.trapping_doppler_shifts_hz[index],
                "atom_frame_wavelength_nm": observable.atom_frame_wavelengths_nm[index],
                "intensity_w_per_m2": observable.component_intensities_w_per_m2[index],
                "intensity_per_watt_path_m_inv2": (
                    observable.component_intensities_w_per_m2[index] / power
                ),
                "field_squared_v2_per_m2": observable.component_field_squared_v2_per_m2[index],
                "alpha_scalar_assumed_si": observable.scalar_polarizability_assumed_si[index],
                "alpha_vector_assumed_si": observable.vector_polarizability_assumed_si[index],
                "alpha_tensor_assumed_si": observable.tensor_polarizability_assumed_si[index],
                "epsilon_x_real": epsilon[0].real,
                "epsilon_x_imag": epsilon[0].imag,
                "epsilon_y_real": epsilon[1].real,
                "epsilon_y_imag": epsilon[1].imag,
                "epsilon_z_real": epsilon[2].real,
                "epsilon_z_imag": epsilon[2].imag,
                "vector_angular_factor": observable.component_vector_angular_factor[index],
                "tensor_geometry_factor": observable.component_tensor_angular_factor[index],
                "scalar_energy_j": observable.component_scalar_energy_j[index],
                "vector_energy_j": observable.component_vector_energy_j[index],
                "tensor_energy_j": observable.component_tensor_energy_j[index],
                "total_energy_j": observable.component_total_energy_j[index],
                "scalar_shift_hz": observable.component_scalar_shift_hz[index],
                "vector_shift_hz": observable.component_vector_shift_hz[index],
                "tensor_shift_hz": observable.component_tensor_shift_hz[index],
                "total_shift_hz": observable.component_total_shift_hz[index],
                "scalar_shift_hz_per_watt_path": observable.component_scalar_shift_hz[index] / power,
                "vector_shift_hz_per_watt_path": observable.component_vector_shift_hz[index] / power,
                "tensor_shift_hz_per_watt_path": observable.component_tensor_shift_hz[index] / power,
                "total_shift_hz_per_watt_path": observable.component_total_shift_hz[index] / power,
            }
        )
    frame = pd.DataFrame(rows)
    csv_path = data_root / "reference_phase_space_component_ledger.csv"
    frame.to_csv(csv_path, index=False)

    labels = frame["short_label"].tolist()
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.8), sharex=True)
    axes[0].bar(
        x,
        frame["intensity_w_per_m2"] / 1.0e3,
        color="#0891b2",
        edgecolor="white",
    )
    axes[0].set_ylabel(r"Local intensity [kW m$^{-2}$]")
    width = 0.19
    for offset, key, label in (
        (-1.5, "scalar_shift_hz", "scalar"),
        (-0.5, "vector_shift_hz", "vector"),
        (0.5, "tensor_shift_hz", "tensor"),
        (1.5, "total_shift_hz", "total"),
    ):
        axes[1].bar(
            x + offset * width,
            frame[key] / 1.0e6,
            width=width,
            color=COLORS[label],
            label=label.capitalize(),
        )
    axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[1].set_ylabel("Differential shift contribution [MHz]")
    axes[1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1].legend(ncol=4, frameon=False)
    figure.suptitle(
        "Beamwise stretched-transition energy ledger\n"
        "r=(0.35,-0.20,0.10) mm, v=(17,-8,4) m/s, fixed quantization axis z",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure_path = _save_figure(
        figure,
        figure_root / "02_reference_phase_space_component_budget.png",
    )
    return observable, frame, csv_path, figure_path


def _magic_coefficient_scan(context, data_root: Path, figure_root: Path):
    velocities = np.linspace(
        -VELOCITY_EXTENT_M_PER_S,
        VELOCITY_EXTENT_M_PER_S,
        LINEOUT_SAMPLE_COUNT,
    )
    wavelength_m = context["apparatus"].trapping_laser.wavelength_m
    wavelengths_nm = 1.0e9 * wavelength_m / (
        1.0 - velocities / SPEED_OF_LIGHT_M_PER_S
    )
    alpha0, alpha1, alpha2 = interpolate_differential_polarizability_arrays(
        wavelengths_nm,
        context["table"],
    )
    common = 1.0e5 / (
        SPEED_OF_LIGHT_M_PER_S
        * VACUUM_PERMITTIVITY_F_PER_M
        * PLANCK_CONSTANT_J_S
        * 1.0e6
    )
    beta0 = -2.0 * alpha0 * common
    beta1 = -2.0 * alpha1 * common
    beta2 = alpha2 * common
    residual = beta0 + beta2
    residual_fraction = residual / np.maximum(np.abs(beta1), np.finfo(float).tiny)
    frame = pd.DataFrame(
        {
            "projected_speed_m_per_s": velocities,
            "atom_frame_wavelength_nm": wavelengths_nm,
            "scalar_mhz_per_1e5_w_per_m2": beta0,
            "vector_mhz_per_1e5_w_per_m2": beta1,
            "tensor_aligned_mhz_per_1e5_w_per_m2": beta2,
            "scalar_plus_tensor_mhz_per_1e5_w_per_m2": residual,
            "scalar_plus_tensor_over_abs_vector": residual_fraction,
        }
    )
    csv_path = data_root / "single_component_magic_coefficient_scan.csv"
    frame.to_csv(csv_path, index=False)

    figure, axes = plt.subplots(2, 1, figsize=(10.8, 8.3), sharex=True)
    axes[0].plot(velocities, beta0, color=COLORS["scalar"], label="Scalar")
    axes[0].plot(velocities, beta1, color=COLORS["vector"], label="Vector")
    axes[0].plot(velocities, beta2, color=COLORS["tensor"], label="Tensor (aligned stretched)")
    axes[0].set_ylabel(r"Coefficient [MHz per $10^5$ W m$^{-2}$]")
    axes[0].legend(ncol=3, frameon=False)
    axes[1].plot(velocities, residual, color=COLORS["residual"], linewidth=2.0)
    axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
    for speed in (-17.0, 0.0, 17.0):
        axes[1].axvline(speed, color="#94a3b8", linewidth=0.7, linestyle="--")
    axes[1].set_ylabel(r"Scalar + tensor residual [MHz per $10^5$ W m$^{-2}$]")
    axes[1].set_xlabel(r"Projected atom speed $\hat{k}\!\cdot\!\mathbf{v}$ [m/s]")
    figure.suptitle(
        "Doppler sensitivity of the single-component magic condition\n"
        "Aligned circular field and fixed stretched-state basis",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure_path = _save_figure(
        figure,
        figure_root / "03_single_component_magic_condition.png",
    )
    return frame, csv_path, figure_path


def _position_lineouts(context, data_root: Path, figure_root: Path):
    coordinates = np.linspace(-POSITION_EXTENT_M, POSITION_EXTENT_M, LINEOUT_SAMPLE_COUNT)
    bare_resonance_hz = context["apparatus"].mot_light.cooling.resonance_frequency_hz
    origin_shifts = {
        label: _evaluate(context, np.zeros(3), np.zeros(3), axis).total_frequency_shift_hz
        for label, axis in AXES.items()
    }
    power = context["power_w_per_path"]
    rows = []
    for label, axis in AXES.items():
        for coordinate in coordinates:
            position = coordinate * axis
            observable = _evaluate(context, position, np.zeros(3), axis)
            fixed_detuning = cooling_effective_detunings_hz(
                context["cooling_repump_beams"],
                np.zeros(3),
                observable.total_frequency_shift_hz,
            )[0]
            centered_detuning = cooling_effective_detunings_hz(
                context["cooling_repump_beams"],
                np.zeros(3),
                observable.total_frequency_shift_hz,
                cooling_carrier_offset_hz=origin_shifts[label],
            )[0]
            rows.append(
                {
                    "axis": label,
                    "coordinate_m": coordinate,
                    "coordinate_mm": 1.0e3 * coordinate,
                    "quantization_axis_x": axis[0],
                    "quantization_axis_y": axis[1],
                    "quantization_axis_z": axis[2],
                    "total_intensity_w_per_m2": observable.total_intensity_w_per_m2,
                    "total_intensity_per_watt_path_m_inv2": (
                        observable.total_intensity_w_per_m2 / power
                    ),
                    "scalar_shift_hz": observable.total_scalar_shift_hz,
                    "vector_shift_hz": observable.total_vector_shift_hz,
                    "tensor_shift_hz": observable.total_tensor_shift_hz,
                    "total_shift_hz": observable.total_frequency_shift_hz,
                    "scalar_shift_hz_per_watt_path": observable.total_scalar_shift_hz / power,
                    "vector_shift_hz_per_watt_path": observable.total_vector_shift_hz / power,
                    "tensor_shift_hz_per_watt_path": observable.total_tensor_shift_hz / power,
                    "total_shift_hz_per_watt_path": observable.total_frequency_shift_hz / power,
                    "bare_stretched_transition_frequency_hz": bare_resonance_hz,
                    "shifted_stretched_transition_frequency_hz": (
                        bare_resonance_hz + observable.total_frequency_shift_hz
                    ),
                    "fixed_carrier_effective_detuning_hz": fixed_detuning,
                    "center_compensated_effective_detuning_hz": centered_detuning,
                }
            )
    frame = pd.DataFrame(rows)
    csv_path = data_root / "fixed_axis_position_lineouts.csv"
    frame.to_csv(csv_path, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for plot_axis, label in zip(axes, AXES):
        selected = frame[frame["axis"] == label]
        for key, legend in (
            ("scalar_shift_hz", "Scalar"),
            ("vector_shift_hz", "Vector (signed)"),
            ("tensor_shift_hz", "Tensor"),
            ("total_shift_hz", "Total"),
        ):
            color_key = key.split("_")[0]
            plot_axis.plot(
                selected["coordinate_mm"],
                selected[key] / 1.0e6,
                color=COLORS[color_key],
                linewidth=2.0 if color_key == "total" else 1.4,
                label=legend,
            )
        plot_axis.axvline(0.0, color="#64748b", linewidth=0.8)
        plot_axis.axhline(0.0, color="#64748b", linewidth=0.8)
        plot_axis.scatter([0.0], [origin_shifts[label] / 1.0e6], color="black", s=22, zorder=5)
        plot_axis.set_title(f"{label}-axis lineout; fixed n = {label}")
        plot_axis.set_xlabel(f"{label} [mm]")
    axes[0].set_ylabel("Stretched-transition differential shift [MHz]")
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(
        "All-six-beam scalar, signed-vector, tensor, and total shift\n"
        "Each panel is a separate fixed-basis diagnostic, not three simultaneous atomic axes",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    shift_path = _save_figure(
        figure,
        figure_root / "04_fixed_axis_shift_position_lineouts.png",
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for plot_axis, label in zip(axes, AXES):
        selected = frame[frame["axis"] == label]
        plot_axis.plot(
            selected["coordinate_mm"],
            selected["fixed_carrier_effective_detuning_hz"] / 1.0e6,
            color=COLORS["fixed"],
            label="Carrier fixed at bare -15 MHz",
        )
        plot_axis.plot(
            selected["coordinate_mm"],
            selected["center_compensated_effective_detuning_hz"] / 1.0e6,
            color=COLORS["centered"],
            label="Carrier recentered to shifted origin",
        )
        plot_axis.axhline(0.0, color="#dc2626", linewidth=1.0, linestyle="--", label="Resonance")
        plot_axis.axhline(-15.0, color="#64748b", linewidth=0.8, linestyle=":", label="-15 MHz target")
        plot_axis.axvline(0.0, color="#64748b", linewidth=0.8)
        plot_axis.set_title(f"{label}-axis; fixed n = {label}")
        plot_axis.set_xlabel(f"{label} [mm]")
    axes[0].set_ylabel("Cooling effective detuning [MHz]")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Shifted 780-nm stretched resonance seen by the cooling carrier\n"
        r"$\delta\nu_{\rm eff}=\delta\nu_L-\hat{k}_{780}\!\cdot\!\mathbf{v}/\lambda_{780}-\Delta E_{AC}/h$; v=0",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    detuning_path = _save_figure(
        figure,
        figure_root / "05_shifted_780_resonance_effective_detuning.png",
    )
    return frame, origin_shifts, csv_path, shift_path, detuning_path


def _pair_magic_scan(context, data_root: Path, figure_root: Path):
    velocities = np.linspace(
        -VELOCITY_EXTENT_M_PER_S,
        VELOCITY_EXTENT_M_PER_S,
        LINEOUT_SAMPLE_COUNT,
    )
    rows = []
    for label, axis in AXES.items():
        indices = [
            index
            for index, beam in enumerate(context["trapping_beams"])
            if beam.axis_name == AXIS_NAMES[label]
        ]
        for speed in velocities:
            observable = _evaluate(context, np.zeros(3), speed * axis, axis)
            scalar = float(np.sum(observable.component_scalar_shift_hz[indices]))
            vector = float(np.sum(observable.component_vector_shift_hz[indices]))
            tensor = float(np.sum(observable.component_tensor_shift_hz[indices]))
            rows.append(
                {
                    "axis": label,
                    "speed_m_per_s": speed,
                    "pair_scalar_shift_hz": scalar,
                    "pair_vector_shift_hz": vector,
                    "pair_tensor_shift_hz": tensor,
                    "pair_scalar_plus_tensor_shift_hz": scalar + tensor,
                    "pair_total_shift_hz": scalar + vector + tensor,
                }
            )
    frame = pd.DataFrame(rows)
    csv_path = data_root / "aligned_axis_pair_magic_scan.csv"
    frame.to_csv(csv_path, index=False)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 7.4),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )
    for column, label in enumerate(AXES):
        selected = frame[frame["axis"] == label]
        top_axis = axes[0, column]
        residual_axis = axes[1, column]
        top_axis.plot(
            selected["speed_m_per_s"],
            selected["pair_scalar_shift_hz"] / 1.0e6,
            color=COLORS["scalar"],
            label="Scalar",
        )
        top_axis.plot(
            selected["speed_m_per_s"],
            selected["pair_tensor_shift_hz"] / 1.0e6,
            color=COLORS["tensor"],
            label="Tensor",
        )
        top_axis.axhline(0.0, color="#64748b", linewidth=0.8)
        top_axis.axvline(0.0, color="#64748b", linewidth=0.8)
        top_axis.set_title(f"{label} incident/retro pair; n = {label}")
        residual_axis.plot(
            selected["speed_m_per_s"],
            selected["pair_scalar_plus_tensor_shift_hz"] / 1.0e3,
            color=COLORS["residual"],
            linewidth=2.0,
            label="Scalar + tensor",
        )
        residual_axis.axhline(0.0, color="#64748b", linewidth=0.8)
        residual_axis.axvline(0.0, color="#64748b", linewidth=0.8)
        residual_axis.set_xlabel("Axial speed [m/s]")
    axes[0, 0].set_ylabel("Scalar or tensor pair term [MHz]")
    axes[1, 0].set_ylabel("Scalar + tensor residual [kHz]")
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[1, 0].legend(frameon=False)
    figure.suptitle(
        "Aligned-pair scalar/tensor cancellation under beamwise Doppler shifts\n"
        "Counterpropagating pair sum suppresses the leading odd-in-velocity error",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure_path = _save_figure(
        figure,
        figure_root / "06_aligned_pair_magic_cancellation.png",
    )
    return frame, csv_path, figure_path


def _origin_budget(context, data_root: Path, figure_root: Path):
    observable = _evaluate(context, np.zeros(3), np.zeros(3), (0.0, 0.0, 1.0))
    power = context["power_w_per_path"]
    rows = []
    for label in AXES:
        indices = [
            index
            for index, beam in enumerate(context["trapping_beams"])
            if beam.axis_name == AXIS_NAMES[label]
        ]
        scalar = float(np.sum(observable.component_scalar_shift_hz[indices]))
        vector = float(np.sum(observable.component_vector_shift_hz[indices]))
        tensor = float(np.sum(observable.component_tensor_shift_hz[indices]))
        rows.append(
            {
                "axis_pair": label,
                "fixed_quantization_axis": "z",
                "scalar_shift_hz": scalar,
                "vector_shift_hz": vector,
                "tensor_shift_hz": tensor,
                "scalar_plus_tensor_shift_hz": scalar + tensor,
                "total_shift_hz": scalar + vector + tensor,
                "scalar_shift_hz_per_watt_path": scalar / power,
                "vector_shift_hz_per_watt_path": vector / power,
                "tensor_shift_hz_per_watt_path": tensor / power,
                "total_shift_hz_per_watt_path": (scalar + vector + tensor) / power,
            }
        )
    rows.append(
        {
            "axis_pair": "all six",
            "fixed_quantization_axis": "z",
            "scalar_shift_hz": observable.total_scalar_shift_hz,
            "vector_shift_hz": observable.total_vector_shift_hz,
            "tensor_shift_hz": observable.total_tensor_shift_hz,
            "scalar_plus_tensor_shift_hz": (
                observable.total_scalar_shift_hz + observable.total_tensor_shift_hz
            ),
            "total_shift_hz": observable.total_frequency_shift_hz,
            "scalar_shift_hz_per_watt_path": observable.total_scalar_shift_hz / power,
            "vector_shift_hz_per_watt_path": observable.total_vector_shift_hz / power,
            "tensor_shift_hz_per_watt_path": observable.total_tensor_shift_hz / power,
            "total_shift_hz_per_watt_path": observable.total_frequency_shift_hz / power,
        }
    )
    frame = pd.DataFrame(rows)
    csv_path = data_root / "origin_cancellation_budget.csv"
    frame.to_csv(csv_path, index=False)

    x = np.arange(len(frame))
    width = 0.21
    figure, axis = plt.subplots(figsize=(10.8, 5.5))
    for offset, key, label in (
        (-1.0, "scalar_shift_hz", "Scalar"),
        (0.0, "tensor_shift_hz", "Tensor"),
        (1.0, "scalar_plus_tensor_shift_hz", "Scalar + tensor"),
    ):
        axis.bar(
            x + offset * width,
            frame[key] / 1.0e6,
            width=width,
            label=label,
            color={"Scalar": COLORS["scalar"], "Tensor": COLORS["tensor"], "Scalar + tensor": COLORS["residual"]}[label],
        )
    axis.axhline(0.0, color="#64748b", linewidth=0.8)
    axis.set_xticks(x, ["x pair", "y pair", "z pair", "all six"])
    axis.set_ylabel("Differential shift [MHz]")
    axis.legend(frameon=False, ncol=3)
    axis.set_title(
        "Why the single-beam magic cancellation does not survive three axes\n"
        "Atom at rest at the origin; named stretched transition uses fixed lab z"
    )
    figure.tight_layout()
    figure_path = _save_figure(
        figure,
        figure_root / "07_origin_scalar_tensor_cancellation_budget.png",
    )
    return observable, frame, csv_path, figure_path


def _axis_aligned_pair_comparison(context, data_root: Path, figure_root: Path):
    """Compare each path pair in a quantization basis aligned to that path."""

    coordinates = np.linspace(-0.5e-3, 0.5e-3, 201)
    power = context["power_w_per_path"]
    rows = []
    for label, axis in AXES.items():
        indices = [
            index
            for index, beam in enumerate(context["trapping_beams"])
            if beam.axis_name == AXIS_NAMES[label]
        ]
        for coordinate in coordinates:
            observable = _evaluate(context, coordinate * axis, np.zeros(3), axis)
            scalar = float(np.sum(observable.component_scalar_shift_hz[indices]))
            vector = float(np.sum(observable.component_vector_shift_hz[indices]))
            tensor = float(np.sum(observable.component_tensor_shift_hz[indices]))
            rows.append(
                {
                    "axis_pair": label,
                    "quantization_axis": label,
                    "coordinate_m": coordinate,
                    "coordinate_mm": 1.0e3 * coordinate,
                    "scalar_shift_hz": scalar,
                    "vector_shift_hz": vector,
                    "tensor_shift_hz": tensor,
                    "scalar_plus_tensor_shift_hz": scalar + tensor,
                    "total_shift_hz": scalar + vector + tensor,
                    "scalar_shift_hz_per_watt_path": scalar / power,
                    "vector_shift_hz_per_watt_path": vector / power,
                    "tensor_shift_hz_per_watt_path": tensor / power,
                    "total_shift_hz_per_watt_path": (scalar + vector + tensor) / power,
                }
            )
    frame = pd.DataFrame(rows)
    csv_path = data_root / "axis_aligned_pair_cancellation_comparison.csv"
    frame.to_csv(csv_path, index=False)

    origin_rows = frame[
        np.isclose(frame["coordinate_m"], 0.0, rtol=0.0, atol=1.0e-15)
    ]
    axis_labels = origin_rows["axis_pair"].tolist()
    x_positions = np.arange(len(axis_labels))
    width = 0.34
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    axes[0].bar(
        x_positions - width / 2.0,
        origin_rows["scalar_shift_hz"] / 1.0e6,
        width=width,
        color=COLORS["scalar"],
        label="Scalar",
    )
    axes[0].bar(
        x_positions + width / 2.0,
        origin_rows["tensor_shift_hz"] / 1.0e6,
        width=width,
        color=COLORS["tensor"],
        label="Tensor",
    )
    axes[0].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[0].set_xticks(
        x_positions,
        [f"{label} pair\nn={label}" for label in axis_labels],
    )
    axes[0].set_ylabel("Pair contribution at origin [MHz]")
    axes[0].set_title("Equal and opposite large terms")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].bar(
        x_positions,
        origin_rows["scalar_plus_tensor_shift_hz"],
        width=0.5,
        color=COLORS["residual"],
    )
    axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[1].set_xticks(
        x_positions,
        [f"{label} pair\nn={label}" for label in axis_labels],
    )
    axes[1].set_ylabel("Scalar + tensor residual [Hz]")
    axes[1].set_title("Magnified cancellation residual")

    line_styles = {
        "x": {"color": "#2563eb", "linestyle": "-", "linewidth": 3.0},
        "y": {"color": "#f97316", "linestyle": "--", "linewidth": 2.0},
        "z": {"color": "#16a34a", "linestyle": "-", "linewidth": 2.0},
    }
    for label in AXES:
        selected = frame[frame["axis_pair"] == label]
        axes[2].plot(
            selected["coordinate_mm"],
            selected["vector_shift_hz"] / 1.0e6,
            label=f"{label} pair; n={label}",
            **line_styles[label],
        )
    axes[2].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[2].axvline(0.0, color="#64748b", linewidth=0.8)
    axes[2].scatter([0.0], [0.0], color="black", s=22, zorder=5)
    axes[2].set_xlabel("Coordinate along corresponding beam axis [mm]")
    axes[2].set_ylabel("Signed vector pair shift [MHz]")
    axes[2].set_title("Odd vector term from displaced foci")
    axes[2].text(
        0.03,
        0.96,
        "x and y coincide exactly\n(y shown dashed)",
        transform=axes[2].transAxes,
        va="top",
        color="#475569",
        fontsize=8,
    )
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Axis-aligned stretched-state comparison: x, y, and z are equivalent\n"
        "For each pair separately, the quantization axis is chosen along that pair",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure_path = _save_figure(
        figure,
        figure_root / "08_axis_aligned_pair_cancellation_comparison.png",
    )
    return frame, origin_rows, csv_path, figure_path


def _cooling_beam_ledger(context, observable, origin_shift_hz: float, data_root: Path):
    bare_resonance_hz = context["apparatus"].mot_light.cooling.resonance_frequency_hz
    fixed = cooling_effective_detunings_hz(
        context["cooling_repump_beams"],
        REFERENCE_VELOCITY_M_PER_S,
        observable.total_frequency_shift_hz,
    )
    centered = cooling_effective_detunings_hz(
        context["cooling_repump_beams"],
        REFERENCE_VELOCITY_M_PER_S,
        observable.total_frequency_shift_hz,
        cooling_carrier_offset_hz=origin_shift_hz,
    )
    cooling = [
        beam for beam in context["cooling_repump_beams"] if beam.family == "cooling"
    ]
    rows = []
    for index, beam in enumerate(cooling):
        projected = float(np.dot(np.asarray(beam.direction), REFERENCE_VELOCITY_M_PER_S))
        rows.append(
            {
                "beam_index": index,
                "beam_label": beam.label,
                "axis_name": beam.axis_name,
                "propagation_sense": beam.propagation_sense,
                "polarization": beam.circular_polarization,
                "projected_speed_m_per_s": projected,
                "cooling_doppler_term_hz": projected / beam.wavelength_m,
                "configured_laser_detuning_hz": beam.detuning_hz,
                "hyperfine_offset_hz_for_reference": 0.0,
                "local_stretched_transition_shift_hz": observable.total_frequency_shift_hz,
                "bare_stretched_transition_frequency_hz": bare_resonance_hz,
                "shifted_stretched_transition_frequency_hz": (
                    bare_resonance_hz + observable.total_frequency_shift_hz
                ),
                "fixed_carrier_effective_detuning_hz": fixed[index],
                "center_compensated_effective_detuning_hz": centered[index],
            }
        )
    frame = pd.DataFrame(rows)
    path = data_root / "reference_phase_space_cooling_effective_detunings.csv"
    frame.to_csv(path, index=False)
    return frame, path


def _run_qa(context, reference, position_frame, magic_frame, origin):
    beams = context["trapping_beams"]
    analytic_doppler = -reference.projected_speeds_m_per_s / np.asarray(
        [beam.wavelength_m for beam in beams]
    )
    doppler_error = float(
        np.max(np.abs(reference.trapping_doppler_shifts_hz - analytic_doppler))
    )
    frequency_wavelength_error = float(
        np.max(
            np.abs(
                reference.atom_frame_frequencies_hz
                * reference.atom_frame_wavelengths_nm
                * 1.0e-9
                / SPEED_OF_LIGHT_M_PER_S
                - 1.0
            )
        )
    )
    component_sum_error_j = abs(
        reference.total_energy_j - float(np.sum(reference.component_total_energy_j))
    )
    joule_hertz_error = abs(
        reference.total_frequency_shift_hz
        - reference.total_energy_j / PLANCK_CONSTANT_J_S
    )

    zero_index = int(np.argmin(np.abs(magic_frame["projected_speed_m_per_s"])))
    zero_magic = magic_frame.iloc[zero_index]
    aligned_residual_fraction = abs(
        zero_magic["scalar_plus_tensor_over_abs_vector"]
    )

    parity = {}
    for label in AXES:
        selected = position_frame[position_frame["axis"] == label]
        vector = selected["vector_shift_hz"].to_numpy()
        scalar = selected["scalar_shift_hz"].to_numpy()
        tensor = selected["tensor_shift_hz"].to_numpy()
        scale = max(float(np.max(np.abs(vector))), np.finfo(float).tiny)
        parity[label] = {
            "vector_even_over_peak": float(np.max(np.abs(vector + vector[::-1])) / scale),
            "scalar_odd_over_peak": float(
                np.max(np.abs(scalar - scalar[::-1]))
                / max(float(np.max(np.abs(scalar))), np.finfo(float).tiny)
            ),
            "tensor_odd_over_peak": float(
                np.max(np.abs(tensor - tensor[::-1]))
                / max(float(np.max(np.abs(tensor))), np.finfo(float).tiny)
            ),
        }

    half_stark = replace(
        context["stark"],
        incident_path_powers_w=tuple(
            0.5 * power for power in context["stark"].incident_path_powers_w
        ),
    )
    half_beams = build_physics_trapping_beams(
        context["apparatus"].trapping_laser,
        half_stark,
    )
    half = evaluate_fixed_stretched_transition_shift(
        half_beams,
        REFERENCE_POSITION_M,
        REFERENCE_VELOCITY_M_PER_S,
        REFERENCE_QUANTIZATION_AXIS,
        context["apparatus"].trapping_laser,
        half_stark,
        context["table"],
    )
    power_linearity_error = abs(
        half.total_frequency_shift_hz / reference.total_frequency_shift_hz - 0.5
    )

    reversed_stark = replace(
        context["stark"],
        incident_helicities_by_axis=("sigma-", "sigma-", "sigma+"),
        retro_helicities_by_axis=("sigma-", "sigma-", "sigma+"),
    )
    reversed_beams = build_physics_trapping_beams(
        context["apparatus"].trapping_laser,
        reversed_stark,
    )
    reversed_observable = evaluate_fixed_stretched_transition_shift(
        reversed_beams,
        REFERENCE_POSITION_M,
        REFERENCE_VELOCITY_M_PER_S,
        REFERENCE_QUANTIZATION_AXIS,
        context["apparatus"].trapping_laser,
        reversed_stark,
        context["table"],
    )
    reversal = {
        "scalar_relative_error": abs(
            reversed_observable.total_scalar_shift_hz / reference.total_scalar_shift_hz - 1.0
        ),
        "vector_sign_flip_relative_error": abs(
            reversed_observable.total_vector_shift_hz / reference.total_vector_shift_hz + 1.0
        ),
        "tensor_relative_error": abs(
            reversed_observable.total_tensor_shift_hz / reference.total_tensor_shift_hz - 1.0
        ),
    }

    checks = {
        "six_unique_trapping_components": len({beam.label for beam in beams}) == 6,
        "full_3d_doppler_matches_analytic_hz": doppler_error < 0.1,
        "frequency_times_wavelength_equals_c": frequency_wavelength_error < 1.0e-14,
        "component_energy_sum_identity": component_sum_error_j < 1.0e-45,
        "joule_to_hertz_identity": joule_hertz_error < 1.0e-9,
        "aligned_magic_residual_small": aligned_residual_fraction < 2.0e-5,
        "fixed_axis_vector_profiles_are_odd": all(
            item["vector_even_over_peak"] < 1.0e-10 for item in parity.values()
        ),
        "fixed_axis_scalar_and_tensor_profiles_are_even": all(
            item["scalar_odd_over_peak"] < 1.0e-10
            and item["tensor_odd_over_peak"] < 1.0e-10
            for item in parity.values()
        ),
        "shift_is_linear_in_path_power": power_linearity_error < 1.0e-12,
        "global_helicity_reversal_flips_only_vector_term": all(
            value < 1.0e-12 for value in reversal.values()
        ),
        "no_external_magnetic_field": (
            context["apparatus"].external_magnetic_field_t == (0.0, 0.0, 0.0)
            and not context["apparatus"].anti_helmholtz_coils_present
        ),
        "origin_vector_pair_sum_is_zero": abs(origin.total_vector_shift_hz) < 1.0e-6,
    }
    return {
        "status": "PASS_WITH_QUALIFICATIONS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "numerical_residuals": {
            "maximum_doppler_formula_error_hz": doppler_error,
            "maximum_nu_lambda_over_c_minus_one": frequency_wavelength_error,
            "component_sum_error_j": component_sum_error_j,
            "joule_to_hertz_error_hz": joule_hertz_error,
            "aligned_magic_scalar_tensor_over_abs_vector": aligned_residual_fraction,
            "power_linearity_absolute_ratio_error": power_linearity_error,
            "parity": parity,
            "helicity_reversal": reversal,
        },
        "qualification": (
            "Passing means the temporary fixed-transition algebra and units are "
            "internally self-consistent. It does not validate a 24-state Stark "
            "Hamiltonian or demonstrate pMOT trapping."
        ),
    }


def _write_readme(path: Path, summary: dict) -> Path:
    origin = summary["origin"]
    power_mw = 1.0e3 * summary["configuration"]["demonstration_power_w_per_path"]
    fixed_detuning = summary["origin"]["fixed_carrier_effective_detuning_mhz"]
    text = f"""# Temporary fixed-stretched-transition pMOT diagnostic

Status: **{summary['qa']['status']}**, with the qualifications below.

This isolated test evaluates only the differential AC-Stark shift of
`{STRETCHED_TRANSITION_LABEL}`. It is not imported by the production pMOT
workflow. The six 1529-nm traveling components use matched propagation-frame
helicities `++-` on x/y/z for incident and retro paths. That is the previously
identified restoring orientation in the *ideal vector-only, red-detuned*
diagnostic.

## What was calculated

For every trapping component, the code evaluates the full three-dimensional
Doppler projection, its own atom-frame wavelength, its local Gaussian
intensity, and the narrow-table scalar/vector/tensor differential
polarizabilities. It then forms energy in joules, sums the six components, and
converts with `delta_nu_AC = DeltaE_AC/h`. The shifted cooling resonance is
`nu_res = nu_bare + delta_nu_AC`.

Ordinary-frequency detuning used for the named reference transition:

`{ORDINARY_FREQUENCY_EFFECTIVE_DETUNING_EQUATION}`

The F'=3 stretched reference has `delta_nu_HFS = 0` by definition. The
equivalent angular-frequency equation is:

`{ANGULAR_FREQUENCY_EFFECTIVE_DETUNING_EQUATION}`

The 1529-nm Doppler shift is used only to choose the polarizability at the
atom-frame wavelength; it is not added again as a 780-nm detuning.

## Principal result

The demonstration uses {power_mw:.6f} mW incident on each Cartesian path,
chosen only as the historical 20 G/cm vector-gradient proxy. The physical
trapping power remains unspecified, and every shift scales linearly with path
power in this incoherent-envelope model.

At rest at the symmetric origin (fixed lab-z basis), the six-beam shift is:

- scalar: {origin['scalar_shift_mhz']:+.6f} MHz
- vector: {origin['vector_shift_mhz']:+.6f} MHz
- tensor: {origin['tensor_shift_mhz']:+.6f} MHz
- total: {origin['total_shift_mhz']:+.6f} MHz

Thus a cooling carrier held 15 MHz below the *bare* stretched line is
{fixed_detuning:+.6f} MHz from the shifted central line. A positive value is
blue detuning. The scalar/tensor cancellation is excellent for one aligned
circular component, but it is not a six-beam identity: for one fixed axis the
orthogonal beams have different tensor geometry. At the symmetric origin the
three tensor-pair contributions cancel one another while all scalar
contributions add.

## Interpretation and boundary

This run corrects the sign problem in the older local-adiabatic transition
proxy by holding the named mF basis fixed and using the signed vector
projection `Q dot n_fixed`. The x, y, and z lineouts are three separate basis
diagnostics. A single atom cannot be simultaneously stretched along all three
axes. With no external magnetic field, the state labels become degenerate and
ambiguous at the optical-spin zero; a physical 3D prediction requires the
complete local Stark Hamiltonian and its eigenvectors.

The supplied CSV contains differential transition coefficients only. It
cannot yield separate ground/excited level shifts for all 24 states,
conservative Stark forces, or transformed 780-nm dipole couplings. Its column
headers also do not state the raw polarizability unit; this test follows the
repository's SI assumption, whose dimensional chain is internally consistent
but whose source provenance remains unverified.

No pMOT trajectory, loading, capture, temperature, or quantitative restoring
force is claimed. Coherent standing waves, 1529-nm scattering/heating/loss,
window and mirror polarization transformations, and nonadiabatic passage
through the fictitious-field zero remain outside this test.

## Files

- `figures/`: eight rendered diagnostic figures.
- `data/`: the underlying beamwise ledgers and lineout tables.
- `run_manifest.json`: formulas, configuration, hashes, and assumptions.
- `qa_result.json`: numerical identity, parity, reversal, and unit-chain checks.
- `summary.json`: concise machine-readable findings.
"""
    path.write_text(text, encoding="utf-8")
    return path


def run_diagnostic(
    *,
    output_directory: Path | None = None,
    power_w_per_path: float | None = None,
) -> dict:
    """Execute all calculations, QA checks, tables, and figures."""

    root = _project_root()
    output = (output_directory or root / "outputs" / "diagnostics" / "pmot" / CAMPAIGN_NAME).resolve()
    data_root = output / "data"
    figure_root = output / "figures"
    data_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    print("[temporary pMOT] 1/8 building isolated fixed-transition context", flush=True)
    context = _build_context(power_w_per_path)
    print(
        "[temporary pMOT] demonstration power="
        f"{1e3*context['power_w_per_path']:.6f} mW/path; "
        "trapping helicities incident/retro = ++-/++-",
        flush=True,
    )

    print("[temporary pMOT] 2/8 evaluating six-component 3D Doppler scan", flush=True)
    doppler_frame, doppler_csv, doppler_figure = _doppler_scan(context, data_root, figure_root)

    print("[temporary pMOT] 3/8 writing beamwise energy and shifted-resonance ledger", flush=True)
    reference, reference_frame, reference_csv, reference_figure = _reference_component_ledger(
        context, data_root, figure_root
    )

    print("[temporary pMOT] 4/8 evaluating Doppler sensitivity of magic cancellation", flush=True)
    magic_frame, magic_csv, magic_figure = _magic_coefficient_scan(
        context, data_root, figure_root
    )

    print("[temporary pMOT] 5/8 evaluating fixed-basis x/y/z position lineouts", flush=True)
    (
        position_frame,
        origin_shifts,
        position_csv,
        position_figure,
        detuning_figure,
    ) = _position_lineouts(context, data_root, figure_root)

    print("[temporary pMOT] 6/8 evaluating aligned-pair and six-beam cancellation", flush=True)
    pair_frame, pair_csv, pair_figure = _pair_magic_scan(context, data_root, figure_root)
    origin, origin_frame, origin_csv, origin_figure = _origin_budget(
        context, data_root, figure_root
    )
    cooling_frame, cooling_csv = _cooling_beam_ledger(
        context,
        reference,
        origin.total_frequency_shift_hz,
        data_root,
    )

    print("[temporary pMOT] 7/8 comparing x/n=x, y/n=y, and z/n=z pairs", flush=True)
    (
        axis_aligned_frame,
        axis_aligned_origin_rows,
        axis_aligned_csv,
        axis_aligned_figure,
    ) = _axis_aligned_pair_comparison(context, data_root, figure_root)

    print("[temporary pMOT] 8/8 running identity, parity, and reversal QA", flush=True)
    qa = _run_qa(context, reference, position_frame, magic_frame, origin)
    qa_path = _write_json(output / "qa_result.json", qa)

    bare_resonance_hz = context["apparatus"].mot_light.cooling.resonance_frequency_hz
    shifted_resonance_hz = origin.shifted_resonance_frequency_hz(bare_resonance_hz)
    fixed_origin_detuning_hz = cooling_effective_detunings_hz(
        context["cooling_repump_beams"],
        np.zeros(3),
        origin.total_frequency_shift_hz,
    )[0]
    zero_magic_index = int(np.argmin(np.abs(magic_frame["projected_speed_m_per_s"])))
    zero_magic = magic_frame.iloc[zero_magic_index]
    at_17_index = int(np.argmin(np.abs(magic_frame["projected_speed_m_per_s"] - 17.0)))
    at_minus_17_index = int(np.argmin(np.abs(magic_frame["projected_speed_m_per_s"] + 17.0)))
    summary = {
        "status": "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION_PHYSICS",
        "qa": qa,
        "transition": STRETCHED_TRANSITION_LABEL,
        "configuration": {
            "trapping_wavelength_nm": 1.0e9 * context["apparatus"].trapping_laser.wavelength_m,
            "trapping_lab_frequency_hz": (
                SPEED_OF_LIGHT_M_PER_S / context["apparatus"].trapping_laser.wavelength_m
            ),
            "demonstration_power_w_per_path": context["power_w_per_path"],
            "power_source": context["power_source"],
            "incident_helicities_xyz": list(context["stark"].incident_helicities_by_axis),
            "retro_helicities_xyz": list(context["stark"].retro_helicities_by_axis),
            "cooling_power_w_per_beam": context["apparatus"].mot_light.cooling.power_w_per_beam,
            "repump_power_w_per_beam": context["apparatus"].mot_light.repump.power_w_per_beam,
            "cooling_detuning_from_bare_hz": context["apparatus"].mot_light.cooling.detuning_hz,
            "external_magnetic_field_t": list(context["apparatus"].external_magnetic_field_t),
            "standing_wave_treatment": context["apparatus"].trapping_laser.envelope_combination,
            "position_lineout_extent_m": POSITION_EXTENT_M,
            "velocity_lineout_extent_m_per_s": VELOCITY_EXTENT_M_PER_S,
            "lineout_sample_count": LINEOUT_SAMPLE_COUNT,
        },
        "equations": {
            "trapping_atom_frame_frequency": "nu_seen_j = nu_lab_j * (1 - dot(khat_j,v)/c)",
            "trapping_atom_frame_wavelength": "lambda_seen_j = lambda_lab_j / (1 - dot(khat_j,v)/c)",
            "field_squared": "E_j^2 = 2 I_j/(c epsilon_0)",
            "scalar_energy": "U0_j = -alpha0_j E_j^2",
            "vector_energy": "U1_j = -alpha1_j E_j^2 s_j dot(khat_j,n_fixed)",
            "tensor_energy": "U2_j = -alpha2_j E_j^2 (3|epsilon_j dot n_fixed|^2-1)/2",
            "shifted_resonance": "nu_res = nu_bare + sum_j(U0_j+U1_j+U2_j)/h",
            "effective_detuning_hz": ORDINARY_FREQUENCY_EFFECTIVE_DETUNING_EQUATION,
            "effective_detuning_rad_per_s": ANGULAR_FREQUENCY_EFFECTIVE_DETUNING_EQUATION,
            "reference_hyperfine_offset_hz": 0.0,
        },
        "origin": {
            "fixed_quantization_axis": [0.0, 0.0, 1.0],
            "total_intensity_w_per_m2": origin.total_intensity_w_per_m2,
            "scalar_shift_mhz": origin.total_scalar_shift_hz / 1.0e6,
            "vector_shift_mhz": origin.total_vector_shift_hz / 1.0e6,
            "tensor_shift_mhz": origin.total_tensor_shift_hz / 1.0e6,
            "total_shift_mhz": origin.total_frequency_shift_hz / 1.0e6,
            "bare_transition_frequency_hz": bare_resonance_hz,
            "shifted_transition_frequency_hz": shifted_resonance_hz,
            "fixed_carrier_effective_detuning_mhz": fixed_origin_detuning_hz / 1.0e6,
            "center_compensated_effective_detuning_mhz": -15.0,
            "per_watt_path": {
                "total_intensity_m_inv2": (
                    origin.total_intensity_w_per_m2 / context["power_w_per_path"]
                ),
                "scalar_shift_mhz_per_watt_path": (
                    origin.total_scalar_shift_hz / 1.0e6 / context["power_w_per_path"]
                ),
                "vector_shift_mhz_per_watt_path": (
                    origin.total_vector_shift_hz / 1.0e6 / context["power_w_per_path"]
                ),
                "tensor_shift_mhz_per_watt_path": (
                    origin.total_tensor_shift_hz / 1.0e6 / context["power_w_per_path"]
                ),
                "total_shift_mhz_per_watt_path": (
                    origin.total_frequency_shift_hz / 1.0e6 / context["power_w_per_path"]
                ),
            },
        },
        "magic_point": {
            "scalar_mhz_per_1e5_w_per_m2": zero_magic["scalar_mhz_per_1e5_w_per_m2"],
            "vector_mhz_per_1e5_w_per_m2": zero_magic["vector_mhz_per_1e5_w_per_m2"],
            "tensor_mhz_per_1e5_w_per_m2": zero_magic["tensor_aligned_mhz_per_1e5_w_per_m2"],
            "scalar_plus_tensor_mhz_per_1e5_w_per_m2": zero_magic[
                "scalar_plus_tensor_mhz_per_1e5_w_per_m2"
            ],
            "residual_over_abs_vector": zero_magic["scalar_plus_tensor_over_abs_vector"],
            "residual_at_minus_17_m_per_s_mhz_per_1e5_w_per_m2": magic_frame.iloc[
                at_minus_17_index
            ]["scalar_plus_tensor_mhz_per_1e5_w_per_m2"],
            "residual_at_plus_17_m_per_s_mhz_per_1e5_w_per_m2": magic_frame.iloc[
                at_17_index
            ]["scalar_plus_tensor_mhz_per_1e5_w_per_m2"],
        },
        "interpretation": {
            "restoring_helicity_context": (
                "Matched propagation-frame ++- on x/y/z is restoring only in the "
                "ideal vector-only red-detuned diagnostic or after the central common "
                "shift is compensated. This run does not calculate a physical net force."
            ),
            "fixed_basis_scope": (
                "x/y/z lineouts define mF along a different fixed axis in each panel; "
                "they are separate diagnostics, not simultaneous eigenenergies."
            ),
            "scalar_tensor_result": (
                "Near-cancellation holds for an aligned circular component/pair but not "
                "for the all-six-beam non-collinear sum with one fixed axis."
            ),
            "axis_aligned_pair_result": {
                row["axis_pair"]: {
                    "quantization_axis": row["quantization_axis"],
                    "scalar_shift_mhz": row["scalar_shift_hz"] / 1.0e6,
                    "tensor_shift_mhz": row["tensor_shift_hz"] / 1.0e6,
                    "scalar_plus_tensor_residual_hz": row[
                        "scalar_plus_tensor_shift_hz"
                    ],
                    "vector_shift_at_origin_hz": row["vector_shift_hz"],
                }
                for row in axis_aligned_origin_rows.to_dict(orient="records")
            },
        },
        "limitations": [
            "CSV raw polarizability units and provenance are absent; SI is assumed.",
            "Only a differential stretched-transition triplet is available, not separate 5S and 5P level shifts.",
            "No 24-state Stark Hamiltonian, local diagonalization, or transformed dipole couplings.",
            "No conservative 1529-nm force, scattering, heating, loss, or coherent standing wave.",
            "No physical trapping power/path split has been supplied; demonstration power is only a historical proxy.",
            "No trajectory or quantitative trapping claim is made.",
        ],
    }
    summary_path = _write_json(output / "summary.json", summary)
    manifest = {
        "campaign": CAMPAIGN_NAME,
        "status": summary["status"],
        "project_root": root,
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_status_porcelain_at_run": _git_value(root, "status", "--porcelain"),
        "source_module": Path(__file__),
        "source_module_sha256": _sha256(Path(__file__)),
        "fixed_shift_module": Path(__file__).with_name("fixed_stretched_shift.py"),
        "fixed_shift_module_sha256": _sha256(Path(__file__).with_name("fixed_stretched_shift.py")),
        "polarizability_csv": context["table"].source_path,
        "polarizability_csv_sha256": _sha256(context["table"].source_path),
        "polarizability_row_count": len(context["table"].wavelengths_nm),
        "polarizability_wavelength_range_nm": list(context["table"].wavelength_range_nm),
        "configuration": summary["configuration"],
        "equations": summary["equations"],
        "reference_phase_space": {
            "position_m": list(REFERENCE_POSITION_M),
            "velocity_m_per_s": list(REFERENCE_VELOCITY_M_PER_S),
            "fixed_quantization_axis": list(REFERENCE_QUANTIZATION_AXIS),
        },
        "units": {
            "internal_length": "m",
            "internal_velocity": "m/s",
            "frequency": "Hz",
            "angular_frequency": "rad/s",
            "intensity": "W/m^2",
            "energy": "J",
            "raw_polarizability": "assumed SI; source CSV header does not declare units",
        },
        "output_files": [],
    }
    output_files = [
        doppler_csv,
        doppler_figure,
        reference_csv,
        reference_figure,
        magic_csv,
        magic_figure,
        position_csv,
        position_figure,
        detuning_figure,
        pair_csv,
        pair_figure,
        origin_csv,
        origin_figure,
        cooling_csv,
        axis_aligned_csv,
        axis_aligned_figure,
        qa_path,
        summary_path,
    ]
    manifest["output_files"] = [str(path.resolve()) for path in output_files]
    manifest_path = _write_json(output / "run_manifest.json", manifest)
    readme_path = _write_readme(output / "README.md", summary)

    result = {
        "status": summary["status"],
        "qa_status": qa["status"],
        "output_directory": output,
        "readme": readme_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "qa": qa_path,
        "data_files": [
            doppler_csv,
            reference_csv,
            magic_csv,
            position_csv,
            pair_csv,
            origin_csv,
            cooling_csv,
            axis_aligned_csv,
        ],
        "figure_files": [
            doppler_figure,
            reference_figure,
            magic_figure,
            position_figure,
            detuning_figure,
            pair_figure,
            origin_figure,
            axis_aligned_figure,
        ],
        "origin_total_shift_mhz": origin.total_frequency_shift_hz / 1.0e6,
        "origin_fixed_carrier_effective_detuning_mhz": fixed_origin_detuning_hz / 1.0e6,
    }
    print(
        "[temporary pMOT] complete: "
        f"QA={qa['status']}; origin shift={result['origin_total_shift_mhz']:+.6f} MHz; "
        f"fixed-carrier detuning={result['origin_fixed_carrier_effective_detuning_mhz']:+.6f} MHz",
        flush=True,
    )
    print(f"[temporary pMOT] outputs: {output}", flush=True)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated fixed-stretched-transition pMOT diagnostic"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="output root; defaults to outputs/diagnostics/pmot/<campaign>",
    )
    parser.add_argument(
        "--power-mw-per-path",
        type=float,
        default=None,
        help=(
            "optional demonstration incident power per Cartesian path; default "
            "uses the explicitly labeled historical 20 G/cm proxy"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_diagnostic(
        output_directory=args.output_directory,
        power_w_per_path=(
            None if args.power_mw_per_path is None else args.power_mw_per_path * 1.0e-3
        ),
    )
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False), flush=True)
    return 0 if result["qa_status"] == "PASS_WITH_QUALIFICATIONS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

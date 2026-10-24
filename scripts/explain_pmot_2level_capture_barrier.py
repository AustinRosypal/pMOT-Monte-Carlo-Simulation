"""Plot the spatial force/intensity structures behind pMOT capture barriers.

The calculation uses the current effective two-level 3D pMOT notebook
configuration.  It is a diagnostic of the provisional transition-level Stark
closure, not a validated physical pMOT prediction.  Gravity is excluded so
that the plotted curves isolate radiation pressure and the AC-Stark detuning.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


OUTPUT_ROOT = Path(
    "outputs/diagnostics/pMOT_2Level/capture_barrier_explainer_20261017"
)
PATH_POWERS_W = np.asarray((19.147243e-3, 19.147243e-3, 38.294486e-3))
COOLING_POWER_W = 27.0e-3
COOLING_DETUNING_HZ = -15.0e6
OFFSETS_MM = (0.0, 0.5, 1.0, 2.0, 12.0)
PLANE_NORMAL_OFFSETS_MM = (0.0, 1.0, 2.0)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
PLANE_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
COLORS = {
    0.0: "#2563eb",
    0.5: "#d97706",
    1.0: "#059669",
    2.0: "#be123c",
    12.0: "#52525b",
}


def _coordinates_m() -> np.ndarray:
    broad = np.linspace(-25.0, 25.0, 501)
    central = np.linspace(-3.0, 3.0, 241)
    negative_focus = np.linspace(-10.5, -9.5, 201)
    positive_focus = np.linspace(9.5, 10.5, 201)
    values_mm = np.unique(
        np.round(
            np.concatenate((broad, central, negative_focus, positive_focus)),
            decimals=9,
        )
    )
    return values_mm * 1.0e-3


def _build_model():
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=COOLING_DETUNING_HZ,
        include_gravity=False,
    )
    baseline = default_simple_mot_apparatus()
    apparatus = replace(
        baseline,
        cooling=replace(
            baseline.cooling,
            power_w_per_beam=COOLING_POWER_W,
            detuning_hz=COOLING_DETUNING_HZ,
            beam_diameter_m=12.7e-3,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    trapping = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    return simple, cooling, trapping, table


def _evaluate(simple, cooling, trapping, table, position, velocity):
    return evaluate_pmot_2level_local_observable(
        cooling,
        trapping,
        position,
        velocity,
        path_powers_w=PATH_POWERS_W,
        previous_axis=(0.0, 0.0, 1.0),
        simple_config=simple,
        polarizability_table=table,
    )


def _line_cut(
    simple,
    cooling,
    trapping,
    table,
    coordinates_m,
    *,
    scan_axis,
    offset_axis,
    offset_mm,
    inward_speed_m_per_s=0.0,
    fixed_axis=None,
    fixed_mm=0.0,
    velocity_mode="toward_origin",
):
    scan_index = AXIS_INDEX[scan_axis]
    offset_index = AXIS_INDEX[offset_axis]
    force = np.empty_like(coordinates_m)
    intensity = np.empty_like(coordinates_m)
    for sample_index, coordinate in enumerate(coordinates_m):
        position = np.zeros(3)
        position[scan_index] = coordinate
        position[offset_index] = offset_mm * 1.0e-3
        if fixed_axis is not None:
            position[AXIS_INDEX[fixed_axis]] = fixed_mm * 1.0e-3
        velocity = np.zeros(3)
        if velocity_mode == "constant_negative":
            velocity[scan_index] = -inward_speed_m_per_s
        elif velocity_mode == "toward_origin":
            # Incoming from either side means velocity points toward the origin.
            if coordinate > 0.0:
                velocity[scan_index] = -inward_speed_m_per_s
            elif coordinate < 0.0:
                velocity[scan_index] = inward_speed_m_per_s
        else:
            raise ValueError("velocity_mode must be 'toward_origin' or 'constant_negative'")
        observable = _evaluate(
            simple, cooling, trapping, table, position, velocity
        )
        force[sample_index] = observable.force_n[scan_index]
        intensity[sample_index] = np.sum(
            observable.stark.component_intensities_w_per_m2
        )
    return {
        "coordinate_m": coordinates_m,
        "force_n": force,
        "force_gradient_n_per_m": np.gradient(force, coordinates_m, edge_order=2),
        "intensity_w_per_m2": intensity,
    }


def _shade_geometry(axis):
    axis.axvspan(
        -2.0,
        2.0,
        color="#16a34a",
        alpha=0.07,
        label=r"reference $|u|\leq2$ mm band",
    )
    axis.axvspan(-15.0, -10.0, color="0.45", alpha=0.05)
    axis.axvspan(10.0, 15.0, color="0.45", alpha=0.05)
    for focus in (-10.0, 10.0):
        axis.axvline(focus, color="0.35", linestyle=":", linewidth=0.9)
    axis.axhline(0.0, color="0.25", linewidth=0.7)
    axis.grid(alpha=0.2)


def _plot_plane(plane, results, output_path, *, fixed_axis, fixed_mm):
    first_axis, second_axis = PLANE_AXES[plane]
    figure, axes = plt.subplots(3, 2, figsize=(15.0, 11.0), sharex="col")
    for column, (scan_axis, offset_axis) in enumerate(
        ((first_axis, second_axis), (second_axis, first_axis))
    ):
        for offset_mm in OFFSETS_MM:
            line = results[
                (plane, fixed_axis, fixed_mm, scan_axis, offset_axis, offset_mm)
            ]
            coordinate_mm = 1.0e3 * line["coordinate_m"]
            label = f"{offset_axis}={offset_mm:g} mm"
            color = COLORS[offset_mm]
            axes[0, column].plot(
                coordinate_mm,
                line["intensity_w_per_m2"],
                color=color,
                linewidth=1.25,
                label=label,
            )
            axes[1, column].plot(
                coordinate_mm,
                1.0e21 * line["force_n"],
                color=color,
                linewidth=1.25,
                label=label,
            )
            axes[2, column].plot(
                coordinate_mm,
                1.0e19 * line["force_gradient_n_per_m"],
                color=color,
                linewidth=1.25,
                label=label,
            )
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("symlog", linthresh=0.01)
        axes[2, column].set_yscale("symlog", linthresh=0.05)
        axes[0, column].set_title(
            f"scan {scan_axis} in {plane.upper()} plane; offset along {offset_axis}"
        )
        axes[2, column].set_xlabel(f"{scan_axis} [mm]")
        for row in range(3):
            _shade_geometry(axes[row, column])
        axes[0, column].legend(loc="best", ncol=2, fontsize=8, frameon=False)
    axes[0, 0].set_ylabel(r"total trap intensity [W/m$^2$]")
    axes[1, 0].set_ylabel(r"stationary $F_i$ [zN]")
    axes[2, 0].set_ylabel(r"$\partial F_i/\partial x_i$ [$10^{-19}$ N/m]")
    axes[0, 1].set_ylabel(r"total trap intensity [W/m$^2$]")
    axes[1, 1].set_ylabel(r"stationary $F_i$ [zN]")
    axes[2, 1].set_ylabel(r"$\partial F_i/\partial x_i$ [$10^{-19}$ N/m]")
    figure.suptitle(
        f"{plane.upper()} plane at {fixed_axis}={fixed_mm:g} mm: intensity, force, "
        "and local force gradient\n"
        "green is only the |scan coordinate|≤2-mm reference band; dotted lines are "
        "ideal waists; pale gray is nominal glass (not applied to beam physics)"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _positive_side_intervals(coordinate_m, force_n, *, outward=True):
    coordinate_m = np.asarray(coordinate_m)
    force_n = np.asarray(force_n)
    selected = (coordinate_m > 0.0) & ((force_n > 0.0) if outward else (force_n < 0.0))
    intervals = []
    start = None
    indices = np.flatnonzero(coordinate_m > 0.0)
    for index in indices:
        if selected[index] and start is None:
            start = index
        if start is not None and (not selected[index] or index == indices[-1]):
            end = index if selected[index] else index - 1
            intervals.append(
                [
                    float(1.0e3 * coordinate_m[start]),
                    float(1.0e3 * coordinate_m[end]),
                ]
            )
            start = None
    return intervals


def _plot_velocity_barrier(barrier_results, on_axis_intensity, output_path):
    figure, axes = plt.subplots(3, 1, figsize=(13.5, 11.0), sharex=True)
    velocity_colors = {
        0.0: "#18181b",
        0.25: "#2563eb",
        1.0: "#059669",
        5.0: "#d97706",
        10.0: "#be123c",
    }
    stationary = barrier_results[(0.5, 0.0)]
    coordinate_mm = 1.0e3 * stationary["coordinate_m"]
    axes[0].plot(
        coordinate_mm,
        stationary["intensity_w_per_m2"],
        color="#6d28d9",
        linewidth=1.5,
        label="total intensity, y=0.5 mm",
    )
    axes[0].plot(
        coordinate_mm,
        on_axis_intensity["intensity_w_per_m2"],
        color="#0891b2",
        linewidth=1.15,
        linestyle="--",
        label="total intensity, on axis y=0",
    )
    axes[0].set_yscale("log")
    for speed in velocity_colors:
        line = barrier_results[(0.5, speed)]
        axes[1].plot(
            coordinate_mm,
            1.0e21 * line["force_n"],
            color=velocity_colors[speed],
            linewidth=1.4,
            label=f"inward speed {speed:g} m/s",
        )
    for offset_mm, style in ((0.0, "--"), (0.5, "-"), (1.0, ":"), (2.0, "-.")):
        line = barrier_results[(offset_mm, 0.0)]
        axes[2].plot(
            coordinate_mm,
            1.0e21 * line["force_n"],
            linewidth=1.35,
            linestyle=style,
            label=f"stationary, y={offset_mm:g} mm",
        )
    axes[0].set_ylabel(r"trap intensity [W/m$^2$]")
    axes[1].set_ylabel(r"$F_x$ [zN]")
    axes[2].set_ylabel(r"$F_x(v=0)$ [zN]")
    axes[2].set_xlabel("x [mm]")
    axes[0].set_title(
        "A 0.5-mm offset misses the 2.23-µm waist; the on-axis curve contains the peaks"
    )
    axes[1].set_title(
        "Incoming force: on +x, positive force brakes or reflects an inward-moving atom"
    )
    axes[2].set_title("Impact parameter moves and reshapes the stationary barrier")
    for axis in axes:
        _shade_geometry(axis)
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    axes[1].legend(loc="best", ncol=2, fontsize=8, frameon=False)
    axes[2].legend(loc="best", ncol=2, fontsize=8, frameon=False)
    figure.suptitle(
        "Continuous +x-incidence diagnostic: v_x is fixed negative across the scan"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    simple, cooling, trapping, table = _build_model()
    coordinates_m = _coordinates_m()
    plane_results = {}
    for plane, (first_axis, second_axis) in PLANE_AXES.items():
        fixed_axis = ({"x", "y", "z"} - {first_axis, second_axis}).pop()
        for fixed_mm in PLANE_NORMAL_OFFSETS_MM:
            for scan_axis, offset_axis in (
                (first_axis, second_axis),
                (second_axis, first_axis),
            ):
                for offset_mm in OFFSETS_MM:
                    print(
                        f"plane={plane} {fixed_axis}={fixed_mm:g} mm "
                        f"scan={scan_axis} offset={offset_axis}={offset_mm:g} mm",
                        flush=True,
                    )
                    key = (
                        plane,
                        fixed_axis,
                        fixed_mm,
                        scan_axis,
                        offset_axis,
                        offset_mm,
                    )
                    plane_results[key] = _line_cut(
                        simple,
                        cooling,
                        trapping,
                        table,
                        coordinates_m,
                        scan_axis=scan_axis,
                        offset_axis=offset_axis,
                        offset_mm=offset_mm,
                        fixed_axis=fixed_axis,
                        fixed_mm=fixed_mm,
                    )
            suffix = "" if fixed_mm == 0.0 else f"_{fixed_axis}_{fixed_mm:g}mm"
            _plot_plane(
                plane,
                plane_results,
                OUTPUT_ROOT / f"{plane}_plane{suffix}_force_gradient_intensity.png",
                fixed_axis=fixed_axis,
                fixed_mm=fixed_mm,
            )

    barrier_results = {}
    for offset_mm in (0.0, 0.5, 1.0, 2.0):
        for speed in ((0.0, 0.25, 1.0, 5.0, 10.0) if offset_mm == 0.5 else (0.0,)):
            print(
                f"barrier scan x, y={offset_mm:g} mm, inward speed={speed:g} m/s",
                flush=True,
            )
            barrier_results[(offset_mm, speed)] = _line_cut(
                simple,
                cooling,
                trapping,
                table,
                coordinates_m,
                scan_axis="x",
                offset_axis="y",
                offset_mm=offset_mm,
                inward_speed_m_per_s=speed,
                velocity_mode="constant_negative",
            )
    on_axis_intensity = _line_cut(
        simple,
        cooling,
        trapping,
        table,
        coordinates_m,
        scan_axis="x",
        offset_axis="y",
        offset_mm=0.0,
    )
    _plot_velocity_barrier(
        barrier_results,
        on_axis_intensity,
        OUTPUT_ROOT / "x_entrance_velocity_barrier.png",
    )

    data = {}
    for key, result in plane_results.items():
        label = "_".join(
            (
                key[0],
                f"fixed_{key[1]}_{key[2]:g}mm",
                f"scan_{key[3]}",
                f"offset_{key[4]}_{key[5]:g}mm",
            )
        )
        for quantity, values in result.items():
            data[f"{label}_{quantity}"] = values
    for key, result in barrier_results.items():
        label = f"barrier_y_{key[0]:g}mm_speed_{key[1]:g}mps"
        for quantity, values in result.items():
            data[f"{label}_{quantity}"] = values
    np.savez_compressed(OUTPUT_ROOT / "linecuts.npz", **data)

    barrier_intervals = {
        f"y={offset:g}mm,inward_speed={speed:g}mps": _positive_side_intervals(
            result["coordinate_m"], result["force_n"], outward=True
        )
        for (offset, speed), result in barrier_results.items()
    }
    origin_slope_diagnostics = {}
    for y_mm in (0.0, 0.5, 1.0, 2.0):
        samples = []
        for x_mm in (-0.1, 0.0, 0.1):
            observable = _evaluate(
                simple,
                cooling,
                trapping,
                table,
                (x_mm * 1.0e-3, y_mm * 1.0e-3, 0.0),
                (0.0, 0.0, 0.0),
            )
            x_rates = [
                {
                    "beam": rate.beam_label,
                    "polarization_weights_q_minus_pi_q_plus": list(
                        rate.polarization_weights_q_minus_pi_q_plus
                    ),
                    "effective_detunings_mhz_q_minus_pi_q_plus": [
                        1.0e-6 * value
                        for value in rate.effective_detunings_hz_q_minus_pi_q_plus
                    ],
                    "total_scattering_rate_per_s": rate.total_scattering_rate_per_s,
                }
                for rate in observable.rate_samples
                if "horizontal_x" in rate.beam_label
            ]
            samples.append(
                {
                    "x_mm": x_mm,
                    "force_x_zN": 1.0e21 * observable.force_n[0],
                    "quantization_axis": list(observable.stark.quantization_axis),
                    "axis_from_vector_shift": (
                        observable.stark.quantization_axis_from_vector_shift
                    ),
                    "transition_shifts_mhz_q_minus_pi_q_plus": [
                        1.0e-6 * value
                        for value in observable.stark.transition_shifts_hz_by_q
                    ],
                    "x_cooling_rates": x_rates,
                }
            )
        origin_slope_diagnostics[f"y={y_mm:g}mm"] = samples

    manifest = {
        "status": "PROVISIONAL_EFFECTIVE_TWO_LEVEL_PMOT_DIAGNOSTIC",
        "cooling_power_mw_per_beam": 1.0e3 * COOLING_POWER_W,
        "cooling_detuning_mhz": 1.0e-6 * COOLING_DETUNING_HZ,
        "trapping_path_powers_mw_xyz": (1.0e3 * PATH_POWERS_W).tolist(),
        "trapping_wavelength_nm": 1529.268881,
        "gravity_included": False,
        "stationary_plane_curves": True,
        "positive_x_outward_force_intervals_mm": barrier_intervals,
        "near_origin_x_slope_diagnostics": origin_slope_diagnostics,
        "interpretation": (
            "For an atom entering from +x, F_x>0 opposes inward motion and is a braking/"
            "reflection region. Negative dF_x/dx is only a local restoring-slope test; "
            "capture also depends on force sign, velocity, and traversal of all resonance shells."
        ),
        "limitations": [
            "transition-level scalar/vector/tensor Stark ansatz",
            "history-dependent quantization-axis fallback at the vector zero",
            "no conservative 1529-nm dipole force",
            "no trapping-light scattering or heating",
            "no m_F-resolved optical pumping",
        ],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

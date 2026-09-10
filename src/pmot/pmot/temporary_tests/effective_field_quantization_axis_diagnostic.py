"""Beamwise fictitious-field and polarization-basis diagnostic for the pMOT.

The available Arora table contains *differential transition* polarizabilities,
not separate lower- and upper-level polarizabilities.  Consequently the field
reported here is the magnetic field that would give the same vector shift of
the named stretched cycling transition.  It is not a state-resolved physical
field for all 24 hyperfine-Zeeman states.
"""

from __future__ import annotations

import argparse
import json
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
from ...mot_multilevel.polarization import polarization_weights
from ...mot_multilevel.polarization import propagation_frame_polarization
from ...mot_multilevel.rate_equations import build_rate_equation_model
from ..ac_stark import ProvisionalStarkConfig
from ..ac_stark import build_physics_trapping_beams
from ..ac_stark import provisional_power_for_target_gradient_w_per_path
from ..ac_stark import provisional_transition_stark_shifts
from ..configuration import default_pmot_apparatus_config
from ..polarizability import interpolate_differential_polarizability_arrays
from ..polarizability import load_differential_polarizability_table
from ..trapping_beams import helicity_sign


CAMPAIGN_NAME = "temporary_effective_field_quantization_axis_20260909"
DEFAULT_POINT_M = (0.10e-3, 0.10e-3, 0.10e-3)
DEFAULT_VELOCITY_M_PER_S = (0.0, 0.0, 0.0)
STRETCHED_TRANSITION = "5S1/2 F=2,mF=+2 -> 5P3/2 F'=3,mF'=+3"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


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


def calculate_beamwise_diagnostic(
    *,
    point_m=DEFAULT_POINT_M,
    velocity_m_per_s=DEFAULT_VELOCITY_M_PER_S,
    power_w_per_path: float | None = None,
):
    """Return the beam ledger, summary, and inputs for one local calculation."""

    point = np.asarray(point_m, dtype=float)
    velocity = np.asarray(velocity_m_per_s, dtype=float)
    if point.shape != (3,) or velocity.shape != (3,):
        raise ValueError("point_m and velocity_m_per_s must be 3-vectors")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(velocity)):
        raise ValueError("point and velocity must be finite")

    apparatus = default_pmot_apparatus_config()
    rate_config = default_multilevel_mot_config()
    model = build_rate_equation_model(rate_config.natural_linewidth_rad_per_s)
    table = load_differential_polarizability_table()
    if power_w_per_path is None:
        power = provisional_power_for_target_gradient_w_per_path(
            model,
            apparatus.trapping_laser,
            target_gradient_g_per_cm=20.0,
            polarizability_table=table,
        )
        power_source = (
            "historical stretched-transition 20 G/cm proxy; demonstration only"
        )
    else:
        power = float(power_w_per_path)
        if not np.isfinite(power) or power <= 0.0:
            raise ValueError("power_w_per_path must be finite and positive")
        power_source = "explicit diagnostic input"

    stark_config = ProvisionalStarkConfig.uniform_power(
        power,
        incident_helicities_by_axis=("sigma+", "sigma+", "sigma-"),
        retro_helicities_by_axis=("sigma+", "sigma+", "sigma-"),
    )
    beams = build_physics_trapping_beams(
        apparatus.trapping_laser,
        stark_config,
    )
    observable = provisional_transition_stark_shifts(
        model,
        beams,
        point,
        velocity,
        apparatus.trapping_laser,
        stark_config,
        (0.0, 0.0, 1.0),
        polarizability_table=table,
    )
    _, alpha_vector, _ = interpolate_differential_polarizability_arrays(
        np.asarray(observable.atom_frame_wavelengths_nm),
        table,
    )
    intensities = np.asarray(observable.component_intensities_w_per_m2)
    field_squared = 2.0 * intensities / (
        SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M
    )

    reference_index = _reference_transition_index(model)
    reference_transition = model.structure.absorption_transitions[reference_index]
    reference_ground = model.structure.states[
        reference_transition.ground_state_index
    ]
    reference_excited = model.structure.states[
        reference_transition.excited_state_index
    ]
    differential_g_m = (
        reference_excited.lande_g * reference_excited.m_f
        - reference_ground.lande_g * reference_ground.m_f
    )
    zeeman_angular_coefficient = float(
        model.transition_zeeman_coefficient[reference_index]
    )
    # Match the existing provisional Stark observable's exact conversion.
    # (The repository's rounded HBAR constant is not used by that observable.)
    hbar_from_planck_j_s = PLANCK_CONSTANT_J_S / (2.0 * np.pi)
    differential_magnetic_moment_j_per_t = (
        zeeman_angular_coefficient * hbar_from_planck_j_s
    )

    optical_spin_intensity_factors = []
    vector_energies = []
    equivalent_fields = []
    for beam, intensity, alpha, electric_field_squared in zip(
        beams,
        intensities,
        alpha_vector,
        field_squared,
    ):
        optical_spin_intensity = (
            helicity_sign(beam.helicity)
            * intensity
            * np.asarray(beam.direction, dtype=float)
        )
        vector_energy = (
            -alpha
            * electric_field_squared
            * helicity_sign(beam.helicity)
            * np.asarray(beam.direction, dtype=float)
        )
        optical_spin_intensity_factors.append(optical_spin_intensity)
        vector_energies.append(vector_energy)
        equivalent_fields.append(
            vector_energy / differential_magnetic_moment_j_per_t
        )
    optical_spin_intensity_factors = np.asarray(
        optical_spin_intensity_factors
    )
    vector_energies = np.asarray(vector_energies)
    equivalent_fields = np.asarray(equivalent_fields)
    net_field = np.sum(equivalent_fields, axis=0)
    net_magnitude = float(np.linalg.norm(net_field))
    if net_magnitude <= 0.0:
        raise RuntimeError(
            "the chosen point is a fictitious-field zero; no local axis is defined"
        )
    quantization_axis = net_field / net_magnitude

    rows = []
    for index, (beam, optical_spin_intensity, energy_vector, field_vector) in enumerate(
        zip(
            beams,
            optical_spin_intensity_factors,
            vector_energies,
            equivalent_fields,
        )
    ):
        weights = polarization_weights(
            propagation_frame_polarization(beam.direction, beam.helicity),
            tuple(float(value) for value in quantization_axis),
        )
        direction = np.asarray(beam.direction, dtype=float)
        rows.append(
            {
                "beam_label": beam.label,
                "axis_name": beam.axis_name,
                "propagation_sense": beam.propagation_sense,
                "propagation_frame_helicity": beam.helicity,
                "k_hat_x": direction[0],
                "k_hat_y": direction[1],
                "k_hat_z": direction[2],
                "atom_frame_wavelength_nm": observable.atom_frame_wavelengths_nm[index],
                "intensity_w_per_m2": intensities[index],
                "intensity_w_per_m2_per_watt_path": intensities[index] / power,
                "optical_spin_intensity_x_w_per_m2": optical_spin_intensity[0],
                "optical_spin_intensity_y_w_per_m2": optical_spin_intensity[1],
                "optical_spin_intensity_z_w_per_m2": optical_spin_intensity[2],
                "differential_vector_polarizability_assumed_si": alpha_vector[index],
                "vector_energy_x_j": energy_vector[0],
                "vector_energy_y_j": energy_vector[1],
                "vector_energy_z_j": energy_vector[2],
                "equivalent_field_x_t": field_vector[0],
                "equivalent_field_y_t": field_vector[1],
                "equivalent_field_z_t": field_vector[2],
                "equivalent_field_x_g": 1.0e4 * field_vector[0],
                "equivalent_field_y_g": 1.0e4 * field_vector[1],
                "equivalent_field_z_g": 1.0e4 * field_vector[2],
                "equivalent_field_x_g_per_watt_path": 1.0e4 * field_vector[0] / power,
                "equivalent_field_y_g_per_watt_path": 1.0e4 * field_vector[1] / power,
                "equivalent_field_z_g_per_watt_path": 1.0e4 * field_vector[2] / power,
                "vector_shift_along_local_axis_hz": (
                    float(np.dot(energy_vector, quantization_axis))
                    / PLANCK_CONSTANT_J_S
                ),
                "sigma_plus_fraction": weights[+1],
                "pi_fraction": weights[0],
                "sigma_minus_fraction": weights[-1],
            }
        )
    frame = pd.DataFrame(rows)

    calculated_net = np.asarray(observable.effective_field_t)
    if not np.allclose(net_field, calculated_net, rtol=1.0e-12, atol=1.0e-18):
        raise RuntimeError("beamwise field sum disagrees with the Stark observable")
    if not np.allclose(
        frame[["sigma_plus_fraction", "pi_fraction", "sigma_minus_fraction"]].sum(axis=1),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    ):
        raise RuntimeError("polarization weights do not sum to one")

    summary = {
        "status": "PROVISIONAL_DIFFERENTIAL_TRANSITION_DIAGNOSTIC",
        "point_m": point.tolist(),
        "point_mm": (1.0e3 * point).tolist(),
        "velocity_m_per_s": velocity.tolist(),
        "trapping_wavelength_nm": 1.0e9 * apparatus.trapping_laser.wavelength_m,
        "power_w_per_path": power,
        "power_source": power_source,
        "path_helicities_xyz": ["sigma+", "sigma+", "sigma-"],
        "reference_transition": STRETCHED_TRANSITION,
        "ground_lande_g_f2": reference_ground.lande_g,
        "excited_lande_g_f3": reference_excited.lande_g,
        "differential_g_times_m": differential_g_m,
        "differential_zeeman_angular_coefficient_rad_per_s_per_t": (
            zeeman_angular_coefficient
        ),
        "differential_magnetic_moment_j_per_t": (
            differential_magnetic_moment_j_per_t
        ),
        "net_transition_equivalent_field_t": net_field.tolist(),
        "net_transition_equivalent_field_g": (1.0e4 * net_field).tolist(),
        "net_transition_equivalent_field_magnitude_g": 1.0e4 * net_magnitude,
        "net_transition_equivalent_field_g_per_watt_path": (
            1.0e4 * net_field / power
        ).tolist(),
        "net_optical_spin_intensity_factor_w_per_m2": np.sum(
            optical_spin_intensity_factors,
            axis=0,
        ).tolist(),
        "quantization_axis_from_vector_proxy": quantization_axis.tolist(),
        "formula_component_vector_energy": (
            "U_vec,j = -alpha_vector(lambda_seen,j) * "
            "[2 I_j/(c epsilon_0)] * s_j * k_hat_j"
        ),
        "formula_transition_equivalent_field": (
            "B_eq,j = U_vec,j / "
            "[mu_B (g_F'=3*m_F'=3 - g_F=2*m_F=2)]"
        ),
        "formula_axis": "n_hat = sum_j(B_eq,j) / |sum_j(B_eq,j)|",
        "important_limit": (
            "The CSV is differential for one transition. It cannot determine "
            "separate ground/excited effective fields or a full 24-state Stark Hamiltonian."
        ),
    }
    return frame, summary


def _short_labels(frame: pd.DataFrame) -> list[str]:
    axis = {
        "horizontal_x": "x",
        "horizontal_y": "y",
        "vertical_z": "z",
    }
    sense = {"incident": "inc", "retro": "retro"}
    return [
        f"{axis[row.axis_name]} {sense[row.propagation_sense]}\n{row.propagation_frame_helicity}"
        for row in frame.itertuples()
    ]


def _make_figure(frame: pd.DataFrame, summary: dict, path: Path) -> Path:
    labels = _short_labels(frame)
    positions = np.arange(len(frame))
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.2))

    colors = {"x": "#2563eb", "y": "#f97316", "z": "#16a34a"}
    for component, color in colors.items():
        axes[0].barh(
            positions,
            frame[f"equivalent_field_{component}_g"],
            color=color,
            label=rf"$B_{{{component}}}$",
        )
    axes[0].axvline(0.0, color="#64748b", linewidth=0.8)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Transition-equivalent field contribution [G]")
    axes[0].set_title("Each 1529-nm traveling component")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(axis="x", alpha=0.22)

    net = np.asarray(summary["net_transition_equivalent_field_g"])
    axes[1].bar(("x", "y", "z"), net, color=list(colors.values()))
    axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[1].set_ylabel("Net transition-equivalent field [G]")
    axes[1].set_title("Vector sum and local axis")
    axis = np.asarray(summary["quantization_axis_from_vector_proxy"])
    axes[1].text(
        0.04,
        0.96,
        (
            rf"$|\mathbf{{B}}_{{eq}}|={summary['net_transition_equivalent_field_magnitude_g']:.6f}$ G"
            "\n"
            rf"$\hat{{n}}=({axis[0]:+.6f},{axis[1]:+.6f},{axis[2]:+.6f})$"
        ),
        transform=axes[1].transAxes,
        va="top",
        fontsize=10,
    )
    axes[1].grid(axis="y", alpha=0.22)

    left = np.zeros(len(frame))
    for column, label, color in (
        ("sigma_plus_fraction", r"$\sigma^+$", "#dc2626"),
        ("pi_fraction", r"$\pi$", "#64748b"),
        ("sigma_minus_fraction", r"$\sigma^-$", "#2563eb"),
    ):
        values = np.asarray(frame[column])
        axes[2].barh(positions, values, left=left, color=color, label=label)
        left += values
    axes[2].set_yticks(positions, labels)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_xlabel("Polarization fraction in local spherical basis")
    axes[2].set_title(r"Decomposition relative to $\hat{n}$")
    axes[2].legend(frameon=False, ncol=3)
    axes[2].grid(axis="x", alpha=0.22)

    point = summary["point_mm"]
    figure.suptitle(
        "Provisional pMOT fictitious-field vector and polarization decomposition\n"
        f"r=({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) mm, atom at rest, "
        f"{1.0e3*summary['power_w_per_path']:.6f} mW/path demonstration scale",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _write_readme(path: Path, summary: dict) -> Path:
    field = summary["net_transition_equivalent_field_g"]
    axis = summary["quantization_axis_from_vector_proxy"]
    text = f"""# Temporary pMOT effective-field/axis diagnostic

This calculation is evaluated at `r = {summary['point_mm']} mm` for a stationary
atom using 1529.268881 nm trapping light and the matched propagation-frame path
helicities `(sigma+, sigma+, sigma-)`.  The displayed
`{1.0e3*summary['power_w_per_path']:.9f} mW/path` is the historical 20 G/cm
stretched-transition proxy scale, not an apparatus power recommendation.

For each component,

`U_vec,j = -alpha_vector(lambda_seen,j) [2 I_j/(c epsilon_0)] s_j k_hat_j`

and the available differential cycling-transition data are represented as

`B_eq,j = U_vec,j / [mu_B (3 g_F'=3 - 2 g_F=2)]`.

The result is `B_eq = ({field[0]:+.9f}, {field[1]:+.9f}, {field[2]:+.9f}) G`,
with magnitude `{summary['net_transition_equivalent_field_magnitude_g']:.9f} G`
and normalized axis
`n = ({axis[0]:+.9f}, {axis[1]:+.9f}, {axis[2]:+.9f})`.

The CSV gives every beam contribution and its sigma+/pi/sigma- fractions in
that local spherical basis.  Cooling and repump components with the same path
direction and propagation-frame helicity have the same geometric fractions.

## Scientific boundary

This is a transition-equivalent diagnostic.  The Arora CSV contains only a
differential polarizability triplet for the reference transition; it cannot
produce separate ground- and excited-manifold fictitious fields or the full
24-state Stark Hamiltonian.  A production pMOT must obtain level-resolved
polarizabilities, sum the complete Stark operators, diagonalize them locally,
and transform the 780-nm dipole couplings into the resulting eigenbasis.
"""
    path.write_text(text, encoding="utf-8")
    return path


def run_diagnostic(
    *,
    output_root: Path | None = None,
    power_w_per_path: float | None = None,
):
    destination = (
        Path(output_root)
        if output_root is not None
        else _project_root() / "outputs" / "diagnostics" / "pmot" / CAMPAIGN_NAME
    )
    destination.mkdir(parents=True, exist_ok=True)
    frame, summary = calculate_beamwise_diagnostic(
        power_w_per_path=power_w_per_path
    )
    csv_path = destination / "beamwise_field_and_polarization.csv"
    summary_path = destination / "summary.json"
    readme_path = destination / "README.md"
    figure_path = destination / "effective_field_and_polarization.png"
    frame.to_csv(csv_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(readme_path, summary)
    _make_figure(frame, summary, figure_path)
    result = {
        "output_directory": str(destination.resolve()),
        "data": str(csv_path.resolve()),
        "summary": str(summary_path.resolve()),
        "readme": str(readme_path.resolve()),
        "figure": str(figure_path.resolve()),
        **summary,
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return frame, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--power-mw-per-path",
        type=float,
        default=None,
        help="Optional diagnostic trapping power. Default uses the historical proxy scale.",
    )
    arguments = parser.parse_args()
    run_diagnostic(
        power_w_per_path=(
            None
            if arguments.power_mw_per_path is None
            else 1.0e-3 * arguments.power_mw_per_path
        )
    )


if __name__ == "__main__":
    main()

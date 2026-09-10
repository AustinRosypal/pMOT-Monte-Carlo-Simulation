"""Notebook-facing diagnostics for vector-only pMOT trajectories.

The trajectory engine stores the net transition-equivalent fictitious field
and the local quantization-axis proxy.  This module independently reconstructs
the six beamwise contributions to that field and projects all 18 cooling,
repump, and trapping components into the local spherical basis at every stored
sample.  These calculations are diagnostics for the explicitly provisional
vector-transition model; the returned field remains a transition-equivalent
proxy, not an externally applied magnetic field.

All numerical arrays and data-frame columns use SI units unless a column name
states otherwise.  Plotting is deliberately backend-neutral: importing this
module does not select ``Agg``, and the plotting helper neither shows nor
closes its figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..configuration import PLANCK_CONSTANT_J_S
from ..configuration import SPEED_OF_LIGHT_M_PER_S
from ..configuration import VACUUM_PERMITTIVITY_F_PER_M
from ..mot_multilevel.coupling import beam_polarization_vector
from ..mot_multilevel.polarization import polarization_weights
from ..mot_multilevel.polarization import propagation_frame_polarization
from .polarizability import interpolate_differential_polarizability_arrays
from .trapping_beams import helicity_sign
from .vector_only_trajectories import VectorOnlyPMOTTrajectoryContext
from .vector_only_trajectories import VectorOnlyPMOTTrajectoryRecord

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


SPHERICAL_COMPONENT_LABELS = (
    "sigma_plus_fraction",
    "pi_fraction",
    "sigma_minus_fraction",
)


@dataclass(frozen=True, slots=True)
class VectorOnlyBeamDescriptor:
    """Stable notebook metadata for one traveling optical component."""

    beam_index: int
    label: str
    family: str
    axis_name: str
    propagation_sense: str
    propagation_frame_polarization: str
    direction: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class VectorOnlyFieldReconstruction:
    """Beamwise reconstruction of the transition-equivalent field history.

    ``component_fields_t`` has shape ``(sample, trapping_beam, xyz)``.
    The two net-field arrays have shape ``(sample, xyz)``.  All are in tesla.
    """

    component_fields_t: np.ndarray
    reconstructed_net_fields_t: np.ndarray
    recorded_net_fields_t: np.ndarray
    component_beams: tuple[VectorOnlyBeamDescriptor, ...]
    reference_transition_index: int
    reference_magnetic_moment_j_per_t: float
    maximum_record_error_t: float
    maximum_component_sum_roundoff_t: float


@dataclass(frozen=True, slots=True)
class VectorOnlyPolarizationHistory:
    """Local spherical-polarization fractions for all 18 beams.

    ``weights`` has shape ``(sample, beam, spherical_component)``.  The final
    axis is ordered sigma+, pi, sigma- as documented by
    :data:`SPHERICAL_COMPONENT_LABELS`.
    """

    weights: np.ndarray
    beams: tuple[VectorOnlyBeamDescriptor, ...]
    spherical_component_labels: tuple[str, str, str]
    maximum_normalization_error: float


@dataclass(frozen=True, slots=True)
class VectorOnlyTrajectoryDiagnostics:
    """Reusable bundle of field and local-polarization trajectory QA."""

    field_reconstruction: VectorOnlyFieldReconstruction
    polarization_history: VectorOnlyPolarizationHistory


def _short_axis_label(value: str) -> str:
    return value.replace("horizontal_", "").replace("vertical_", "")


def _trapping_label(value: str) -> str:
    return (
        value.replace("horizontal_", "")
        .replace("vertical_", "")
        .replace("_trapping", "")
        .replace(" ", "_")
    )


def _beam_descriptors(
    context: VectorOnlyPMOTTrajectoryContext,
) -> tuple[VectorOnlyBeamDescriptor, ...]:
    descriptors: list[VectorOnlyBeamDescriptor] = []
    for index, beam in enumerate(context.cooling_repump_beams):
        descriptors.append(
            VectorOnlyBeamDescriptor(
                beam_index=index,
                label=(
                    f"{beam.family}_{_short_axis_label(beam.axis_name)}_"
                    f"{beam.propagation_sense}"
                ),
                family=beam.family,
                axis_name=beam.axis_name,
                propagation_sense=beam.propagation_sense,
                propagation_frame_polarization=beam.circular_polarization,
                direction=tuple(float(value) for value in beam.direction),
            )
        )
    offset = len(descriptors)
    for index, beam in enumerate(context.trapping_beams):
        descriptors.append(
            VectorOnlyBeamDescriptor(
                beam_index=offset + index,
                label=f"trapping_{_trapping_label(beam.label)}",
                family="trapping",
                axis_name=beam.axis_name,
                propagation_sense=beam.propagation_sense,
                propagation_frame_polarization=beam.helicity,
                direction=tuple(float(value) for value in beam.direction),
            )
        )
    return tuple(descriptors)


def _reference_transition_index(context: VectorOnlyPMOTTrajectoryContext) -> int:
    for index, transition in enumerate(
        context.model.structure.absorption_transitions
    ):
        if (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
        ) == (2, 2, 3, 3):
            return index
    raise RuntimeError("the rate model is missing the stretched cycling transition")


def _times_s(record: VectorOnlyPMOTTrajectoryRecord) -> np.ndarray:
    times = np.asarray(record.rate_equation.times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("trajectory record must contain at least one time sample")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
        raise ValueError("trajectory times must be finite and nondecreasing")
    return times


def reconstruct_vector_only_component_fields(
    record: VectorOnlyPMOTTrajectoryRecord,
    context: VectorOnlyPMOTTrajectoryContext,
    *,
    validate: bool = True,
    rtol: float = 2.0e-12,
    atol_t: float = 2.0e-14,
) -> VectorOnlyFieldReconstruction:
    """Independently reconstruct every trapping-beam field contribution.

    The calculation follows the production Stark observable's operation
    order: beamwise vector energies are summed and the result is divided by
    the stretched cycling-transition magnetic moment.  With ``validate=True``
    (the default), disagreement with the net field stored by the trajectory
    raises ``RuntimeError``.
    """

    if rtol < 0.0 or atol_t < 0.0:
        raise ValueError("field comparison tolerances must be non-negative")
    times = _times_s(record)
    beam_count = len(context.trapping_beams)
    wavelengths_nm = np.asarray(record.atom_frame_wavelengths_nm, dtype=float)
    intensities = np.asarray(
        record.trapping_component_intensities_w_per_m2,
        dtype=float,
    )
    recorded_net = np.asarray(record.effective_fields_t, dtype=float)
    expected_beam_shape = (len(times), beam_count)
    if wavelengths_nm.shape != expected_beam_shape:
        raise ValueError(
            "atom-frame wavelength history must have shape "
            f"{expected_beam_shape}, got {wavelengths_nm.shape}"
        )
    if intensities.shape != expected_beam_shape:
        raise ValueError(
            "trapping intensity history must have shape "
            f"{expected_beam_shape}, got {intensities.shape}"
        )
    if recorded_net.shape != (len(times), 3):
        raise ValueError("recorded effective-field history must have shape (n, 3)")
    if not (
        np.all(np.isfinite(wavelengths_nm))
        and np.all(np.isfinite(intensities))
        and np.all(np.isfinite(recorded_net))
    ):
        raise ValueError("field-reconstruction inputs must all be finite")

    _, alpha_vector, _ = interpolate_differential_polarizability_arrays(
        wavelengths_nm,
        context.polarizability_table,
    )
    field_squared_v2_per_m2 = 2.0 * intensities / (
        SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M
    )
    directions = np.asarray(
        [beam.direction for beam in context.trapping_beams],
        dtype=float,
    )
    helicity_signs = np.asarray(
        [helicity_sign(beam.helicity) for beam in context.trapping_beams],
        dtype=float,
    )
    component_vector_energy_j = (
        -alpha_vector[..., np.newaxis]
        * field_squared_v2_per_m2[..., np.newaxis]
        * helicity_signs[np.newaxis, :, np.newaxis]
        * directions[np.newaxis, :, :]
    )
    reference_index = _reference_transition_index(context)
    reference_magnetic_moment_j_per_t = float(
        context.model.transition_zeeman_coefficient[reference_index]
        * PLANCK_CONSTANT_J_S
        / (2.0 * pi)
    )
    if not np.isfinite(reference_magnetic_moment_j_per_t) or abs(
        reference_magnetic_moment_j_per_t
    ) <= 0.0:
        raise RuntimeError("stretched cycling transition has zero magnetic moment")

    component_fields_t = (
        component_vector_energy_j / reference_magnetic_moment_j_per_t
    )
    component_sum = np.sum(component_fields_t, axis=1)
    reconstructed_net = (
        np.sum(component_vector_energy_j, axis=1)
        / reference_magnetic_moment_j_per_t
    )
    record_error = float(
        np.max(np.linalg.norm(reconstructed_net - recorded_net, axis=1))
    )
    sum_roundoff = float(
        np.max(np.linalg.norm(component_sum - reconstructed_net, axis=1))
    )
    if validate and not np.allclose(
        reconstructed_net,
        recorded_net,
        rtol=rtol,
        atol=atol_t,
    ):
        raise RuntimeError(
            "beamwise effective-field reconstruction disagrees with the "
            f"trajectory record (maximum vector error {record_error:.6e} T)"
        )

    return VectorOnlyFieldReconstruction(
        component_fields_t=component_fields_t,
        reconstructed_net_fields_t=reconstructed_net,
        recorded_net_fields_t=recorded_net,
        component_beams=_beam_descriptors(context)[
            len(context.cooling_repump_beams) :
        ],
        reference_transition_index=reference_index,
        reference_magnetic_moment_j_per_t=reference_magnetic_moment_j_per_t,
        maximum_record_error_t=record_error,
        maximum_component_sum_roundoff_t=sum_roundoff,
    )


def vector_only_component_field_dataframe(
    record: VectorOnlyPMOTTrajectoryRecord,
    context: VectorOnlyPMOTTrajectoryContext,
    reconstruction: VectorOnlyFieldReconstruction | None = None,
) -> pd.DataFrame:
    """Return a wide beamwise field table, with every field column in tesla."""

    result = reconstruction or reconstruct_vector_only_component_fields(
        record,
        context,
    )
    times = _times_s(record)
    if result.component_fields_t.shape != (
        len(times),
        len(result.component_beams),
        3,
    ):
        raise ValueError("field reconstruction is not aligned with this trajectory")
    data: dict[str, np.ndarray] = {"time_s": times}
    for beam_index, beam in enumerate(result.component_beams):
        components = result.component_fields_t[:, beam_index, :]
        for component_index, axis_name in enumerate("xyz"):
            data[f"{beam.label}_beq_{axis_name}_t"] = components[
                :, component_index
            ]
        data[f"{beam.label}_beq_magnitude_t"] = np.linalg.norm(
            components,
            axis=1,
        )
    for component_index, axis_name in enumerate("xyz"):
        data[f"reconstructed_net_beq_{axis_name}_t"] = (
            result.reconstructed_net_fields_t[:, component_index]
        )
        data[f"recorded_net_effective_field_proxy_{axis_name}_t"] = (
            result.recorded_net_fields_t[:, component_index]
        )
    data["reconstructed_net_beq_magnitude_t"] = np.linalg.norm(
        result.reconstructed_net_fields_t,
        axis=1,
    )
    data["recorded_net_effective_field_proxy_magnitude_t"] = np.linalg.norm(
        result.recorded_net_fields_t,
        axis=1,
    )
    return pd.DataFrame(data)


def calculate_vector_only_polarization_history(
    record: VectorOnlyPMOTTrajectoryRecord,
    context: VectorOnlyPMOTTrajectoryContext,
    *,
    validate: bool = True,
    atol: float = 2.0e-14,
) -> VectorOnlyPolarizationHistory:
    """Project all 18 traveling components into every stored local basis."""

    if atol < 0.0:
        raise ValueError("polarization normalization tolerance must be non-negative")
    times = _times_s(record)
    axes = np.asarray(record.quantization_axes, dtype=float)
    if axes.shape != (len(times), 3) or not np.all(np.isfinite(axes)):
        raise ValueError("quantization-axis history must be a finite (n, 3) array")
    descriptors = _beam_descriptors(context)
    lab_polarizations = [
        beam_polarization_vector(beam)
        for beam in context.cooling_repump_beams
    ]
    lab_polarizations.extend(
        propagation_frame_polarization(beam.direction, beam.helicity)
        for beam in context.trapping_beams
    )
    weights = np.empty((len(times), len(descriptors), 3), dtype=float)
    for sample_index, axis in enumerate(axes):
        axis_tuple = tuple(float(value) for value in axis)
        for beam_index, polarization in enumerate(lab_polarizations):
            projected = polarization_weights(polarization, axis_tuple)
            weights[sample_index, beam_index] = (
                projected[+1],
                projected[0],
                projected[-1],
            )
    normalization_error = float(
        np.max(np.abs(np.sum(weights, axis=2) - 1.0))
    )
    if validate and normalization_error > atol:
        raise RuntimeError(
            "local spherical-polarization fractions do not sum to one "
            f"(maximum error {normalization_error:.6e})"
        )
    return VectorOnlyPolarizationHistory(
        weights=weights,
        beams=descriptors,
        spherical_component_labels=SPHERICAL_COMPONENT_LABELS,
        maximum_normalization_error=normalization_error,
    )


def vector_only_polarization_dataframe(
    record: VectorOnlyPMOTTrajectoryRecord,
    context: VectorOnlyPMOTTrajectoryContext,
    history: VectorOnlyPolarizationHistory | None = None,
) -> pd.DataFrame:
    """Return all 18 local spherical-polarization histories as a wide table."""

    result = history or calculate_vector_only_polarization_history(record, context)
    times = _times_s(record)
    if result.weights.shape != (len(times), len(result.beams), 3):
        raise ValueError("polarization history is not aligned with this trajectory")
    data: dict[str, np.ndarray] = {"time_s": times}
    for beam_index, beam in enumerate(result.beams):
        for component_index, component_name in enumerate(
            result.spherical_component_labels
        ):
            data[f"{beam.label}_{component_name}"] = result.weights[
                :, beam_index, component_index
            ]
    return pd.DataFrame(data)


def _time_average(times_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(times_s) == 1 or times_s[-1] <= times_s[0]:
        return np.mean(values, axis=0)
    intervals_s = np.diff(times_s)
    trapezoids = 0.5 * (values[:-1] + values[1:])
    return np.sum(
        trapezoids * intervals_s.reshape((-1,) + (1,) * (values.ndim - 1)),
        axis=0,
    ) / float(times_s[-1] - times_s[0])


def time_averaged_cooling_polarization_dataframe(
    record: VectorOnlyPMOTTrajectoryRecord,
    context: VectorOnlyPMOTTrajectoryContext,
    history: VectorOnlyPolarizationHistory | None = None,
) -> pd.DataFrame:
    """Return trapezoidal time averages for the six cooling-beam fractions."""

    result = history or calculate_vector_only_polarization_history(record, context)
    times = _times_s(record)
    if result.weights.shape != (len(times), len(result.beams), 3):
        raise ValueError("polarization history is not aligned with this trajectory")
    cooling_indices = [
        index for index, beam in enumerate(result.beams) if beam.family == "cooling"
    ]
    if not cooling_indices:
        raise ValueError("trajectory context contains no cooling beams")
    averages = _time_average(times, result.weights[:, cooling_indices, :])
    duration_s = float(times[-1] - times[0])
    rows = []
    for row_index, beam_index in enumerate(cooling_indices):
        beam = result.beams[beam_index]
        row: dict[str, object] = {
            "beam_index": beam.beam_index,
            "label": beam.label,
            "family": beam.family,
            "axis_name": beam.axis_name,
            "propagation_sense": beam.propagation_sense,
            "propagation_frame_polarization": (
                beam.propagation_frame_polarization
            ),
            "direction_x": beam.direction[0],
            "direction_y": beam.direction[1],
            "direction_z": beam.direction[2],
            "averaging_duration_s": duration_s,
        }
        for component_index, component_name in enumerate(
            result.spherical_component_labels
        ):
            row[component_name] = float(averages[row_index, component_index])
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_vector_only_trajectory(
    record: VectorOnlyPMOTTrajectoryRecord,
    context: VectorOnlyPMOTTrajectoryContext,
    *,
    validate: bool = True,
) -> VectorOnlyTrajectoryDiagnostics:
    """Calculate both production-side diagnostics for one trajectory record."""

    return VectorOnlyTrajectoryDiagnostics(
        field_reconstruction=reconstruct_vector_only_component_fields(
            record,
            context,
            validate=validate,
        ),
        polarization_history=calculate_vector_only_polarization_history(
            record,
            context,
            validate=validate,
        ),
    )


def plot_vector_only_field_axis_polarization(
    record: VectorOnlyPMOTTrajectoryRecord,
    context: VectorOnlyPMOTTrajectoryContext,
    path: str | Path | None = None,
    *,
    diagnostics: VectorOnlyTrajectoryDiagnostics | None = None,
    title: str | None = None,
    dpi: int = 220,
) -> tuple[Figure, np.ndarray]:
    """Plot net-field, quantization-axis, and cooling-polarization histories.

    Supplying ``path`` saves the figure and creates its parent directory.  The
    figure is always returned and is never shown or closed, which makes the
    function suitable for both interactive notebooks and noninteractive runs.
    """

    import matplotlib.pyplot as plt

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    result = diagnostics or analyze_vector_only_trajectory(record, context)
    times = _times_s(record)
    fields_g = 1.0e4 * result.field_reconstruction.recorded_net_fields_t
    axes_history = np.asarray(record.quantization_axes, dtype=float)
    if fields_g.shape != (len(times), 3) or axes_history.shape != (len(times), 3):
        raise ValueError("diagnostics are not aligned with this trajectory")
    average_frame = time_averaged_cooling_polarization_dataframe(
        record,
        context,
        result.polarization_history,
    )

    times_ms = 1.0e3 * times
    figure, panels = plt.subplots(2, 2, figsize=(14.5, 9.0))
    colors = ("#2563eb", "#f97316", "#16a34a")
    for component_index, (axis_name, color) in enumerate(zip("xyz", colors)):
        panels[0, 0].plot(
            times_ms,
            fields_g[:, component_index],
            color=color,
            label=rf"$B_{{{axis_name}}}$",
        )
        panels[0, 1].plot(
            times_ms,
            axes_history[:, component_index],
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
    panels[1, 0].semilogy(
        times_ms,
        np.maximum(np.linalg.norm(fields_g, axis=1), 1.0e-15),
        color="#111827",
    )
    panels[1, 0].set(
        title="Transition-equivalent field magnitude",
        ylabel="|B proxy| [G]",
    )

    mean_weights = average_frame[list(SPHERICAL_COMPONENT_LABELS)].to_numpy()
    image = panels[1, 1].imshow(
        mean_weights,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        aspect="auto",
    )
    panels[1, 1].set_xticks(
        (0, 1, 2),
        (r"$\sigma^+$", r"$\pi$", r"$\sigma^-$"),
    )
    panels[1, 1].set_yticks(
        np.arange(len(average_frame)),
        average_frame["label"].tolist(),
    )
    panels[1, 1].set_title("Time-averaged cooling polarization fractions")
    for row_index in range(mean_weights.shape[0]):
        for column_index in range(mean_weights.shape[1]):
            value = mean_weights[row_index, column_index]
            panels[1, 1].text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.45 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=panels[1, 1], label="Fraction")
    for panel in panels.flat:
        if panel is not panels[1, 1]:
            panel.set_xlabel("Time [ms]")
            panel.grid(alpha=0.22)
            if panel is not panels[1, 0]:
                panel.legend(frameon=False, fontsize=8)
    figure.suptitle(
        title or "Vector-only pMOT local-field trajectory diagnostic"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return figure, panels


__all__ = [
    "SPHERICAL_COMPONENT_LABELS",
    "VectorOnlyBeamDescriptor",
    "VectorOnlyFieldReconstruction",
    "VectorOnlyPolarizationHistory",
    "VectorOnlyTrajectoryDiagnostics",
    "analyze_vector_only_trajectory",
    "calculate_vector_only_polarization_history",
    "plot_vector_only_field_axis_polarization",
    "reconstruct_vector_only_component_fields",
    "time_averaged_cooling_polarization_dataframe",
    "vector_only_component_field_dataframe",
    "vector_only_polarization_dataframe",
]

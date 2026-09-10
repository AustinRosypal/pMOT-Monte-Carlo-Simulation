"""Focused checks for notebook-facing vector-only trajectory diagnostics."""

from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pmot.mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from pmot.pmot.vector_only_trajectories import (
    build_vector_only_trajectory_context,
)
from pmot.pmot.vector_only_trajectories import inward_launch_state
from pmot.pmot.vector_only_trajectories import (
    simulate_vector_only_pmot_trajectory,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    SPHERICAL_COMPONENT_LABELS,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    analyze_vector_only_trajectory,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    calculate_vector_only_polarization_history,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    plot_vector_only_field_axis_polarization,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    reconstruct_vector_only_component_fields,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    time_averaged_cooling_polarization_dataframe,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    vector_only_component_field_dataframe,
)
from pmot.pmot.vector_only_trajectory_diagnostics import (
    vector_only_polarization_dataframe,
)


@pytest.fixture(scope="module")
def short_trajectory():
    context = build_vector_only_trajectory_context(
        trapping_power_w_per_path=0.010,
    )
    record = simulate_vector_only_pmot_trajectory(
        inward_launch_state(),
        duration_s=5.0e-6,
        context=context,
        trajectory_config=RateEquationTrajectoryConfig(
            time_step_s=2.5e-6,
            include_diffusion=False,
            escape_radius_m=30.0e-3,
        ),
    )
    return record, context


def test_beamwise_field_reconstruction_matches_record(short_trajectory) -> None:
    record, context = short_trajectory
    reconstruction = reconstruct_vector_only_component_fields(record, context)

    assert reconstruction.component_fields_t.shape == (3, 6, 3)
    assert reconstruction.reconstructed_net_fields_t.shape == (3, 3)
    assert reconstruction.maximum_record_error_t < 2.0e-14
    assert reconstruction.maximum_component_sum_roundoff_t < 2.0e-14
    assert [beam.label for beam in reconstruction.component_beams] == [
        "trapping_x_incident",
        "trapping_x_retro",
        "trapping_y_incident",
        "trapping_y_retro",
        "trapping_z_incident",
        "trapping_z_retro",
    ]

    frame = vector_only_component_field_dataframe(
        record,
        context,
        reconstruction,
    )
    assert len(frame) == 3
    assert {
        "time_s",
        "trapping_x_incident_beq_x_t",
        "trapping_z_retro_beq_z_t",
        "reconstructed_net_beq_magnitude_t",
        "recorded_net_effective_field_proxy_magnitude_t",
    } <= set(frame.columns)
    np.testing.assert_allclose(
        frame[
            [f"reconstructed_net_beq_{axis}_t" for axis in "xyz"]
        ].to_numpy(),
        np.asarray(record.effective_fields_t),
        rtol=2.0e-12,
        atol=2.0e-14,
    )


def test_field_validation_detects_a_tampered_record(short_trajectory) -> None:
    record, context = short_trajectory
    modified = deepcopy(record)
    modified.effective_fields_t[0] = tuple(
        np.asarray(modified.effective_fields_t[0]) + (1.0e-6, 0.0, 0.0)
    )

    with pytest.raises(RuntimeError, match="disagrees"):
        reconstruct_vector_only_component_fields(modified, context)


def test_all_18_beam_polarizations_are_exposed_and_normalized(
    short_trajectory,
) -> None:
    record, context = short_trajectory
    history = calculate_vector_only_polarization_history(record, context)

    assert history.weights.shape == (3, 18, 3)
    assert history.spherical_component_labels == SPHERICAL_COMPONENT_LABELS
    assert history.maximum_normalization_error < 2.0e-14
    assert len({beam.label for beam in history.beams}) == 18
    assert [beam.family for beam in history.beams].count("cooling") == 6
    assert [beam.family for beam in history.beams].count("repump") == 6
    assert [beam.family for beam in history.beams].count("trapping") == 6
    np.testing.assert_allclose(np.sum(history.weights, axis=2), 1.0, atol=2.0e-14)

    frame = vector_only_polarization_dataframe(record, context, history)
    assert frame.shape == (3, 55)
    assert {
        "cooling_x_incident_sigma_plus_fraction",
        "repump_z_retro_pi_fraction",
        "trapping_y_incident_sigma_minus_fraction",
    } <= set(frame.columns)

    averages = time_averaged_cooling_polarization_dataframe(
        record,
        context,
        history,
    )
    assert len(averages) == 6
    np.testing.assert_allclose(
        averages["averaging_duration_s"].to_numpy(),
        5.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        averages[list(SPHERICAL_COMPONENT_LABELS)].sum(axis=1),
        1.0,
        atol=2.0e-14,
    )


def test_analysis_bundle_and_plot_are_notebook_friendly(
    short_trajectory,
    tmp_path,
) -> None:
    record, context = short_trajectory
    diagnostics = analyze_vector_only_trajectory(record, context)

    figure, panels = plot_vector_only_field_axis_polarization(
        record,
        context,
        diagnostics=diagnostics,
        title="short trajectory",
    )
    assert panels.shape == (2, 2)
    assert figure._suptitle.get_text() == "short trajectory"
    assert plt.fignum_exists(figure.number)
    plt.close(figure)

    output = tmp_path / "field_axis_polarization.png"
    saved_figure, _ = plot_vector_only_field_axis_polarization(
        record,
        context,
        output,
        diagnostics=diagnostics,
    )
    assert output.is_file()
    assert plt.fignum_exists(saved_figure.number)
    plt.close(saved_figure)

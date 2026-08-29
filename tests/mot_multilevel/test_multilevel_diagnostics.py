"""Trajectory-coupled and force-law diagnostics analogous to mot_simple tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import MultilevelAtomState, build_atomic_structure
from pmot.mot_multilevel import build_multilevel_cooling_beams
from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import simulate_multilevel_trajectory
from pmot.mot_multilevel import unpolarized_f2_mean_observable
from pmot.mot_multilevel.simulation import local_magnetic_field_t
from pmot.mot_multilevel.simulation import MultilevelTrajectoryRecord
from pmot.mot_multilevel.sampling import MultilevelSamplingConfig
from pmot.mot_multilevel.sampling import generate_launch_samples
from pmot.mot_multilevel.screening import core_entry_count
from pmot.mot_multilevel.screening import capture_classification
from pmot.mot_multilevel.screening import beam_rate_column
from pmot.mot_multilevel.screening import hyperfine_walk_frame_indices
from pmot.mot_multilevel.screening import lifetime_file_tag
from pmot.mot_multilevel.performance import generate_performance_bundle
from pmot.mot_multilevel.performance import bounded_trajectory_diagnostics
from pmot.mot_multilevel.performance import trajectory_performance_summary


@pytest.fixture(scope="module")
def apparatus():
    return build_atomic_structure(), build_multilevel_cooling_beams(), default_anti_helmholtz_config(), default_multilevel_mot_config()


def test_field_zero_and_component_gradients(apparatus) -> None:
    _, _, coil, _ = apparatus
    assert np.linalg.norm(local_magnetic_field_t((0, 0, 0), coil)) < 1e-15
    step = 0.1e-3
    slopes = []
    for component in range(3):
        plus = np.zeros(3); minus = np.zeros(3)
        plus[component] = step; minus[component] = -step
        slopes.append((local_magnetic_field_t(tuple(plus), coil)[component] - local_magnetic_field_t(tuple(minus), coil)[component]) / (2 * step))
    assert np.allclose(slopes[:2], (0.05, 0.05), rtol=2e-3)
    assert np.isclose(slopes[2], -0.1, rtol=2e-3)
    assert abs(sum(slopes)) < 2e-3


def test_unpolarized_force_is_zero_at_origin(apparatus) -> None:
    structure, beams, coil, config = apparatus
    observable = unpolarized_f2_mean_observable(structure, beams, (0, 0, 0), (0, 0, 0), coil, replace(config, include_gravity=False))
    assert np.linalg.norm(observable.force_n) < 1e-30
    assert observable.total_absorption_rate_per_s > 0


@pytest.mark.parametrize("component", range(3))
def test_unpolarized_force_damps_velocity(apparatus, component: int) -> None:
    structure, beams, coil, config = apparatus
    plus = np.zeros(3); minus = np.zeros(3); plus[component] = 2.0; minus[component] = -2.0
    force_plus = unpolarized_f2_mean_observable(structure, beams, (0, 0, 0), tuple(plus), coil, config).force_n[component]
    force_minus = unpolarized_f2_mean_observable(structure, beams, (0, 0, 0), tuple(minus), coil, config).force_n[component]
    assert force_plus < 0 < force_minus
    assert np.isclose(force_plus, -force_minus, rtol=1e-12)


@pytest.mark.parametrize("component", range(3))
def test_unpolarized_force_is_restoring(apparatus, component: int) -> None:
    structure, beams, coil, config = apparatus
    point = np.zeros(3); point[component] = 1e-3
    force = unpolarized_f2_mean_observable(structure, beams, tuple(point), (0, 0, 0), coil, config).force_n[component]
    assert force < -1e-25


def test_counterpropagating_pair_helicities_follow_propagation_frame_convention(apparatus) -> None:
    _, beams, _, _ = apparatus
    by_axis = {axis: [beam for beam in beams if beam.axis_name == axis] for axis in ("horizontal_x", "horizontal_y", "vertical_z")}
    assert all({beam.circular_polarization for beam in pair} == {"sigma+"} for pair in (by_axis["horizontal_x"], by_axis["horizontal_y"]))
    assert {beam.circular_polarization for beam in by_axis["vertical_z"]} == {"sigma-"}


def test_event_trajectory_records_recoil_and_enters_only_allowed_dark_parent(apparatus) -> None:
    structure, beams, coil, config = apparatus
    initial = MultilevelAtomState((0, 0, 0), (0, 0, 0), structure.state_index("ground", 2, 0))
    record = simulate_multilevel_trajectory(initial, 0.01, coil, beams=beams, structure=structure, config=config, seed=8675309)
    assert record.counters.absorption_events > 0
    assert record.counters.spontaneous_emissions > 0
    assert np.max(np.linalg.norm(np.asarray(record.velocities_m_per_s), axis=1)) > 0
    if record.termination_reason == "dark_state":
        assert record.counters.dark_parent_excited_f in (1, 2)
        assert structure.states[record.internal_state_indices[-1]].f == 1


def test_repumper_returns_f1_atom_to_f2_and_records_dwell_time(apparatus) -> None:
    structure, _, coil, config = apparatus
    repump_config = replace(config, repumper_enabled=True)
    beams = build_multilevel_mot_beams(config=repump_config)
    initial = MultilevelAtomState((0, 0, 0), (0, 0, 0), structure.state_index("ground", 1, 0))
    record = simulate_multilevel_trajectory(initial, 100e-6, coil, beams=beams, structure=structure, config=repump_config, seed=29, max_events=10_000)
    assert record.counters.repump_absorption_events > 0
    assert record.repump_absorptions
    assert record.counters.f1_visit_count > 0
    assert record.counters.total_f1_time_s > 0.0
    assert record.counters.f1_returns_to_f2 > 0
    assert record.repump_photons_per_f1_return
    assert any(structure.states[index].is_ground and structure.states[index].f == 2 for index in record.internal_state_indices)
    assert record.counters.total_f1_time_s < 100e-6


def test_gravity_is_applied_during_event_waits(apparatus) -> None:
    structure, _, coil, config = apparatus
    bright = MultilevelAtomState((0, 0, 0), (0, 0, 0), structure.state_index("ground", 2, 0))
    record = simulate_multilevel_trajectory(bright, 1e-3, coil, beams=[], structure=structure, config=config, seed=1)
    assert np.isclose(record.velocities_m_per_s[-1][2], -9.80665e-3)


def test_event_trajectory_records_per_beam_rates(apparatus) -> None:
    structure, beams, coil, config = apparatus
    initial = MultilevelAtomState((0, 0, 0), (0, 0, 0), structure.state_index("ground", 2, 0))
    record = simulate_multilevel_trajectory(initial, 5e-6, coil, beams=beams, structure=structure, config=config, seed=3)
    assert len(record.beam_scattering_rates_per_s) == len(record.times_s)
    assert all(len(rates) == len(beams) for rates in record.beam_scattering_rates_per_s)
    for total, rates in zip(record.total_scattering_rates_per_s, record.beam_scattering_rates_per_s):
        assert np.isclose(total, sum(rates))


def test_signed_side_beam_lookup_uses_beam_source_side(apparatus) -> None:
    _, beams, _, _ = apparatus
    plus_x = beam_rate_column(beams, "horizontal_x", "+")
    minus_x = beam_rate_column(beams, "horizontal_x", "-")
    assert beams[plus_x].direction[0] < 0.0
    assert beams[minus_x].direction[0] > 0.0


def test_lifetime_tag_uses_dark_entry_time_when_present() -> None:
    record = MultilevelTrajectoryRecord(times_s=[0.0, 2.0e-6])
    record.counters.dark_entry_time_s = 1.25e-6
    assert lifetime_file_tag(record) == "lifetime_1p250us"


def test_hyperfine_animation_preserves_actual_dark_entry_parent(apparatus) -> None:
    structure, beams, coil, config = apparatus
    initial = MultilevelAtomState((0.0, 8.0e-3, 0.0), (0.0, -6.0, 0.2), structure.state_index("ground", 2, 0))
    record = simulate_multilevel_trajectory(initial, 5e-3, coil, beams=beams, structure=structure, config=config, seed=31082028, max_events=5_000)
    dark_index = next(i for i, state_index in enumerate(record.internal_state_indices) if structure.states[state_index].is_dark)
    frames = hyperfine_walk_frame_indices(record, structure, max_transition_frames=200)
    dark_frame = frames.index(dark_index)
    previous = structure.states[record.internal_state_indices[frames[dark_frame - 1]]]
    dark = structure.states[record.internal_state_indices[frames[dark_frame]]]
    assert previous.is_excited
    assert dark.is_ground and dark.f == 1
    assert record.event_types[dark_index] == "spontaneous_emission"


def test_multilevel_sampling_generates_disc_points() -> None:
    rng = np.random.default_rng(11)
    search = MultilevelSamplingConfig(disc_count=2, points_per_disc=3, include_center_point=True)
    launches = generate_launch_samples(search, rng)
    assert len(launches) == 2
    assert all(len(points) == 3 for _, points in launches)
    assert all(points[0].s_m == 0.0 for _, points in launches)


def test_two_core_entry_counter_requires_an_intervening_exit() -> None:
    record = MultilevelTrajectoryRecord(
        positions_m=[(3e-3, 0, 0), (1e-3, 0, 0), (0, 0, 0), (3e-3, 0, 0), (1e-3, 0, 0)],
    )
    assert core_entry_count(record) == 2


def test_performance_bundle_saves_all_requested_histories_and_plots(apparatus, tmp_path) -> None:
    structure, _, coil, config = apparatus
    repump_config = replace(config, repumper_enabled=True)
    beams = build_multilevel_mot_beams(config=repump_config)
    initial = MultilevelAtomState((1e-3, 0, 0), (-0.2, 0, 0), structure.state_index("ground", 2, 0))
    record = simulate_multilevel_trajectory(
        initial, 2e-6, coil, beams=beams, structure=structure,
        config=repump_config, seed=17, max_events=1_000,
    )
    result = generate_performance_bundle(record, structure, beams, tmp_path, 0)
    assert len(result["outputs"]) == 8
    assert all(Path(path).is_file() for path in result["outputs"])
    with np.load(tmp_path / "trajectory_000.npz") as data:
        assert data["position_m"].shape[1] == 3
        assert data["velocity_m_per_s"].shape[1] == 3
        assert data["beam_available_absorption_rate_per_s"].shape[1] == len(beams)
        assert len(data["F"]) == len(record.times_s)
        assert set(data["beam_family"]) == {"cooling", "repump"}


def test_performance_summary_flags_event_cap() -> None:
    record = MultilevelTrajectoryRecord(
        times_s=[0.0, 1e-6],
        positions_m=[(2e-3, 0, 0), (1e-3, 0, 0)],
        velocities_m_per_s=[(-1, 0, 0), (-0.5, 0, 0)],
        termination_reason="max_events",
    )
    summary = trajectory_performance_summary(record)
    assert not summary["completed_requested_duration"]
    assert summary["final_speed_m_per_s"] < summary["initial_speed_m_per_s"]


def test_event_cap_takes_precedence_over_dark_visit() -> None:
    record = MultilevelTrajectoryRecord(
        times_s=[0.0],
        positions_m=[(0.0, 0.0, 0.0)],
        termination_reason="max_events",
    )
    record.counters.dark_entry_time_s = 1e-6
    assert capture_classification(record, repumper_enabled=True) == "indeterminate_event_cap"
    assert capture_classification(record, repumper_enabled=False) == "indeterminate_event_cap"


def test_repumped_f1_visit_is_not_classified_as_terminal_dark() -> None:
    record = MultilevelTrajectoryRecord(
        times_s=[0.0, 5e-3],
        positions_m=[(4e-3, 0.0, 0.0), (3e-3, 0.0, 0.0)],
        termination_reason="duration",
    )
    record.counters.dark_entry_time_s = 1e-6
    assert capture_classification(record, repumper_enabled=True) == "indeterminate_duration"
    assert capture_classification(record, repumper_enabled=False) == "untrapped_dark"


def test_multilevel_trajectory_stops_on_outward_escape(apparatus) -> None:
    structure, _, coil, config = apparatus
    initial = MultilevelAtomState((29e-3, 0.0, 0.0), (2.0, 0.0, 0.0), structure.state_index("ground", 2, 0))
    record = simulate_multilevel_trajectory(
        initial, 10e-3, coil, beams=[], structure=structure, config=config,
        seed=8, max_events=100, escape_radius_m=30e-3,
    )
    assert record.termination_reason == "escaped"
    assert np.linalg.norm(record.positions_m[-1]) >= 30e-3


def test_bounded_diagnostic_requires_completed_long_final_window() -> None:
    record = MultilevelTrajectoryRecord(
        times_s=[0.0, 4e-3, 5e-3],
        positions_m=[(8e-3, 0.0, 0.0), (1e-3, 0.0, 0.0), (0.5e-3, 0.0, 0.0)],
        termination_reason="duration",
    )
    diagnostic = bounded_trajectory_diagnostics(record)
    assert diagnostic["candidate_bounded_trapped"]
    record.termination_reason = "max_events"
    assert not bounded_trajectory_diagnostics(record)["candidate_bounded_trapped"]

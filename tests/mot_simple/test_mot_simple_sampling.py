from __future__ import annotations

from dataclasses import replace

from pmot.forces import AtomState
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_simple.sampling import CaptureSearchConfig
from pmot.mot_simple.sampling import DiscSample
from pmot.mot_simple.sampling import PointSample
from pmot.mot_simple.sampling import classify_trajectory
from pmot.mot_simple.sampling import plot_disc_plane_view


def test_two_core_entries_are_required_for_trapping(monkeypatch) -> None:
    positions = iter(
        [
            (1.0e-3, 0.0, 0.0),
            (3.0e-3, 0.0, 0.0),
            (1.0e-3, 0.0, 0.0),
        ]
    )

    def fake_rk4_step(beams, atom_state, time_step_s, coil_config, simple_config):
        return AtomState(next(positions), atom_state.velocity_m_per_s), (0.0, 0.0, 0.0), [], (0.0, 0.0, 0.0)

    monkeypatch.setattr("pmot.mot_simple.sampling.rk4_step", fake_rk4_step)
    point = PointSample(
        disc_index=0,
        point_index=0,
        theta_rad=0.0,
        phi_rad=0.0,
        theta_prime_rad=0.0,
        s_m=0.0,
        radial_distance_m=15.0e-3,
        initial_position_m=(15.0e-3, 0.0, 0.0),
        incident_unit_vector=(-1.0, 0.0, 0.0),
        launch_axis_unit_vector=(-1.0, 0.0, 0.0),
    )
    config = replace(
        CaptureSearchConfig(),
        time_step_s=1.0,
        max_simulation_time_s=4.0,
        required_core_entries=2,
    )
    result = classify_trajectory([], point, 1.0, object(), default_simple_mot_config(), config)
    assert result.trapped
    assert result.termination_reason == "two_core_entries"
    assert result.core_entry_count == 2


def test_disc_plane_view_uses_expected_output_name(tmp_path) -> None:
    disc = DiscSample(
        disc_index=7,
        theta_rad=0.0,
        phi_rad=0.0,
        outward_unit_vector=(0.0, 0.0, 1.0),
        incident_unit_vector=(0.0, 0.0, -1.0),
        center_position_m=(0.0, 0.0, 15.0e-3),
        basis_u=(1.0, 0.0, 0.0),
        basis_v=(0.0, 1.0, 0.0),
    )
    point = PointSample(
        disc_index=7,
        point_index=0,
        theta_rad=0.0,
        phi_rad=0.0,
        theta_prime_rad=0.0,
        s_m=0.0,
        radial_distance_m=15.0e-3,
        initial_position_m=disc.center_position_m,
        incident_unit_vector=disc.incident_unit_vector,
        launch_axis_unit_vector=disc.incident_unit_vector,
    )
    path = plot_disc_plane_view(disc, [point], [], tmp_path)
    assert path == tmp_path / "disc_0007_plane_view.png"
    assert path.is_file()

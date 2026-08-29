"""Focused checks for the multilevel-only August 22 loading sweeps."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.capture_statistics import CaptureVelocitySample, TrajectoryClassification
from pmot.launch_geometry import PointSample
from pmot.mot_multilevel import loading_sweeps as sweeps
from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.rate_capture import RateCaptureSearchConfig
from pmot.mot_multilevel.rate_equations import build_rate_equation_model


def test_requested_multilevel_parameter_grids() -> None:
    assert sweeps.SATURATION_N_VALUES == tuple(float(value) for value in range(1, 36))
    assert len(sweeps.BEAM_DIAMETER_MM_VALUES) == 25
    expected_step = (30.0 - 12.7) / 17.0
    expected_start = 12.7 - 7.0 * expected_step
    assert sweeps.BEAM_DIAMETER_MM_VALUES[0] == pytest.approx(expected_start)
    assert sweeps.BEAM_DIAMETER_MM_VALUES[7] == 12.7
    assert sweeps.BEAM_DIAMETER_MM_VALUES[-1] == pytest.approx(30.0)
    assert np.allclose(
        np.diff(sweeps.BEAM_DIAMETER_MM_VALUES),
        expected_step,
        rtol=0.0,
        atol=5.0e-15,
    )


def test_saturation_configuration_uses_multilevel_units_and_graph() -> None:
    n_value = 7.0
    config, apparatus, beams = sweeps.build_multilevel_loading_configuration(
        "saturation", n_value
    )
    base = default_multilevel_mot_config()
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]

    assert config.repumper_enabled
    assert config.natural_linewidth_rad_per_s == base.natural_linewidth_rad_per_s
    assert config.cooling_detuning_rad_per_s == base.cooling_detuning_rad_per_s
    assert config.saturation_intensity_w_per_m2 == 16.7
    assert model.ground_count == 8
    assert model.excited_count == 16
    assert model.state_count == 24
    assert len(cooling) == len(repump) == 6
    assert apparatus.cooling.power_w_per_beam == pytest.approx(
        sweeps.saturation_power_w_per_beam(n_value)
    )
    assert all(
        sweeps._beam_peak_intensity_w_per_m2(beam)
        / config.saturation_intensity_w_per_m2
        == pytest.approx(n_value)
        for beam in cooling
    )
    assert all(beam.power_w == pytest.approx(base.repump_power_w_per_beam) for beam in repump)


def test_beam_size_varies_only_cooling_and_resets_repump_to_baseline() -> None:
    base_config, _, base_beams = sweeps.build_multilevel_loading_configuration(
        "beam_size", 12.7
    )
    config, apparatus, beams = sweeps.build_multilevel_loading_configuration("beam_size", 30.0)
    base_cooling = next(beam for beam in base_beams if beam.family == "cooling")
    base_repump = next(beam for beam in base_beams if beam.family == "repump")
    cooling = next(beam for beam in beams if beam.family == "cooling")
    repump = next(beam for beam in beams if beam.family == "repump")

    assert apparatus.cooling.beam_diameter_m == pytest.approx(30.0e-3)
    assert cooling.power_w == pytest.approx(
        sweeps.beam_size_power_w_per_beam(30.0)
    )
    assert config.repump_power_w_per_beam == pytest.approx(
        base_config.repump_power_w_per_beam
    )
    assert sweeps._beam_peak_intensity_w_per_m2(cooling) == pytest.approx(
        sweeps._beam_peak_intensity_w_per_m2(base_cooling)
    )
    assert sweeps._beam_peak_intensity_w_per_m2(repump) == pytest.approx(
        sweeps._beam_peak_intensity_w_per_m2(base_repump)
    )
    assert repump.power_w == pytest.approx(base_repump.power_w)
    assert repump.beam_radius_m == pytest.approx(base_repump.beam_radius_m)


def test_common_geometry_is_reproducible_parallel_and_fixed_at_12_mm() -> None:
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=3,
        points_per_disc=5,
        worker_count=1,
        seed=71,
    )
    first = sweeps.generate_common_capture_points(search)
    second = sweeps.generate_common_capture_points(search)
    assert first == second
    assert sweeps._geometry_sha256(first) == sweeps._geometry_sha256(second)
    assert len(first) == 15
    assert RateCaptureSearchConfig().disc_count == 50
    assert RateCaptureSearchConfig().points_per_disc == 25
    for disc_index in range(search.disc_count):
        disc_points = [point for point in first if point.disc_index == disc_index]
        incident = np.asarray(disc_points[0].incident_unit_vector)
        center = -search.radial_distance_m * incident
        for point in disc_points:
            assert np.array_equal(point.incident_unit_vector, disc_points[0].incident_unit_vector)
            offset = np.asarray(point.initial_position_m) - center
            assert np.dot(offset, point.incident_unit_vector) == pytest.approx(0.0, abs=1.0e-17)
            assert 0.0 < point.s_m < 12.0e-3
            assert point.launch_axis_unit_vector == point.incident_unit_vector
        # Independent full-disc draws are not locked to point-index annuli.
        area_fractions = [
            (point.s_m / search.disc_radius_m) ** 2 for point in disc_points
        ]
        assert any(
            not (point.point_index / search.points_per_disc <= area_fraction
                 < (point.point_index + 1) / search.points_per_disc)
            for point, area_fraction in zip(disc_points, area_fractions, strict=True)
        )

    with pytest.raises(ValueError, match="12 mm"):
        sweeps.generate_common_capture_points(replace(search, disc_radius_m=11.9e-3))


def test_ten_by_twenty_five_design_and_preliminary_checkpoints_are_rejected(
    tmp_path,
) -> None:
    new_search = replace(
        RateCaptureSearchConfig(),
        disc_count=10,
        points_per_disc=25,
        include_center_point=False,
        worker_count=1,
    )
    points = sweeps.generate_common_capture_points(new_search)
    assert len(points) == 250
    assert {point.disc_index for point in points} == set(range(10))
    assert {point.point_index for point in points} == set(range(25))
    assert all(0.0 < point.s_m < new_search.disc_radius_m for point in points)

    values = (1.0,)
    new_signature, new_payload = sweeps._run_signature(
        "saturation",
        values,
        new_search,
    )
    old_search = replace(new_search, points_per_disc=1)
    old_signature, _ = sweeps._run_signature("saturation", values, old_search)
    assert new_signature != old_signature
    assert new_payload["plot_point_replicate_count"] == 10
    assert new_payload["points_per_replicate"] == 25
    assert new_payload["capture_threshold_simulation_count"] == 250

    output = tmp_path / "statistics"
    parameter_key = sweeps._parameter_key(0, "n", 1.0)
    old_spec = sweeps._ParameterWorkerSpec(
        sweep_kind="saturation",
        parameter_index=0,
        parameter_count=1,
        parameter_value=1.0,
        parameter_key=parameter_key,
        run_signature=old_signature,
        search=old_search,
        output_directory=output,
        resume=True,
    )
    old_point = sweeps.generate_common_capture_points(old_search)[0]
    parameter_directory = sweeps._parameter_directory(output, parameter_key)
    sweeps._save_checkpoint(
        parameter_directory,
        [_fake_capture_sample(old_point)],
        old_spec,
    )
    new_spec = replace(old_spec, run_signature=new_signature, search=new_search)
    assert sweeps._checkpointed_parameter_results(new_spec, points) == {}

    sweeps._atomic_write_json(
        parameter_directory / "aggregate_row.json",
        {"run_signature": old_signature, "parameter_key": parameter_key},
    )
    assert sweeps._completed_rows(output, new_signature, 250) == {}


def test_dynamic_spectrum_starts_at_zero_and_clears_high_capture_threshold() -> None:
    point = _inside_point()
    sample = CaptureVelocitySample(
        disc_index=point.disc_index,
        point_index=point.point_index,
        theta_rad=point.theta_rad,
        phi_rad=point.phi_rad,
        theta_prime_rad=point.theta_prime_rad,
        s_m=point.s_m,
        radial_distance_m=point.radial_distance_m,
        initial_position_m=point.initial_position_m,
        incident_unit_vector=point.incident_unit_vector,
        capture_velocity_m_per_s=34.6875,
        velocity_resolution_m_per_s=0.25,
        trapped_velocity_lower_m_per_s=34.6875,
        untrapped_velocity_upper_m_per_s=34.9375,
        lower_classification="two_core_entries",
        upper_classification="escaped",
        lower_entered_trap_core=True,
        upper_entered_trap_core=True,
        lower_core_entry_count=2,
        upper_core_entry_count=1,
    )
    search = replace(RateCaptureSearchConfig(), analysis_velocity_step_m_per_s=0.25)
    spectrum = sweeps.build_dynamic_capture_spectrum([sample], search)
    velocity = np.asarray([row.velocity_m_per_s for row in spectrum])
    assert velocity[0] == 0.0
    assert velocity[-1] >= sample.capture_velocity_m_per_s + 0.25
    assert velocity[-1] == pytest.approx(35.0)
    assert spectrum[-1].capture_cross_section_m2 == 0.0


def _inside_point() -> PointSample:
    return PointSample(
        disc_index=0,
        point_index=0,
        theta_rad=0.0,
        phi_rad=0.0,
        theta_prime_rad=0.0,
        s_m=0.0,
        radial_distance_m=1.0e-3,
        initial_position_m=(1.0e-3, 0.0, 0.0),
        incident_unit_vector=(0.0, 0.0, 0.0),
        launch_axis_unit_vector=(0.0, 0.0, 0.0),
    )


def test_continuous_five_ms_core_residence_is_capture(monkeypatch) -> None:
    config, _, beams = sweeps.build_multilevel_loading_configuration("saturation", 1.0)
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    calls: list[int] = []

    def fake_observable(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(force_n=(0.0, 0.0, 0.0), quantization_axis=(0.0, 0.0, 1.0))

    monkeypatch.setattr(sweeps, "rate_equation_observable", fake_observable)
    config = replace(config, include_gravity=False)
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=1,
        points_per_disc=1,
        max_simulation_time_s=6.0e-3,
        time_step_s=1.0e-3,
        worker_count=1,
    )
    result = sweeps.classify_multilevel_loading_trajectory(
        _inside_point(),
        0.0,
        search,
        model=model,
        beams=beams,
        coil_config=default_anti_helmholtz_config(),
        config=config,
    )
    assert result.trapped
    assert result.termination_reason == "bounded_core_residence"
    assert result.core_entry_count == 1
    assert result.elapsed_time_s == pytest.approx(5.0e-3)
    assert len(calls) == 5


def test_both_hybrid_trapped_reasons_are_valid_capture_endpoints() -> None:
    for reason in ("two_core_entries", "bounded_core_residence"):
        sample = replace(
            _fake_capture_sample(_inside_point()),
            lower_classification=reason,
            lower_core_entry_count=2 if reason == "two_core_entries" else 1,
        )
        sweeps._validate_capture_sample_endpoints(sample)


def test_capture_search_requires_a_trapped_and_untrapped_endpoint(monkeypatch) -> None:
    config, _, beams = sweeps.build_multilevel_loading_configuration("saturation", 1.0)
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)

    def fake_classification(point, speed, search, **kwargs):
        del point, search, kwargs
        trapped = speed <= 4.0
        return TrajectoryClassification(
            trapped=trapped,
            termination_reason="two_core_entries" if trapped else "escaped",
            entered_trap_core=trapped,
            core_entry_count=2 if trapped else 0,
            elapsed_time_s=1.0e-3,
            minimum_radius_m=1.0e-3 if trapped else 5.0e-3,
            final_radius_m=1.0e-3 if trapped else 30.0e-3,
            final_position_m=(0.0, 0.0, 0.0),
            final_velocity_m_per_s=(0.0, 0.0, 0.0),
        )

    monkeypatch.setattr(sweeps, "classify_multilevel_loading_trajectory", fake_classification)
    search = replace(
        RateCaptureSearchConfig(),
        initial_velocity_guess_m_per_s=20.0,
        velocity_tolerance_m_per_s=0.25,
        worker_count=1,
    )
    result = sweeps.find_multilevel_capture_velocity(
        _inside_point(),
        search,
        model=model,
        beams=beams,
        coil_config=default_anti_helmholtz_config(),
        config=config,
    )
    assert result.lower_classification == "two_core_entries"
    assert result.upper_classification == "escaped"
    assert result.trapped_velocity_lower_m_per_s <= 4.0
    assert result.untrapped_velocity_upper_m_per_s > 4.0
    assert (
        result.untrapped_velocity_upper_m_per_s
        - result.trapped_velocity_lower_m_per_s
        <= search.velocity_tolerance_m_per_s
    )


def test_capture_search_treats_nonfinite_trajectory_as_fatal(monkeypatch) -> None:
    config, _, beams = sweeps.build_multilevel_loading_configuration("saturation", 1.0)
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)

    def fake_classification(point, speed, search, **kwargs):
        del point, speed, search, kwargs
        return TrajectoryClassification(
            trapped=False,
            termination_reason="non_finite",
            entered_trap_core=False,
            core_entry_count=0,
            elapsed_time_s=1.0e-3,
            minimum_radius_m=5.0e-3,
            final_radius_m=float("nan"),
            final_position_m=(float("nan"), 0.0, 0.0),
            final_velocity_m_per_s=(float("nan"), 0.0, 0.0),
        )

    monkeypatch.setattr(sweeps, "classify_multilevel_loading_trajectory", fake_classification)
    with pytest.raises(RuntimeError, match="fatal non_finite"):
        sweeps.find_multilevel_capture_velocity(
            _inside_point(),
            RateCaptureSearchConfig(),
            model=model,
            beams=beams,
            coil_config=default_anti_helmholtz_config(),
            config=config,
        )


def _fake_capture_sample(point: PointSample) -> CaptureVelocitySample:
    threshold = 4.0 + 20.0 * point.s_m
    return CaptureVelocitySample(
        disc_index=point.disc_index,
        point_index=point.point_index,
        theta_rad=point.theta_rad,
        phi_rad=point.phi_rad,
        theta_prime_rad=point.theta_prime_rad,
        s_m=point.s_m,
        radial_distance_m=point.radial_distance_m,
        initial_position_m=point.initial_position_m,
        incident_unit_vector=point.incident_unit_vector,
        capture_velocity_m_per_s=threshold,
        velocity_resolution_m_per_s=0.25,
        trapped_velocity_lower_m_per_s=threshold,
        untrapped_velocity_upper_m_per_s=threshold + 0.25,
        lower_classification="two_core_entries",
        upper_classification="escaped",
        lower_entered_trap_core=True,
        upper_entered_trap_core=True,
        lower_core_entry_count=2,
        upper_core_entry_count=1,
    )


def test_capture_sample_rejects_other_invalid_upper_endpoint() -> None:
    sample = replace(
        _fake_capture_sample(_inside_point()),
        upper_classification="unknown_termination",
    )
    with pytest.raises(RuntimeError, match="invalid upper termination reason"):
        sweeps._validate_capture_sample_endpoints(sample)


def test_saturation_sweep_writes_raw_data_plots_and_resumes_partial(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_find(point, search, **kwargs):
        del search, kwargs
        calls.append((point.disc_index, point.point_index))
        sample = _fake_capture_sample(point)
        if point.disc_index % 2 == 1:
            sample = replace(sample, upper_classification="timeout")
        return sample

    monkeypatch.setattr(sweeps, "find_multilevel_capture_velocity", fake_find)
    statistics = tmp_path / "statistics"
    figures = tmp_path / "figures"
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=10,
        points_per_disc=25,
        save_every=1,
        worker_count=1,
    )
    rows = sweeps.run_multilevel_saturation_loading_sweep(
        output_directory=statistics,
        figure_directory=figures,
        search=search,
        worker_count=1,
        resume=True,
        n_values=(1.0, 2.0),
    )
    progress = capsys.readouterr().out
    assert "plot point 1/2" in progress
    assert "disc 1/10 points 5/25" in progress
    assert "all samples 250/250" in progress
    assert "internal capture sample" not in progress
    assert len(rows) == 2
    assert len(calls) == 500
    assert rows[0]["geometry_sha256"] == rows[1]["geometry_sha256"]
    assert all(row["model"] == sweeps._MODEL_NAME for row in rows)
    assert all(row["indexed_state_count"] == 24 for row in rows)
    assert all(row["upper_escaped_count"] == 125 for row in rows)
    assert all(row["upper_timeout_count"] == 125 for row in rows)
    assert all(row["upper_other_count"] == 0 for row in rows)
    assert all(row["simulation_replicate_count"] == 10 for row in rows)
    assert all(row["launch_disc_count"] == 10 for row in rows)
    assert all(row["points_per_disc"] == 25 for row in rows)
    assert all(row["capture_threshold_simulation_count"] == 250 for row in rows)
    assert all(
        row["loading_rate_atoms_per_s"]
        == pytest.approx(row["replicate_loading_rate_mean_atoms_per_s"])
        for row in rows
    )
    assert (statistics / "aggregate.csv").is_file()
    assert (statistics / "sweep_metadata.json").is_file()
    assert (figures / "loading_rate_vs_saturation.png").is_file()
    original_close = sweeps.plt.close
    captured_figures = []
    monkeypatch.setattr(sweeps.plt, "close", captured_figures.append)
    sweeps.plot_multilevel_loading_rate_vs_saturation(
        statistics / "aggregate.csv",
        figures / "loading_rate_vs_saturation_axis_check.png",
    )
    saturation_axis = captured_figures[-1].axes[0]
    assert saturation_axis.get_xlim() == pytest.approx((0.0, 35.0))
    assert 0.0 in saturation_axis.get_xticks()
    assert 35.0 in saturation_axis.get_xticks()
    original_close(captured_figures[-1])
    monkeypatch.setattr(sweeps.plt, "close", original_close)
    sweep_metadata = json.loads(
        (statistics / "sweep_metadata.json").read_text(encoding="utf-8")
    )
    assert (
        sweep_metadata["signature_payload"]["base_multilevel_config"]
        ["saturation_intensity_w_per_m2"]
        == 16.7
    )
    assert "no forced center or boundary" in sweep_metadata["signature_payload"][
        "geometry_sampler"
    ]
    assert sweep_metadata["signature_payload"]["format_version"] == 4
    assert sweep_metadata["signature_payload"]["plot_point_replicate_count"] == 10
    assert sweep_metadata["signature_payload"]["points_per_replicate"] == 25
    assert sweep_metadata["simulations_per_plot_point"] == 250
    assert "non_finite" in sweep_metadata["signature_payload"][
        "capture_endpoint_policy"
    ]
    assert "two core entries" in sweep_metadata["signature_payload"][
        "capture_endpoint_policy"
    ]
    assert "5 ms continuous residence" in sweep_metadata["signature_payload"][
        "capture_endpoint_policy"
    ]
    for row in rows:
        parameter = statistics / "parameters" / str(row["parameter_key"])
        assert (parameter / "capture_velocity_samples.csv").is_file()
        assert (parameter / "capture_velocity_partial_samples.csv").is_file()
        assert (parameter / "capture_velocity_spectrum.csv").is_file()
        assert (parameter / "replicate_loading_rates.csv").is_file()
        assert (parameter / "loading_rate.json").is_file()
        assert (parameter / "metadata.json").is_file()
        assert (parameter / "aggregate_row.json").is_file()
        metadata = json.loads((parameter / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["internal_frequency_units"] == "angular frequency in rad/s"
        assert metadata["capture_dynamics"].startswith("deterministic multilevel mean force")
        assert metadata["multilevel_config"]["saturation_intensity_w_per_m2"] == 16.7
        assert metadata["upper_termination_counts"] == {
            "escaped": 125,
            "other": 0,
            "other_reasons": [],
            "timeout": 125,
        }
        replicate_rows = sweeps._read_csv_rows(
            parameter / "replicate_loading_rates.csv"
        )
        assert len(replicate_rows) == 10
        assert all(int(item["point_count"]) == 25 for item in replicate_rows)
        replicate_rates = np.asarray(
            [float(item["loading_rate_atoms_per_s"]) for item in replicate_rows]
        )
        assert np.mean(replicate_rates) == pytest.approx(row["loading_rate_atoms_per_s"])
        replicate_std = np.std(replicate_rates, ddof=1)
        replicate_sem = replicate_std / np.sqrt(10.0)
        assert replicate_std == pytest.approx(
            row["replicate_loading_rate_sample_std_atoms_per_s"]
        )
        assert replicate_sem == pytest.approx(
            row["replicate_loading_rate_standard_error_atoms_per_s"]
        )
        assert 2.2621571627409915 * replicate_sem == pytest.approx(
            row["replicate_loading_rate_95_percent_half_width_atoms_per_s"]
        )
        loading_payload = json.loads(
            (parameter / "loading_rate.json").read_text(encoding="utf-8")
        )
        assert loading_payload["upper_termination_counts"] == metadata[
            "upper_termination_counts"
        ]
        spectrum_rows = sweeps._read_csv_rows(parameter / "capture_velocity_spectrum.csv")
        assert float(spectrum_rows[0]["velocity_m_per_s"]) == 0.0
        assert float(spectrum_rows[-1]["velocity_m_per_s"]) == pytest.approx(
            row["spectrum_velocity_max_m_per_s"]
        )

    # A complete-marker resume performs no trajectory work.
    resumed = sweeps.run_multilevel_saturation_loading_sweep(
        output_directory=statistics,
        figure_directory=figures,
        search=search,
        worker_count=1,
        resume=True,
        n_values=(1.0, 2.0),
    )
    assert len(resumed) == 2
    assert len(calls) == 500

    # Removing one completion marker exercises sample-level checkpoint resume.
    first_parameter = statistics / "parameters" / str(rows[0]["parameter_key"])
    (first_parameter / "aggregate_row.json").unlink()
    resumed_partial = sweeps.run_multilevel_saturation_loading_sweep(
        output_directory=statistics,
        figure_directory=figures,
        search=search,
        worker_count=1,
        resume=True,
        n_values=(1.0, 2.0),
    )
    assert len(resumed_partial) == 2
    assert len(calls) == 500
    assert (first_parameter / "aggregate_row.json").is_file()


def test_sample_shards_match_legacy_sequential_parameter_results(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sweeps,
        "find_multilevel_capture_velocity",
        lambda point, search, **kwargs: _fake_capture_sample(point),
    )
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=10,
        points_per_disc=25,
        include_center_point=False,
        save_every=10,
        worker_count=1,
    )
    values = (1.0,)
    signature, payload = sweeps._run_signature("saturation", values, search)
    legacy_output = tmp_path / "legacy_statistics"
    parameter_key = sweeps._parameter_key(0, "n", 1.0)
    legacy_spec = sweeps._ParameterWorkerSpec(
        sweep_kind="saturation",
        parameter_index=0,
        parameter_count=1,
        parameter_value=1.0,
        parameter_key=parameter_key,
        run_signature=signature,
        search=search,
        output_directory=legacy_output,
        resume=False,
    )
    legacy_row = sweeps._run_parameter_worker(legacy_spec)

    shard_output = tmp_path / "shard_statistics"
    shard_rows = sweeps.run_multilevel_saturation_loading_sweep(
        output_directory=shard_output,
        figure_directory=tmp_path / "shard_figures",
        search=search,
        worker_count=1,
        resume=False,
        n_values=values,
    )
    shard_row = shard_rows[0]
    legacy_samples = (
        legacy_output
        / "parameters"
        / parameter_key
        / "capture_velocity_samples.csv"
    )
    shard_samples = (
        shard_output
        / "parameters"
        / parameter_key
        / "capture_velocity_samples.csv"
    )
    assert legacy_samples.read_bytes() == shard_samples.read_bytes()
    assert payload["format_version"] == 4
    assert shard_row["run_signature"] == legacy_row["run_signature"] == signature
    assert shard_row["geometry_sha256"] == legacy_row["geometry_sha256"]
    assert {
        key: value for key, value in shard_row.items() if key != "elapsed_s"
    } == {
        key: value for key, value in legacy_row.items() if key != "elapsed_s"
    }


def test_sample_shards_resume_existing_partial_checkpoint_missing_only(
    tmp_path,
    monkeypatch,
) -> None:
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=10,
        points_per_disc=25,
        include_center_point=False,
        worker_count=1,
    )
    statistics = tmp_path / "statistics"
    figures = tmp_path / "figures"
    prepared = sweeps._prepare_loading_sweep(
        "saturation",
        (1.0,),
        output_directory=statistics,
        figure_directory=figures,
        search=search,
        worker_count=1,
        resume=True,
        execution_mode="test_signed_checkpoint",
    )
    spec = prepared.specs[0]
    points = sweeps.generate_common_capture_points(search)
    original_samples = [_fake_capture_sample(point) for point in points[:3]]
    parameter_directory = sweeps._parameter_directory(
        statistics,
        spec.parameter_key,
    )
    sweeps._save_checkpoint(parameter_directory, original_samples, spec)

    calls: list[tuple[int, int]] = []

    def fake_find(point, search, **kwargs):
        del search, kwargs
        calls.append((point.disc_index, point.point_index))
        return _fake_capture_sample(point)

    monkeypatch.setattr(sweeps, "find_multilevel_capture_velocity", fake_find)
    rows = sweeps.run_multilevel_saturation_loading_sweep(
        output_directory=statistics,
        figure_directory=figures,
        search=search,
        worker_count=1,
        resume=True,
        n_values=(1.0,),
    )
    assert calls == [
        (point.disc_index, point.point_index) for point in points[3:]
    ]
    assert rows[0]["run_signature"] == spec.run_signature
    checkpoint = json.loads(
        (parameter_directory / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint == {
        "completed_sample_count": 250,
        "expected_sample_count": 250,
        "parameter_key": spec.parameter_key,
        "run_signature": spec.run_signature,
    }
    samples = sweeps.load_capture_velocity_samples(
        parameter_directory / "capture_velocity_partial_samples.csv"
    )
    assert len(samples) == 250
    assert samples[:3] == original_samples


def test_sample_shard_failure_checkpoints_prior_shard_and_resumes_without_duplicates(
    tmp_path,
    monkeypatch,
) -> None:
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=10,
        points_per_disc=25,
        include_center_point=False,
        worker_count=1,
    )
    statistics = tmp_path / "statistics"
    figures = tmp_path / "figures"
    first_calls: list[tuple[int, int]] = []

    class LazyFuture:
        def __init__(self, function, shard):
            self.function = function
            self.shard = shard
            self.cancel_called = False

        def result(self):
            return self.function(self.shard)

        def cancel(self):
            self.cancel_called = True
            return True

    class LazyPool:
        instances = []

        def __init__(self, *, max_workers):
            self.max_workers = max_workers
            self.futures = []
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            return False

        def submit(self, function, shard):
            future = LazyFuture(function, shard)
            self.futures.append(future)
            return future

    monkeypatch.setattr(sweeps, "ProcessPoolExecutor", LazyPool)
    monkeypatch.setattr(sweeps, "as_completed", lambda futures: list(futures))

    def failing_find(point, search, **kwargs):
        del search, kwargs
        key = (point.disc_index, point.point_index)
        first_calls.append(key)
        if point.disc_index == 5:
            raise RuntimeError("synthetic shard failure")
        return _fake_capture_sample(point)

    monkeypatch.setattr(sweeps, "find_multilevel_capture_velocity", failing_find)
    with pytest.raises(RuntimeError, match="sample shard worker failed"):
        sweeps.run_multilevel_saturation_loading_sweep(
            output_directory=statistics,
            figure_directory=figures,
            search=search,
            worker_count=2,
            resume=True,
            n_values=(1.0,),
        )
    parameter_directory = next((statistics / "parameters").iterdir())
    checkpoint = json.loads(
        (parameter_directory / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["completed_sample_count"] == 125
    assert sweeps.SAMPLE_SHARD_SIZE == 5
    assert not (parameter_directory / "aggregate_row.json").exists()
    assert first_calls == [
        (disc_index, point_index)
        for disc_index in range(5)
        for point_index in range(25)
    ] + [(5, 0)]
    assert len(LazyPool.instances) == 1
    assert all(future.cancel_called for future in LazyPool.instances[0].futures)

    resumed_calls: list[tuple[int, int]] = []

    def resumed_find(point, search, **kwargs):
        del search, kwargs
        resumed_calls.append((point.disc_index, point.point_index))
        return _fake_capture_sample(point)

    monkeypatch.setattr(sweeps, "find_multilevel_capture_velocity", resumed_find)
    rows = sweeps.run_multilevel_saturation_loading_sweep(
        output_directory=statistics,
        figure_directory=figures,
        search=search,
        worker_count=2,
        resume=True,
        n_values=(1.0,),
    )
    assert len(rows) == 1
    assert resumed_calls == [
        (disc_index, point_index)
        for disc_index in range(5, 10)
        for point_index in range(25)
    ]
    assert (parameter_directory / "aggregate_row.json").is_file()


def test_beam_size_sweep_uses_separate_plot_and_12_mm_sampling_disc(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sweeps,
        "find_multilevel_capture_velocity",
        lambda point, search, **kwargs: _fake_capture_sample(point),
    )
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=10,
        points_per_disc=25,
        save_every=1,
        worker_count=1,
    )
    rows = sweeps.run_multilevel_beam_size_loading_sweep(
        output_directory=tmp_path / "statistics",
        figure_directory=tmp_path / "figures",
        search=search,
        worker_count=1,
        resume=False,
        diameter_mm_values=(12.7,),
    )
    assert len(rows) == 1
    assert rows[0]["beam_diameter_mm"] == pytest.approx(12.7)
    assert rows[0]["sampling_disc_radius_mm"] == pytest.approx(12.0)
    assert (tmp_path / "figures" / "loading_rate_vs_beam_size.png").is_file()
    metadata = json.loads(
        (tmp_path / "statistics" / "sweep_metadata.json").read_text(encoding="utf-8")
    )
    assert "aperture-limited" in metadata["scientific_limitations"][0]


def test_saturation_and_beam_size_are_separate_studies_with_independent_pools(
    tmp_path,
    monkeypatch,
) -> None:
    capture_calls: list[tuple[str, int, int]] = []

    def fake_find(point, search, **kwargs):
        del search
        capture_calls.append(
            (
                kwargs["config"].__class__.__name__,
                point.disc_index,
                point.point_index,
            )
        )
        return _fake_capture_sample(point)

    class ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class ImmediatePool:
        instances = []

        def __init__(self, *, max_workers):
            self.max_workers = max_workers
            self.submissions = []
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            return False

        def submit(self, function, shard):
            self.submissions.append(shard)
            return ImmediateFuture(function(shard))

    monkeypatch.setattr(sweeps, "find_multilevel_capture_velocity", fake_find)
    monkeypatch.setattr(sweeps, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(
        sweeps,
        "as_completed",
        lambda futures: list(reversed(list(futures))),
    )
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=10,
        points_per_disc=25,
        worker_count=1,
    )
    saturation_statistics = tmp_path / "saturation_statistics"
    beam_statistics = tmp_path / "beam_statistics"
    saturation_rows = sweeps.run_multilevel_saturation_loading_sweep(
        output_directory=saturation_statistics,
        figure_directory=tmp_path / "saturation_figures",
        search=search,
        worker_count=3,
        resume=True,
        n_values=(1.0, 2.0),
    )
    beam_rows = sweeps.run_multilevel_beam_size_loading_sweep(
        output_directory=beam_statistics,
        figure_directory=tmp_path / "beam_figures",
        search=search,
        worker_count=3,
        resume=True,
        diameter_mm_values=(12.7, 30.0),
    )

    assert len(saturation_rows) == len(beam_rows) == 2
    assert len(ImmediatePool.instances) == 2
    assert all(pool.max_workers == 3 for pool in ImmediatePool.instances)
    assert len(ImmediatePool.instances[0].submissions) == 100
    assert len(ImmediatePool.instances[1].submissions) == 100
    assert {
        shard.parameter.sweep_kind
        for shard in ImmediatePool.instances[0].submissions
    } == {"saturation"}
    assert {
        shard.parameter.sweep_kind
        for shard in ImmediatePool.instances[1].submissions
    } == {"beam_size"}
    assert all(
        len(shard.points) == 5
        for pool in ImmediatePool.instances
        for shard in pool.submissions
    )
    assert len(capture_calls) == 1000

    for statistics in (saturation_statistics, beam_statistics):
        metadata = json.loads(
            (statistics / "sweep_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["execution_mode"] == "single_sweep_sample_shard_pool"
        assert metadata["simulations_per_plot_point"] == 250
        assert metadata["sample_shard_size"] == 5
        assert metadata["checkpoint_writer"] == "parent process only"
        assert metadata["signature_payload"]["study_execution_mode"] == (
            "one separately presented parameter study"
        )
        assert metadata["signature_payload"]["plot_point_replicate_count"] == 10
        assert metadata["signature_payload"]["points_per_replicate"] == 25

    saturation_metadata = json.loads(
        (saturation_statistics / "sweep_metadata.json").read_text(encoding="utf-8")
    )
    beam_metadata = json.loads(
        (beam_statistics / "sweep_metadata.json").read_text(encoding="utf-8")
    )
    assert saturation_metadata["run_signature"] != beam_metadata["run_signature"]
    assert (
        beam_metadata["signature_payload"]["beam_size_configuration_policy_version"]
        == 2
    )
    assert "beam_size_repump_rule" not in saturation_metadata["signature_payload"]


def test_real_multilevel_multiprocessing_smoke(tmp_path) -> None:
    """Exercise two spawned workers on shards of one microscopic parameter."""

    search = replace(
        RateCaptureSearchConfig(),
        disc_count=10,
        points_per_disc=25,
        initial_velocity_guess_m_per_s=0.0,
        max_simulation_time_s=5.0e-6,
        time_step_s=5.0e-6,
        save_every=1,
        worker_count=1,
    )
    rows = sweeps.run_multilevel_saturation_loading_sweep(
        output_directory=tmp_path / "statistics",
        figure_directory=tmp_path / "figures",
        search=search,
        worker_count=2,
        resume=False,
        n_values=(1.0,),
    )
    assert len(rows) == 1
    assert all(row["indexed_state_count"] == 24 for row in rows)
    assert all(row["sample_count"] == 250 for row in rows)
    assert all(row["simulation_replicate_count"] == 10 for row in rows)
    assert all(row["spectrum_velocity_min_m_per_s"] == 0.0 for row in rows)
    assert all(row["zero_capture_no_bracket_count"] == 250 for row in rows)
    assert all(row["upper_escaped_count"] == 0 for row in rows)
    assert all(row["upper_timeout_count"] == 250 for row in rows)
    assert all(row["upper_other_count"] == 0 for row in rows)
    assert (tmp_path / "statistics" / "aggregate.csv").is_file()
    assert (tmp_path / "figures" / "loading_rate_vs_saturation.png").is_file()

def test_cli_exposes_parameter_workers_and_safe_plot_only_paths() -> None:
    args = sweeps.build_argument_parser().parse_args(
        [
            "beam-size",
            "--workers",
            "2",
            "--output-dir",
            "statistics",
            "--figures-dir",
            "figures",
            "--plot-only",
        ]
    )
    assert not hasattr(args, "disc_count")
    assert not hasattr(args, "points_per_disc")
    assert args.workers == 2
    assert args.output_dir.name == "statistics"
    assert args.figures_dir.name == "figures"
    assert args.plot_only

    with pytest.raises(SystemExit):
        sweeps.build_argument_parser().parse_args(["all"])

    saturation_output, saturation_figures = sweeps._default_output_directories(
        "saturation"
    )
    beam_output, beam_figures = sweeps._default_output_directories("beam_size")
    assert saturation_output.name.endswith("_10_discs_25_points")
    assert saturation_figures.name.endswith("_10_discs_25_points")
    assert beam_output.name.endswith("_10_discs_25_points")
    assert beam_figures.name.endswith("_10_discs_25_points")

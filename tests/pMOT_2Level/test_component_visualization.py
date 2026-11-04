"""Exercise selected Stark ranks through the live APIs and plotting scripts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from pmot.pMOT_2Level import pmot_stark_component_model_label


ROOT = Path(__file__).resolve().parents[2]


def _load(relative_path):
    spec = spec_from_file_location(Path(relative_path).stem, ROOT / relative_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def server():
    return _load("visualizations/pmot_interactive/serve.py")


def _query(**kwargs):
    return {key: [str(value)] for key, value in kwargs.items()}


@pytest.mark.parametrize("code", ("none", "s", "v", "t", "sv", "st", "vt", "svt"))
def test_selected_stark_maps_disable_exactly_the_requested_ranks(server, code):
    payload = server.api_stark_plane(_query(
        components=code, points=15, extent_mm=1, slice_mm=0.25,
        trap_x_mw=20, trap_y_mw=30, trap_z_mw=40,
    ))
    assert payload["model_label"] == pmot_stark_component_model_label(code)
    for letter, field in (("s", "scalar_mhz"), ("v", "vector_mhz"), ("t", "tensor_mhz")):
        assert bool(np.any(payload[field] != 0)) == (letter in code)
    assert np.all(np.isfinite(payload["axis_u"]))
    assert bool(np.any(payload["axis_defined"])) == ("v" in code)
    # Turning Stark ranks off must preserve the trapping-light geometry.
    assert np.all(payload["intensity_w_per_m2"] > 0)


def test_comparison_matches_standalone_force_and_gradient_for_each_selection(server):
    query = dict(points=31, scan_limit_mm=2, offset_1_mm=0.3, offset_2_mm=-0.2)
    for prefix, code in (("full", "svt"), ("vector", "sv")):
        query.update({
            f"{prefix}_components": code,
            f"{prefix}_pathway": "scalar_tensor_cancellation",
            f"{prefix}_cooling_mw": 6,
            f"{prefix}_detuning_mhz": -30,
            **{f"{prefix}_trap_{axis}_mw": power for axis, power in zip("xyz", (20, 30, 40))},
        })
    comparison = server.api_model_comparison(_query(**query))
    for key, code in (("pMOT_2Level", "svt"), ("pmot_vec_2Level", "sv")):
        model = comparison["models"][key]
        assert model["label"] == pmot_stark_component_model_label(code)
        for axis in "xyz":
            line = server.api_force_line(_query(
                components=code, pathway="scalar_tensor_cancellation", axis=axis,
                points=31, min_mm=-2, max_mm=2, offset_1_mm=0.3, offset_2_mm=-0.2,
                cooling_power_mw=6, cooling_detuning_mhz=-30,
                trap_x_mw=20, trap_y_mw=30, trap_z_mw=40,
            ))
            np.testing.assert_allclose(model["force_zN"][axis], line["force_zN"], atol=1e-12)
            np.testing.assert_allclose(
                model["gradient_1e19_n_per_m"][axis],
                10 * line["gradient_zN_per_mm"], atol=1e-12,
            )
    assert not np.allclose(
        comparison["models"]["pMOT_2Level"]["force_zN"]["x"],
        comparison["models"]["pmot_vec_2Level"]["force_zN"]["x"],
    )


def test_all_off_phase_space_retains_doppler_damping(server):
    payload = server.api_phase_space(_query(
        components="none", position_points=15, velocity_points=15,
        position_extent_mm=1, velocity_extent_m_per_s=1,
    ))
    assert payload["model_label"] == "2Level_pMOT_none"
    assert payload["force_zN"][7, 7] == pytest.approx(0, abs=1e-10)
    assert payload["force_zN"][8, 7] < 0
    assert payload["force_zN"][6, 7] > 0


def test_overlay_script_uses_canonical_models_and_selected_wavelength(monkeypatch):
    script = _load("scripts/plot_updated_force_gradient_overlays.py")
    with pytest.raises(ValueError, match="unique"):
        script.configure_pmot_component_models(("sv", "vs"))
    script.configure_pmot_component_models(("svt", "sv", "none"))
    monkeypatch.setattr(script, "PMOT_PATHWAY", "scalar_zero")
    frame = script.evaluate_models(np.array([-0.001, 0, 0.001]))
    assert set(frame.model) == {
        "mot_multilevel", "mot_strange", "2Level_pMOT_svt", "2Level_pMOT_sv", "2Level_pMOT_none",
    }
    assert script.pmot_environment()[2][0].wavelength_m == pytest.approx(1529.273195181438e-9)
    assert np.all(np.isfinite(frame.force_gradient_n_per_m))


def test_derived_overlay_keeps_surrogate_vector_field_when_pmot_vector_is_off():
    script = _load("scripts/plot_pmot_derived_strange_overlay.py")
    frame, metadata = script.evaluate_linecuts(
        np.array([-0.001, 0, 0.001]), cooling_power_mw=6, cooling_detuning_mhz=-30,
        repump_power_mw=0.1, trapping_powers_mw=np.array([20, 30, 40]),
        pmot_components=("svt", "sv", "none"), pathway_name="scalar_zero",
    )
    assert metadata["two_level_model_keys"] == ["2Level_pMOT_svt", "2Level_pMOT_sv", "2Level_pMOT_none"]
    assert metadata["pathway"] == "scalar_zero"
    assert set(frame.model) == {*metadata["two_level_model_keys"], "mot_multilevel", "pmot_derived_mot_strange"}
    assert np.any(np.asarray(metadata["origin_effective_field_gradient_g_per_cm_xyz"]) != 0)
    assert np.all(np.isfinite(frame.force_gradient_n_per_m))

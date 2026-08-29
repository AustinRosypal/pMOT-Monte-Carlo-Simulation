"""Boundary and data-selection checks for the future pMOT package."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from pmot.fields import build_mot_beams, sample_intensity_cloud_by_polarization
from pmot.pmot.configuration import pmot_paths
from pmot.pmot.polarizability import (
    choose_polarizability_csv_path,
    differential_shift_coefficients_for_wavelength,
)


def test_pmot_paths_are_model_specific(tmp_path: Path) -> None:
    paths = pmot_paths(tmp_path)
    assert paths["notebooks"] == tmp_path / "notebooks" / "pmot"
    assert paths["outputs_statistics"] == tmp_path / "outputs" / "statistics" / "pmot"
    assert paths["data_raw"] == tmp_path / "data" / "raw" / "pmot"


def test_780_nm_lookup_selects_full_range_table() -> None:
    selected_path = choose_polarizability_csv_path(780.0)
    assert "FullRange" in selected_path.name
    coefficients = differential_shift_coefficients_for_wavelength(780.0)
    assert isfinite(coefficients.scalar_mhz_per_intensity)
    assert isfinite(coefficients.vector_mhz_per_intensity)
    assert isfinite(coefficients.tensor_mhz_per_intensity)


def test_missing_pi_beams_produce_an_explicit_empty_cloud() -> None:
    clouds = sample_intensity_cloud_by_polarization(
        build_mot_beams(),
        axial_extent_m=1.0e-3,
        axial_samples=2,
        radial_rings=1,
        angular_samples=2,
    )
    assert clouds["sigma+"]
    assert clouds["sigma-"]
    assert clouds["pi"] == []

"""Differential-polarizability utilities for future pMOT trapping light."""

from __future__ import annotations

from bisect import bisect_left
from csv import DictReader
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..configuration import PLANCK_CONSTANT_J_S
from ..configuration import SPEED_OF_LIGHT_M_PER_S
from ..configuration import VACUUM_PERMITTIVITY_F_PER_M


@dataclass(frozen=True, slots=True)
class DifferentialPolarizabilitySample:
    """One tabulated differential-polarizability sample."""

    wavelength_nm: float
    scalar_si: float
    vector_si: float
    tensor_si: float


@dataclass(frozen=True, slots=True)
class DifferentialShiftCoefficients:
    """Converted coefficients in MHz/[mW/(100 um)^2]."""

    wavelength_nm: float
    scalar_mhz_per_intensity: float
    vector_mhz_per_intensity: float
    tensor_mhz_per_intensity: float


@dataclass(frozen=True, slots=True)
class DifferentialPolarizabilityTable:
    """Preloaded narrow-band data for trajectory-safe interpolation."""

    wavelengths_nm: np.ndarray
    scalar_si: np.ndarray
    vector_si: np.ndarray
    tensor_si: np.ndarray
    source_path: Path

    @property
    def wavelength_range_nm(self) -> tuple[float, float]:
        return float(self.wavelengths_nm[0]), float(self.wavelengths_nm[-1])


def default_polarizability_csv_path(root: Path | None = None) -> Path:
    """Return the raw Arora CCSD polarizability table path."""

    project_root = root or Path(__file__).resolve().parents[3]
    return project_root / "data" / "raw" / "pmot" / "Arora_CCSD_Differential_Polarizabilities.csv"


def full_range_polarizability_csv_path(root: Path | None = None) -> Path:
    """Return the full-range Arora CCSD polarizability table path."""

    project_root = root or Path(__file__).resolve().parents[3]
    return project_root / "data" / "raw" / "pmot" / "Arora_CCSD_FullRange_Differential_Polarizabilities.csv"


def load_differential_polarizability_csv(
    csv_path: Path | None = None,
) -> list[DifferentialPolarizabilitySample]:
    """Load the differential-polarizability table from CSV."""

    path = csv_path or default_polarizability_csv_path()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = DictReader(handle)
        samples = [
            DifferentialPolarizabilitySample(
                wavelength_nm=float(row["Wavelength (nm)"]),
                scalar_si=float(row["Differential Scalar Polarizability"]),
                vector_si=float(row["Differential Vector Polarizability"]),
                tensor_si=float(row["Differential Tensor Polarizability"]),
            )
            for row in reader
        ]
    if not samples:
        raise ValueError(f"no polarizability samples found in {path}")
    return samples


@lru_cache(maxsize=4)
def load_differential_polarizability_table(
    csv_path: Path | None = None,
) -> DifferentialPolarizabilityTable:
    """Load one CSV once into immutable NumPy arrays.

    Continuously Doppler-shifted wavelengths should use this table rather than
    the scalar, wavelength-cached helper below.  That avoids re-reading all
    70,400 narrow-band rows for every new trajectory velocity.
    """

    path = (csv_path or default_polarizability_csv_path()).resolve()
    dataframe = pd.read_csv(
        path,
        usecols=(
            "Wavelength (nm)",
            "Differential Scalar Polarizability",
            "Differential Vector Polarizability",
            "Differential Tensor Polarizability",
        ),
    )
    arrays = [
        dataframe[column].to_numpy(dtype=float)
        for column in dataframe.columns
    ]
    if not arrays or len(arrays[0]) < 2:
        raise ValueError(f"polarizability table must contain at least two rows: {path}")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"polarizability table contains non-finite values: {path}")
    if not np.all(np.diff(arrays[0]) > 0.0):
        raise ValueError(f"polarizability wavelengths must be strictly increasing: {path}")
    for values in arrays:
        values.setflags(write=False)
    return DifferentialPolarizabilityTable(
        wavelengths_nm=arrays[0],
        scalar_si=arrays[1],
        vector_si=arrays[2],
        tensor_si=arrays[3],
        source_path=path,
    )


def interpolate_differential_polarizability_arrays(
    wavelengths_nm,
    table: DifferentialPolarizabilityTable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized, range-checked interpolation of all three differential ranks.

    Extrapolation and silent clipping are forbidden because the narrow table
    contains sharp resonant structure near the selected wavelength.
    """

    data = table or load_differential_polarizability_table()
    requested = np.asarray(wavelengths_nm, dtype=float)
    if not np.all(np.isfinite(requested)):
        raise ValueError("wavelengths_nm must contain only finite values")
    lower, upper = data.wavelength_range_nm
    if np.any(requested < lower) or np.any(requested > upper):
        actual_lower = float(np.min(requested))
        actual_upper = float(np.max(requested))
        raise ValueError(
            f"requested wavelength range {actual_lower} to {actual_upper} nm is "
            f"outside the narrow table range {lower} to {upper} nm"
        )
    return (
        np.interp(requested, data.wavelengths_nm, data.scalar_si),
        np.interp(requested, data.wavelengths_nm, data.vector_si),
        np.interp(requested, data.wavelengths_nm, data.tensor_si),
    )


def wavelength_is_in_sample_range(
    wavelength_nm: float,
    samples: list[DifferentialPolarizabilitySample],
) -> bool:
    """Return whether a wavelength lies within a table's sampled range."""

    if not samples:
        return False
    return samples[0].wavelength_nm <= wavelength_nm <= samples[-1].wavelength_nm


def choose_polarizability_csv_path(
    wavelength_nm: float,
    preferred_csv_path: Path | None = None,
    fallback_csv_path: Path | None = None,
) -> Path:
    """Choose the narrow or full-range CSV based on wavelength coverage."""

    preferred_path = preferred_csv_path or default_polarizability_csv_path()
    fallback_path = fallback_csv_path or full_range_polarizability_csv_path()
    preferred_samples = load_differential_polarizability_csv(preferred_path)
    if wavelength_is_in_sample_range(wavelength_nm, preferred_samples):
        return preferred_path
    fallback_samples = load_differential_polarizability_csv(fallback_path)
    if wavelength_is_in_sample_range(wavelength_nm, fallback_samples):
        return fallback_path
    raise ValueError(
        f"wavelength {wavelength_nm} nm is outside both table ranges: "
        f"{preferred_samples[0].wavelength_nm} to {preferred_samples[-1].wavelength_nm} nm, "
        f"{fallback_samples[0].wavelength_nm} to {fallback_samples[-1].wavelength_nm} nm"
    )


def interpolate_differential_polarizability(
    wavelength_nm: float,
    samples: list[DifferentialPolarizabilitySample],
) -> DifferentialPolarizabilitySample:
    """Linearly interpolate the differential-polarizability table in wavelength."""

    if not samples:
        raise ValueError("samples must be non-empty")
    wavelengths = [sample.wavelength_nm for sample in samples]
    if wavelength_nm < wavelengths[0] or wavelength_nm > wavelengths[-1]:
        raise ValueError(
            f"wavelength {wavelength_nm} nm is outside table range "
            f"{wavelengths[0]} nm to {wavelengths[-1]} nm"
        )
    upper_index = bisect_left(wavelengths, wavelength_nm)
    if upper_index < len(samples) and wavelengths[upper_index] == wavelength_nm:
        return samples[upper_index]
    lower = samples[upper_index - 1]
    upper = samples[upper_index]
    fraction = (wavelength_nm - lower.wavelength_nm) / (upper.wavelength_nm - lower.wavelength_nm)
    return DifferentialPolarizabilitySample(
        wavelength_nm=wavelength_nm,
        scalar_si=lower.scalar_si + fraction * (upper.scalar_si - lower.scalar_si),
        vector_si=lower.vector_si + fraction * (upper.vector_si - lower.vector_si),
        tensor_si=lower.tensor_si + fraction * (upper.tensor_si - lower.tensor_si),
    )


def convert_differential_polarizability_to_mhz_per_intensity(
    sample: DifferentialPolarizabilitySample,
) -> DifferentialShiftCoefficients:
    """Convert raw polarizabilities to MHz/[mW/(100 um)^2]."""

    scalar_vector_factor = (
        -2.0
        / SPEED_OF_LIGHT_M_PER_S
        / PLANCK_CONSTANT_J_S
        / VACUUM_PERMITTIVITY_F_PER_M
        / 1.0e6
        * 1.0e5
    )
    tensor_factor = (
        1.0
        / SPEED_OF_LIGHT_M_PER_S
        / PLANCK_CONSTANT_J_S
        / VACUUM_PERMITTIVITY_F_PER_M
        / 1.0e6
        * 1.0e5
    )
    return DifferentialShiftCoefficients(
        wavelength_nm=sample.wavelength_nm,
        scalar_mhz_per_intensity=sample.scalar_si * scalar_vector_factor,
        vector_mhz_per_intensity=sample.vector_si * scalar_vector_factor,
        tensor_mhz_per_intensity=sample.tensor_si * tensor_factor,
    )


@lru_cache(maxsize=None)
def differential_shift_coefficients_for_wavelength(
    wavelength_nm: float,
    csv_path: Path | None = None,
) -> DifferentialShiftCoefficients:
    """Return converted differential-shift coefficients for one wavelength."""

    path = csv_path or choose_polarizability_csv_path(wavelength_nm)
    samples = load_differential_polarizability_csv(path)
    sample = interpolate_differential_polarizability(wavelength_nm, samples)
    return convert_differential_polarizability_to_mhz_per_intensity(sample)


def polarizability_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    """Return the tabulated polarizability data with converted coefficients."""

    path = csv_path or default_polarizability_csv_path()
    dataframe = pd.read_csv(path)
    sv_factor = (
        -2.0
        / SPEED_OF_LIGHT_M_PER_S
        / PLANCK_CONSTANT_J_S
        / VACUUM_PERMITTIVITY_F_PER_M
        / 1.0e6
        * 1.0e5
    )
    tensor_factor = (
        1.0
        / SPEED_OF_LIGHT_M_PER_S
        / PLANCK_CONSTANT_J_S
        / VACUUM_PERMITTIVITY_F_PER_M
        / 1.0e6
        * 1.0e5
    )
    dataframe = dataframe.rename(
        columns={
            "Wavelength (nm)": "wavelength_nm",
            "Differential Scalar Polarizability": "scalar_si",
            "Differential Vector Polarizability": "vector_si",
            "Differential Tensor Polarizability": "tensor_si",
        }
    ).copy()
    dataframe["frequency_thz"] = SPEED_OF_LIGHT_M_PER_S / (dataframe["wavelength_nm"] * 1e-9) / 1e12
    dataframe["scalar_mhz_per_intensity"] = dataframe["scalar_si"] * sv_factor
    dataframe["vector_mhz_per_intensity"] = dataframe["vector_si"] * sv_factor
    dataframe["tensor_mhz_per_intensity"] = dataframe["tensor_si"] * tensor_factor
    return dataframe

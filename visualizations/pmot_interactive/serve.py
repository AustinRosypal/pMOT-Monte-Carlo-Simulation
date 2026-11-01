"""Local server for the interactive effective-two-level pMOT visualizations.

Run from the repository root with::

    uv run python visualizations/pmot_interactive/serve.py --open

The server binds only to 127.0.0.1.  Dynamic endpoints evaluate the live
``pmot.pMOT_2Level`` implementation; the HTML pages do not contain a duplicate
JavaScript force model.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from dataclasses import replace
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

import numpy as np


VISUALIZATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALIZATION_ROOT.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pmot.configuration import PLANCK_CONSTANT_J_S  # noqa: E402
from pmot.configuration import SPEED_OF_LIGHT_M_PER_S  # noqa: E402
from pmot.magnetic_fields import default_anti_helmholtz_config  # noqa: E402
from pmot.mot_multilevel import build_multilevel_mot_beams  # noqa: E402
from pmot.mot_multilevel import build_rate_equation_model  # noqa: E402
from pmot.mot_multilevel import default_multilevel_mot_config  # noqa: E402
from pmot.mot_multilevel import rate_equation_observable  # noqa: E402
from pmot.mot_multilevel.accelerated import observable as accelerated_observable  # noqa: E402
from pmot.mot_multilevel.accelerated import prepare as prepare_accelerated  # noqa: E402
from pmot.mot_simple.configuration import default_simple_mot_apparatus  # noqa: E402
from pmot.mot_simple.configuration import default_simple_mot_config  # noqa: E402
from pmot.mot_strange.polarizability import (  # noqa: E402
    differential_shift_coefficients_for_wavelength,
)
from pmot.mot_strange.polarizability import polarizability_dataframe  # noqa: E402
from pmot.mot_strange.surrogate_effective_field import (  # noqa: E402
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
)
from pmot.mot_strange.surrogate_effective_field import (  # noqa: E402
    numerical_field_jacobian_t_per_m,
)
from pmot.mot_strange.surrogate_effective_field import (  # noqa: E402
    surrogate_effective_field_t,
)
from pmot.pMOT_2Level import PMOT_2LEVEL_PATHWAYS  # noqa: E402
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams  # noqa: E402
from pmot.pMOT_2Level import evaluate_differential_ac_stark_shift  # noqa: E402
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable  # noqa: E402
from pmot.pMOT_2Level import get_pmot_2level_pathway  # noqa: E402
from pmot.pMOT_2Level import (  # noqa: E402
    load_pmot_2level_component_polarizability_table,
)
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table  # noqa: E402
from pmot.pMOT_2Level import normalize_pmot_stark_component_code  # noqa: E402
from pmot.pMOT_2Level import pmot_stark_component_model_label  # noqa: E402


AXIS_NAMES = ("x", "y", "z")
FALLBACK_AXIS = (0.0, 0.0, 1.0)
_POLARIZABILITY_FRAME = None
_POLARIZABILITY_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _one(query: dict[str, list[str]], name: str, default: str) -> str:
    values = query.get(name)
    return default if not values else values[-1]


def _float(
    query: dict[str, list[str]],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = float(_one(query, name, str(default)))
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _int(
    query: dict[str, list[str]],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = int(_one(query, name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _axis(query: dict[str, list[str]], name: str = "axis") -> tuple[str, int]:
    axis_name = _one(query, name, "x")
    if axis_name not in AXIS_NAMES:
        raise ValueError(f"{name} must be x, y, or z")
    return axis_name, AXIS_NAMES.index(axis_name)


def _pathway_name(query: dict[str, list[str]]) -> str:
    name = _one(query, "pathway", "scalar_zero")
    get_pmot_2level_pathway(name)
    return name


def _path_powers_mw(query: dict[str, list[str]]) -> tuple[float, float, float]:
    return tuple(
        _float(query, f"trap_{axis}_mw", 46.0, minimum=0.0, maximum=1000.0)
        for axis in AXIS_NAMES
    )


@lru_cache(maxsize=128)
def _model_context(
    pathway_name: str,
    cooling_power_mw: float,
    cooling_detuning_mhz: float,
    cooling_diameter_mm: float,
):
    pathway = get_pmot_2level_pathway(pathway_name)
    baseline = default_simple_mot_apparatus()
    apparatus = replace(
        baseline,
        cooling=replace(
            baseline.cooling,
            power_w_per_beam=1.0e-3 * cooling_power_mw,
            detuning_hz=1.0e6 * cooling_detuning_mhz,
            beam_diameter_m=1.0e-3 * cooling_diameter_mm,
        ),
    )
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=1.0e6 * cooling_detuning_mhz,
        include_gravity=False,
    )
    cooling = tuple(
        build_pmot_2level_cooling_beams(
            apparatus_config=apparatus,
            simple_config=simple,
        )
    )
    trapping = pathway.build_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    return pathway, simple, cooling, trapping, table


def _common_model_parameters(query: dict[str, list[str]]):
    pathway_name = _pathway_name(query)
    components = normalize_pmot_stark_component_code(
        _one(query, "components", "svt")
    )
    cooling_power_mw = _float(
        query, "cooling_power_mw", 6.0, minimum=0.0, maximum=1000.0
    )
    cooling_detuning_mhz = _float(
        query, "cooling_detuning_mhz", -30.0, minimum=-500.0, maximum=100.0
    )
    cooling_diameter_mm = _float(
        query, "cooling_diameter_mm", 12.7, minimum=0.1, maximum=100.0
    )
    powers_mw = _path_powers_mw(query)
    context = _model_context(
        pathway_name,
        cooling_power_mw,
        cooling_detuning_mhz,
        cooling_diameter_mm,
    )
    pathway, simple, cooling, trapping, _ = context
    context = (
        pathway,
        simple,
        cooling,
        trapping,
        load_pmot_2level_component_polarizability_table(components),
    )
    return context, powers_mw


def _polarizability_data():
    global _POLARIZABILITY_FRAME
    if _POLARIZABILITY_FRAME is None:
        with _POLARIZABILITY_LOCK:
            if _POLARIZABILITY_FRAME is None:
                _POLARIZABILITY_FRAME = polarizability_dataframe()
    return _POLARIZABILITY_FRAME


def _downsample_extrema(frame, max_points: int):
    if len(frame) <= max_points:
        return frame
    value_columns = (
        "scalar_mhz_per_intensity",
        "vector_mhz_per_intensity",
        "tensor_mhz_per_intensity",
    )
    bin_count = max(1, max_points // (2 * len(value_columns)))
    edges = np.linspace(0, len(frame), bin_count + 1, dtype=int)
    selected: set[int] = {0, len(frame) - 1}
    arrays = [frame[column].to_numpy() for column in value_columns]
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue
        for values in arrays:
            segment = values[start:stop]
            selected.add(start + int(np.argmin(segment)))
            selected.add(start + int(np.argmax(segment)))
    return frame.iloc[sorted(selected)]


def api_meta(_query: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "status": "PROVISIONAL_EFFECTIVE_TWO_LEVEL_PMOT_DIAGNOSTIC",
        "pathways": [
            {
                "key": pathway.key,
                "display_name": pathway.display_name,
                "wavelength_nm": pathway.trapping_wavelength_nm,
                "condition": pathway.polarizability_condition,
                "geometry_note": pathway.geometry_note,
            }
            for pathway in PMOT_2LEVEL_PATHWAYS.values()
        ],
        "limitations": [
            "Effective F=2 to F'=3 differential transition closure, not an mF-resolved Hamiltonian.",
            "Radiation-pressure force only; gravity is excluded from force diagnostics.",
            "No conservative Stark-gradient force, 1529-nm scattering/heating, recoil diffusion, or loading prediction.",
        ],
    }


def api_polarizability(query: dict[str, list[str]]) -> dict[str, Any]:
    frame = _polarizability_data()
    full_min = float(frame["wavelength_nm"].iloc[0])
    full_max = float(frame["wavelength_nm"].iloc[-1])
    lower = _float(query, "min_nm", full_min, minimum=full_min, maximum=full_max)
    upper = _float(query, "max_nm", full_max, minimum=full_min, maximum=full_max)
    if upper <= lower:
        raise ValueError("max_nm must be greater than min_nm")
    # The complete narrow-band table contains only 70,400 rows, so the browser
    # explorer can receive every CSV sample without a lossy display grid.  A
    # lower explicit limit remains available for API clients that prefer an
    # extrema-preserving overview.
    max_points = _int(query, "max_points", 100000, minimum=300, maximum=100000)
    selected = frame[
        (frame["wavelength_nm"] >= lower) & (frame["wavelength_nm"] <= upper)
    ]
    if len(selected) < 2:
        raise ValueError("selected interval contains fewer than two samples")
    displayed = _downsample_extrema(selected, max_points)
    exact_samples = len(displayed) == len(selected)
    markers = []
    for pathway in PMOT_2LEVEL_PATHWAYS.values():
        coefficients = differential_shift_coefficients_for_wavelength(
            pathway.trapping_wavelength_nm
        )
        markers.append(
            {
                "key": pathway.key,
                "label": pathway.display_name,
                "wavelength_nm": pathway.trapping_wavelength_nm,
                "frequency_thz": (
                    SPEED_OF_LIGHT_M_PER_S
                    / (pathway.trapping_wavelength_nm * 1.0e-9)
                    / 1.0e12
                ),
                "scalar": coefficients.scalar_mhz_per_intensity,
                "vector": coefficients.vector_mhz_per_intensity,
                "tensor": coefficients.tensor_mhz_per_intensity,
            }
        )
    return {
        "range_nm": [lower, upper],
        "full_range_nm": [full_min, full_max],
        "source_rows": len(selected),
        "display_rows": len(displayed),
        "exact_samples": exact_samples,
        "median_sample_spacing_nm": float(
            np.median(np.diff(selected["wavelength_nm"].to_numpy()))
        ),
        "wavelength_nm": displayed["wavelength_nm"].to_numpy(),
        "frequency_thz": displayed["frequency_thz"].to_numpy(),
        "scalar": displayed["scalar_mhz_per_intensity"].to_numpy(),
        "vector": displayed["vector_mhz_per_intensity"].to_numpy(),
        "tensor": displayed["tensor_mhz_per_intensity"].to_numpy(),
        "markers": markers,
        "resonances": [
            {
                "frequency_thz": 196.0245,
                "label": "5p3/2 → 4d5/2 resonance",
            },
            {
                "frequency_thz": 196.0380,
                "label": "5p3/2 → 4d3/2 resonance",
            },
        ],
        "units": "MHz per [mW/(100 um)^2]",
    }


def _line_position(axis_index: int, coordinate_m: float, offsets_m: tuple[float, float]):
    position = np.zeros(3)
    position[axis_index] = coordinate_m
    other = [index for index in range(3) if index != axis_index]
    position[other[0]], position[other[1]] = offsets_m
    return position


def api_force_line(query: dict[str, list[str]]) -> dict[str, Any]:
    (pathway, simple, cooling, trapping, table), powers_mw = (
        _common_model_parameters(query)
    )
    axis_name, axis_index = _axis(query)
    other_names = [name for name in AXIS_NAMES if name != axis_name]
    offset_1_mm = _float(query, "offset_1_mm", 0.0, minimum=-25.0, maximum=25.0)
    offset_2_mm = _float(query, "offset_2_mm", 0.0, minimum=-25.0, maximum=25.0)
    velocity_m_per_s = _float(
        query, "velocity_m_per_s", 0.0, minimum=-100.0, maximum=100.0
    )
    lower_mm = _float(query, "min_mm", -12.5, minimum=-50.0, maximum=50.0)
    upper_mm = _float(query, "max_mm", 12.5, minimum=-50.0, maximum=50.0)
    if upper_mm <= lower_mm:
        raise ValueError("max_mm must be greater than min_mm")
    points = _int(query, "points", 251, minimum=31, maximum=801)
    coordinates_mm = np.linspace(lower_mm, upper_mm, points)
    force_n = np.empty(points)
    velocity = np.zeros(3)
    velocity[axis_index] = velocity_m_per_s
    powers_w = 1.0e-3 * np.asarray(powers_mw)
    offsets_m = (1.0e-3 * offset_1_mm, 1.0e-3 * offset_2_mm)
    for index, coordinate_mm in enumerate(coordinates_mm):
        observable = evaluate_pmot_2level_local_observable(
            cooling,
            trapping,
            _line_position(axis_index, 1.0e-3 * coordinate_mm, offsets_m),
            velocity,
            path_powers_w=powers_w,
            previous_axis=FALLBACK_AXIS,
            simple_config=simple,
            polarizability_table=table,
        )
        force_n[index] = observable.force_n[axis_index]
    coordinates_m = 1.0e-3 * coordinates_mm
    gradient_n_per_m = np.gradient(force_n, coordinates_m, edge_order=2)
    center_index = int(np.argmin(np.abs(coordinates_mm)))
    central_mask = np.abs(coordinates_mm) <= 1.0
    restoring_area_zn = 1.0e21 * float(
        np.trapezoid(
            np.maximum(-gradient_n_per_m[central_mask], 0.0),
            coordinates_m[central_mask],
        )
    )
    return {
        "pathway": pathway.key,
        "stark_components": (
            normalize_pmot_stark_component_code(
                _one(query, "components", "svt")
            )
            or "none"
        ),
        "model_label": pmot_stark_component_model_label(
            _one(query, "components", "svt")
        ),
        "wavelength_nm": pathway.trapping_wavelength_nm,
        "axis": axis_name,
        "offset_coordinates": other_names,
        "offsets_mm": [offset_1_mm, offset_2_mm],
        "velocity_m_per_s": velocity_m_per_s,
        "coordinate_mm": coordinates_mm,
        "force_zN": 1.0e21 * force_n,
        "gradient_zN_per_mm": 1.0e18 * gradient_n_per_m,
        "metrics": {
            "origin_force_zN": 1.0e21 * force_n[center_index],
            "origin_gradient_zN_per_mm": 1.0e18 * gradient_n_per_m[center_index],
            "negative_gradient_area_pm1mm_zN": restoring_area_zn,
        },
    }


def _plane_definition(plane: str):
    if plane == "xy":
        return (0, 1, 2)
    if plane == "xz":
        return (0, 2, 1)
    if plane == "yz":
        return (1, 2, 0)
    raise ValueError("plane must be xy, xz, or yz")


def api_stark_plane(query: dict[str, list[str]]) -> dict[str, Any]:
    (pathway, _simple, _cooling, trapping, table), powers_mw = (
        _common_model_parameters(query)
    )
    plane = _one(query, "plane", "xy")
    horizontal_index, vertical_index, slice_index = _plane_definition(plane)
    extent_mm = _float(query, "extent_mm", 3.0, minimum=0.1, maximum=15.0)
    slice_mm = _float(query, "slice_mm", 0.0, minimum=-15.0, maximum=15.0)
    points = _int(query, "points", 35, minimum=15, maximum=61)
    velocity = np.zeros(3)
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        velocity[axis_index] = _float(
            query, f"v{axis_name}_m_per_s", 0.0, minimum=-100.0, maximum=100.0
        )
    coordinates_mm = np.linspace(-extent_mm, extent_mm, points)
    shape = (points, points)
    scalar = np.empty(shape)
    vector = np.empty(shape)
    tensor = np.empty(shape)
    intensity = np.empty(shape)
    axes = np.empty(shape + (3,))
    axis_defined = np.empty(shape, dtype=bool)
    powers_w = 1.0e-3 * np.asarray(powers_mw)
    for row, vertical_mm in enumerate(coordinates_mm):
        for column, horizontal_mm in enumerate(coordinates_mm):
            position = np.zeros(3)
            position[horizontal_index] = 1.0e-3 * horizontal_mm
            position[vertical_index] = 1.0e-3 * vertical_mm
            position[slice_index] = 1.0e-3 * slice_mm
            sample = evaluate_differential_ac_stark_shift(
                trapping,
                position,
                velocity,
                path_powers_w=powers_w,
                previous_axis=FALLBACK_AXIS,
                polarizability_table=table,
            )
            scalar[row, column] = sample.total_scalar_energy_j / PLANCK_CONSTANT_J_S / 1.0e6
            vector[row, column] = sample.total_vector_energy_j / PLANCK_CONSTANT_J_S / 1.0e6
            tensor[row, column] = sample.total_tensor_energy_j / PLANCK_CONSTANT_J_S / 1.0e6
            intensity[row, column] = sum(sample.component_intensities_w_per_m2)
            axes[row, column] = sample.quantization_axis
            axis_defined[row, column] = sample.quantization_axis_from_vector_shift
    return {
        "pathway": pathway.key,
        "wavelength_nm": pathway.trapping_wavelength_nm,
        "plane": plane,
        "slice_axis": AXIS_NAMES[slice_index],
        "slice_mm": slice_mm,
        "horizontal_axis": AXIS_NAMES[horizontal_index],
        "vertical_axis": AXIS_NAMES[vertical_index],
        "coordinate_mm": coordinates_mm,
        "scalar_mhz": scalar,
        "vector_mhz": vector,
        "tensor_mhz": tensor,
        "common_mhz": scalar + tensor,
        "intensity_w_per_m2": intensity,
        "axis_u": axes[:, :, horizontal_index],
        "axis_v": axes[:, :, vertical_index],
        "axis_out": axes[:, :, slice_index],
        "axis_defined": axis_defined,
    }


def api_phase_space(query: dict[str, list[str]]) -> dict[str, Any]:
    (pathway, simple, cooling, trapping, table), powers_mw = (
        _common_model_parameters(query)
    )
    axis_name, axis_index = _axis(query)
    position_extent_mm = _float(
        query, "position_extent_mm", 5.0, minimum=0.1, maximum=15.0
    )
    velocity_extent_m_per_s = _float(
        query, "velocity_extent_m_per_s", 25.0, minimum=0.1, maximum=100.0
    )
    position_points = _int(
        query, "position_points", 41, minimum=15, maximum=61
    )
    velocity_points = _int(
        query, "velocity_points", 41, minimum=15, maximum=61
    )
    offset_1_mm = _float(query, "offset_1_mm", 0.0, minimum=-15.0, maximum=15.0)
    offset_2_mm = _float(query, "offset_2_mm", 0.0, minimum=-15.0, maximum=15.0)
    positions_mm = np.linspace(-position_extent_mm, position_extent_mm, position_points)
    velocities = np.linspace(
        -velocity_extent_m_per_s,
        velocity_extent_m_per_s,
        velocity_points,
    )
    force_z_n = np.empty((velocity_points, position_points))
    powers_w = 1.0e-3 * np.asarray(powers_mw)
    offsets_m = (1.0e-3 * offset_1_mm, 1.0e-3 * offset_2_mm)
    for row, velocity_value in enumerate(velocities):
        velocity = np.zeros(3)
        velocity[axis_index] = velocity_value
        for column, position_mm in enumerate(positions_mm):
            observable = evaluate_pmot_2level_local_observable(
                cooling,
                trapping,
                _line_position(axis_index, 1.0e-3 * position_mm, offsets_m),
                velocity,
                path_powers_w=powers_w,
                previous_axis=FALLBACK_AXIS,
                simple_config=simple,
                polarizability_table=table,
            )
            force_z_n[row, column] = 1.0e21 * observable.force_n[axis_index]
    return {
        "pathway": pathway.key,
        "wavelength_nm": pathway.trapping_wavelength_nm,
        "axis": axis_name,
        "position_mm": positions_mm,
        "velocity_m_per_s": velocities,
        "force_zN": force_z_n,
    }


def _comparison_multilevel_config(
    cooling_power_mw: float,
    repump_power_mw: float,
    cooling_detuning_mhz: float,
    repump_detuning_mhz: float,
):
    return replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=1.0e-3 * cooling_power_mw,
        repump_power_w_per_beam=1.0e-3 * repump_power_mw,
        cooling_detuning_rad_per_s=2.0 * np.pi * 1.0e6 * cooling_detuning_mhz,
        repump_detuning_rad_per_s=2.0 * np.pi * 1.0e6 * repump_detuning_mhz,
        include_gravity=False,
    )


@lru_cache(maxsize=32)
def _comparison_conventional_context(
    cooling_power_mw: float,
    repump_power_mw: float,
    cooling_detuning_mhz: float,
    repump_detuning_mhz: float,
    axial_gradient_g_per_cm: float,
):
    config = _comparison_multilevel_config(
        cooling_power_mw,
        repump_power_mw,
        cooling_detuning_mhz,
        repump_detuning_mhz,
    )
    coil = (
        replace(default_anti_helmholtz_config(), current_a=0.0)
        if axial_gradient_g_per_cm == 0.0
        else default_anti_helmholtz_config(
            target_gradient_g_per_cm=axial_gradient_g_per_cm
        )
    )
    return config, coil, prepare_accelerated(config=config, coil=coil)


@lru_cache(maxsize=32)
def _comparison_strange_context(
    cooling_power_mw: float,
    repump_power_mw: float,
    cooling_detuning_mhz: float,
    repump_detuning_mhz: float,
):
    config = _comparison_multilevel_config(
        cooling_power_mw,
        repump_power_mw,
        cooling_detuning_mhz,
        repump_detuning_mhz,
    )
    return config, build_rate_equation_model(), tuple(
        build_multilevel_mot_beams(config=config)
    )


def _comparison_pathway(
    query: dict[str, list[str]], name: str, default: str
):
    pathway_name = _one(query, name, default)
    return get_pmot_2level_pathway(pathway_name)


def api_model_comparison(query: dict[str, list[str]]) -> dict[str, Any]:
    """Return live force and same-line force-gradient overlays for four models."""

    scan_limit_mm = _float(
        query, "scan_limit_mm", 12.5, minimum=0.5, maximum=25.0
    )
    points = _int(query, "points", 81, minimum=31, maximum=201)
    offset_1_mm = _float(
        query, "offset_1_mm", 0.0, minimum=-15.0, maximum=15.0
    )
    offset_2_mm = _float(
        query, "offset_2_mm", 0.0, minimum=-15.0, maximum=15.0
    )
    velocity_m_per_s = _float(
        query, "velocity_m_per_s", 0.0, minimum=-100.0, maximum=100.0
    )

    mot_cooling = _float(query, "mot_cooling_mw", 27.0, minimum=0.0, maximum=1000.0)
    mot_repump = _float(query, "mot_repump_mw", 0.1, minimum=0.0, maximum=100.0)
    mot_detuning = _float(query, "mot_detuning_mhz", -15.0, minimum=-500.0, maximum=100.0)
    mot_repump_detuning = _float(query, "mot_repump_detuning_mhz", 0.0, minimum=-100.0, maximum=100.0)
    mot_axial_gradient = _float(query, "mot_axial_gradient_g_per_cm", 20.0, minimum=0.0, maximum=200.0)

    strange_cooling = _float(query, "strange_cooling_mw", 27.0, minimum=0.0, maximum=1000.0)
    strange_repump = _float(query, "strange_repump_mw", 0.1, minimum=0.0, maximum=100.0)
    strange_detuning = _float(query, "strange_detuning_mhz", -15.0, minimum=-500.0, maximum=100.0)
    strange_repump_detuning = _float(query, "strange_repump_detuning_mhz", 0.0, minimum=-100.0, maximum=100.0)
    strange_powers_mw = tuple(
        _float(
            query,
            f"strange_trap_{axis}_mw",
            default,
            minimum=0.0,
            maximum=1000.0,
        )
        for axis, default in zip(AXIS_NAMES, (10.8, 10.8, 21.6), strict=True)
    )

    full_pathway = _comparison_pathway(query, "full_pathway", "scalar_zero")
    full_components = normalize_pmot_stark_component_code(
        _one(query, "full_components", "svt")
    )
    full_cooling = _float(query, "full_cooling_mw", 1.0571, minimum=0.0, maximum=1000.0)
    full_detuning = _float(query, "full_detuning_mhz", -15.0, minimum=-500.0, maximum=100.0)
    full_powers_mw = tuple(
        _float(query, f"full_trap_{axis}_mw", 31.524310589374092, minimum=0.0, maximum=1000.0)
        for axis in AXIS_NAMES
    )

    vector_pathway = _comparison_pathway(
        query, "vector_pathway", "scalar_tensor_cancellation"
    )
    vector_components = normalize_pmot_stark_component_code(
        _one(query, "vector_components", "v")
    )
    vector_cooling = _float(query, "vector_cooling_mw", 1.0571, minimum=0.0, maximum=1000.0)
    vector_detuning = _float(query, "vector_detuning_mhz", -15.0, minimum=-500.0, maximum=100.0)
    vector_powers_mw = tuple(
        _float(query, f"vector_trap_{axis}_mw", 31.524310589374092, minimum=0.0, maximum=1000.0)
        for axis in AXIS_NAMES
    )

    _, mot_coil, conventional = _comparison_conventional_context(
        mot_cooling,
        mot_repump,
        mot_detuning,
        mot_repump_detuning,
        mot_axial_gradient,
    )
    strange_config, strange_model, strange_beams = _comparison_strange_context(
        strange_cooling,
        strange_repump,
        strange_detuning,
        strange_repump_detuning,
    )
    strange_field_config = replace(
        DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
        x_path_power_w=1.0e-3 * strange_powers_mw[0],
        y_path_power_w=1.0e-3 * strange_powers_mw[1],
        z_path_power_w=1.0e-3 * strange_powers_mw[2],
    )

    full_simple, full_cooling_beams, full_trapping_beams, full_table = (
        _model_context(
            full_pathway.key,
            full_cooling,
            full_detuning,
            12.7,
        )[1:]
    )
    vector_simple, vector_cooling_beams, vector_trapping_beams, _ = (
        _model_context(
            vector_pathway.key,
            vector_cooling,
            vector_detuning,
            12.7,
        )[1:]
    )
    full_table = load_pmot_2level_component_polarizability_table(full_components)
    vector_table = load_pmot_2level_component_polarizability_table(
        vector_components
    )

    coordinates_mm = np.linspace(-scan_limit_mm, scan_limit_mm, points)
    coordinates_m = 1.0e-3 * coordinates_mm
    offsets_m = (1.0e-3 * offset_1_mm, 1.0e-3 * offset_2_mm)
    fallback_axis = np.asarray(FALLBACK_AXIS)
    full_powers_w = 1.0e-3 * np.asarray(full_powers_mw)
    vector_powers_w = 1.0e-3 * np.asarray(vector_powers_mw)
    force_by_model = {
        model: {axis: np.empty(points) for axis in AXIS_NAMES}
        for model in ("mot_multilevel", "mot_strange", "pMOT_2Level", "pmot_vec_2Level")
    }

    def strange_field(position_m):
        return surrogate_effective_field_t(position_m, strange_field_config)

    started = perf_counter()
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        velocity = np.zeros(3)
        velocity[axis_index] = velocity_m_per_s
        for index, coordinate_m in enumerate(coordinates_m):
            position = _line_position(axis_index, coordinate_m, offsets_m)
            conventional_force, _, _, _ = accelerated_observable(
                position,
                velocity,
                fallback_axis,
                conventional,
            )
            force_by_model["mot_multilevel"][axis_name][index] = (
                conventional_force[axis_index]
            )

            strange = rate_equation_observable(
                strange_model,
                strange_beams,
                tuple(position),
                tuple(velocity),
                None,
                strange_config,
                FALLBACK_AXIS,
                magnetic_field_function=strange_field,
            )
            force_by_model["mot_strange"][axis_name][index] = strange.force_n[
                axis_index
            ]

            full = evaluate_pmot_2level_local_observable(
                full_cooling_beams,
                full_trapping_beams,
                position,
                velocity,
                path_powers_w=full_powers_w,
                previous_axis=FALLBACK_AXIS,
                simple_config=full_simple,
                polarizability_table=full_table,
            )
            force_by_model["pMOT_2Level"][axis_name][index] = full.force_n[
                axis_index
            ]

            vector = evaluate_pmot_2level_local_observable(
                vector_cooling_beams,
                vector_trapping_beams,
                position,
                velocity,
                path_powers_w=vector_powers_w,
                previous_axis=FALLBACK_AXIS,
                simple_config=vector_simple,
                polarizability_table=vector_table,
            )
            force_by_model["pmot_vec_2Level"][axis_name][index] = vector.force_n[
                axis_index
            ]

    model_payload = {}
    for model_name, force_by_axis in force_by_model.items():
        label = {
            "mot_multilevel": "Multilevel MOT",
            "mot_strange": "mot_strange",
            "pMOT_2Level": pmot_stark_component_model_label(full_components),
            "pmot_vec_2Level": pmot_stark_component_model_label(
                vector_components
            ),
        }[model_name]
        model_payload[model_name] = {
            "label": label,
            "force_zN": {
                axis: 1.0e21 * values for axis, values in force_by_axis.items()
            },
            "gradient_1e19_n_per_m": {
                axis: 1.0e19
                * np.gradient(values, coordinates_m, edge_order=2)
                for axis, values in force_by_axis.items()
            },
        }

    strange_gradient = 100.0 * np.diag(
        numerical_field_jacobian_t_per_m(
            step_m=1.0e-7,
            config=strange_field_config,
        )
    )
    return {
        "coordinate_mm": coordinates_mm,
        "models": model_payload,
        "line": {
            "offset_1_mm": offset_1_mm,
            "offset_2_mm": offset_2_mm,
            "velocity_m_per_s": velocity_m_per_s,
            "axis_offset_order": {
                "x": ["y", "z"],
                "y": ["x", "z"],
                "z": ["x", "y"],
            },
        },
        "parameters": {
            "mot_multilevel": {
                "cooling_power_mw_per_component": mot_cooling,
                "repump_power_mw_per_component": mot_repump,
                "cooling_detuning_mhz": mot_detuning,
                "repump_detuning_mhz": mot_repump_detuning,
                "field_gradient_g_per_cm_xyz": [
                    0.5 * mot_axial_gradient,
                    0.5 * mot_axial_gradient,
                    -mot_axial_gradient,
                ],
            },
            "mot_strange": {
                "cooling_power_mw_per_component": strange_cooling,
                "repump_power_mw_per_component": strange_repump,
                "cooling_detuning_mhz": strange_detuning,
                "repump_detuning_mhz": strange_repump_detuning,
                "surrogate_path_powers_mw_xyz": strange_powers_mw,
                "origin_effective_field_gradient_g_per_cm_xyz": strange_gradient,
                "cell_half_length_mm": 1.0e3 * strange_field_config.cell_half_length_m,
            },
            "pMOT_2Level": {
                "cooling_power_mw_per_component": full_cooling,
                "cooling_detuning_mhz": full_detuning,
                "trapping_path_powers_mw_xyz": full_powers_mw,
                "pathway": full_pathway.key,
                "stark_components": full_components or "none",
                "model_label": pmot_stark_component_model_label(full_components),
                "trapping_wavelength_nm": full_pathway.trapping_wavelength_nm,
                "external_magnetic_field": "zero",
            },
            "pmot_vec_2Level": {
                "cooling_power_mw_per_component": vector_cooling,
                "cooling_detuning_mhz": vector_detuning,
                "trapping_path_powers_mw_xyz": vector_powers_mw,
                "pathway": vector_pathway.key,
                "stark_components": vector_components or "none",
                "model_label": pmot_stark_component_model_label(
                    vector_components
                ),
                "trapping_wavelength_nm": vector_pathway.trapping_wavelength_nm,
                "external_magnetic_field": "zero",
            },
        },
        "glass_wall_bands_mm": [[-12.5, -7.5], [7.5, 12.5]],
        "elapsed_seconds": perf_counter() - started,
        "scientific_status": (
            "stationary deterministic mean-force line cuts; gravity excluded; "
            "pMOT branches remain provisional transition-level diagnostics"
        ),
    }


API_ROUTES = {
    "/api/meta": api_meta,
    "/api/polarizability": api_polarizability,
    "/api/force-line": api_force_line,
    "/api/stark-plane": api_stark_plane,
    "/api/phase-space": api_phase_space,
    "/api/model-comparison": api_model_comparison,
}


class VisualizationHandler(SimpleHTTPRequestHandler):
    """Serve static pages plus JSON endpoints from the visualization root."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VISUALIZATION_ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = API_ROUTES.get(parsed.path)
        if route is None:
            super().do_GET()
            return
        try:
            payload = route(parse_qs(parsed.query))
            encoded = json.dumps(_jsonable(payload), separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(HTTPStatus.OK)
        except (ValueError, KeyError) as error:
            encoded = json.dumps({"error": str(error)}).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - defensive server boundary
            encoded = json.dumps(
                {"error": f"{type(error).__name__}: {error}"}
            ).encode("utf-8")
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def smoke_test() -> None:
    """Exercise every endpoint without starting a long-lived server."""

    checks = {
        "meta": api_meta({}),
        "polarizability": api_polarizability(
            {"min_nm": ["1529.267"], "max_nm": ["1529.275"], "max_points": ["600"]}
        ),
        "force-line": api_force_line({"points": ["31"]}),
        "stark-plane": api_stark_plane({"points": ["15"], "extent_mm": ["1"]}),
        "phase-space": api_phase_space(
            {"position_points": ["15"], "velocity_points": ["15"]}
        ),
        "model-comparison": api_model_comparison(
            {"points": ["31"], "scan_limit_mm": ["2"]}
        ),
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "checks": {
                    name: list(payload)[:6] for name, payload in checks.items()
                },
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the index page")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("this diagnostic server intentionally binds only to localhost")
    server = ThreadingHTTPServer((args.host, args.port), VisualizationHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"pMOT visualization suite: {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualization server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

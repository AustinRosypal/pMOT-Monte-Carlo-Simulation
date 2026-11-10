"""Validate the current symmetric surrogate and time representative scan rays."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json

import numpy as np
from scipy.optimize import brentq

from pmot.mot_multilevel import build_rate_equation_model, rate_equation_observable
from pmot.mot_strange import accelerated_surrogate
from pmot.mot_strange.loading_scan_inputs import (
    ScanPoint,
    cooling_configuration,
    cooling_saturation,
    current_symmetric_surrogate,
    field_axes,
    field_config,
    prepare,
    sample_launch_geometry,
    scan_points,
)
from pmot.mot_strange.loading_scan_restoring import (
    ROOT_RESIDUAL_M_PER_S2, stability_spectra, stable_spectra,
)
from pmot.mot_strange.surrogate_effective_field import (
    surrogate_component_intensities_w_per_m2,
    surrogate_effective_field_t,
)
from pmot.mot_strange.trapping_beams import (
    build_trapping_beams,
    trapping_beam_intensity_response_m_inv2,
)


OUTPUT = Path("outputs/diagnostics/mot_strange/symmetric_exact_D2_loading_20261105")
AXIS = np.array([0., 0., 1.])
REFERENCE_STATES = (
    (np.zeros(3), np.zeros(3)),
    (np.array([.001, -.002, .0004]), np.array([-2., .5, 1.])),
    (np.array([0., 0., -.009]), np.array([0., 0., 20.])),
    (np.array([.00999, .000003, -.000002]), np.array([-6., .1, .2])),
)


def validate_local(geometry):
    model = build_rate_equation_model()
    optical = build_trapping_beams()
    positions = np.vstack([state[0] for state in REFERENCE_STATES])
    points, _ = scan_points()
    records = []
    for point in points:
        signs = (1, -1) if point.geometry == "isotropic" else (-1,)
        for sign in signs:
            field = field_config(point, geometry, isotropic_axial_helicity=sign)
            cfg, beams = cooling_configuration(point)
            data = prepare(point, field)
            np.testing.assert_array_equal(data[10][:5], [0, 0, 0, 1e-12, 9.80665])
            powers = np.repeat([field.x_path_power_w, field.y_path_power_w, field.z_path_power_w], 2)
            expected_intensity = np.column_stack([
                power * trapping_beam_intensity_response_m_inv2(beam, positions)
                for power, beam in zip(powers, optical, strict=True)
            ])
            np.testing.assert_allclose(
                surrogate_component_intensities_w_per_m2(positions, field),
                expected_intensity, rtol=1e-10, atol=1e-8,
            )
            axes = field_axes(field)
            jac = np.asarray(axes["origin_surrogate_field_jacobian_g_per_cm"])
            expected_diagonal = (1, 1, sign if point.geometry == "isotropic" else -2)
            np.testing.assert_allclose(jac / jac[0, 0], np.diag(expected_diagonal), atol=1e-9)
            np.testing.assert_allclose(surrogate_effective_field_t(positions[0], field), 0, atol=1e-15)
            for r, v in REFERENCE_STATES:
                b = surrogate_effective_field_t(r, field)
                # The compiled radial square is |r|² - r_axis². At a micron-scale
                # waist a centimetre from the origin this subtraction has an
                # absolute roundoff proportional to |r|², not to the radial square.
                # Bound its impact on the Gaussian exponent using the minimum waist.
                min_waist_sq = min(beam.waist_radius_m**2 for beam in optical)
                exponent_roundoff = 16 * np.finfo(float).eps * (1 + np.dot(r, r) / min_waist_sq)
                np.testing.assert_allclose(
                    accelerated_surrogate.field(r, data), b,
                    rtol=exponent_roundoff, atol=1e-16,
                )
                actual = accelerated_surrogate.observable(r, v, AXIS, data)
                reference = rate_equation_observable(
                    model, beams, tuple(r), tuple(v), None, cfg, tuple(AXIS),
                    magnetic_field_function=lambda position: surrogate_effective_field_t(position, field),
                )
                np.testing.assert_allclose(actual[0], reference.force_n, rtol=4e-6, atol=3e-29)
                np.testing.assert_allclose(actual[1], reference.populations, rtol=4e-6, atol=5e-10)
            records.append({
                "point": asdict(point), "axial_helicity": sign,
                **axes, **cooling_saturation(point),
            })
    return records


def trajectory_probe(geometry):
    positions, directions, _ = sample_launch_geometry(3, 3, 2026110501, geometry)
    scenarios = (
        (ScanPoint(), -1, 0, 0),
        (ScanPoint(), -1, 1, 1),
        (ScanPoint("axial", 1), -1, 2, 0),
        (ScanPoint("axial", 40), -1, 1, 2),
        (ScanPoint("isotropic", 40), 1, 0, 1),
        (ScanPoint("isotropic", 40), -1, 0, 1),
        (ScanPoint("axial", 10, -40), -1, 2, 2),
        (ScanPoint("axial", 10, -15, 5), -1, 0, 2),
    )
    speeds = np.arange(0., 80.25, .25)
    setup = prepare(ScanPoint(), field_config(ScanPoint(), geometry, isotropic_axial_helicity=-1))
    accelerated_surrogate.integrate(positions[0, 0], 30 * directions[0], .2, setup)
    accelerated_surrogate.classify(positions[0, 0], 30 * directions[0], 5e-6, .2, setup)
    probes = []
    for point, sign, disc, index in scenarios:
        field = field_config(point, geometry, isotropic_axial_helicity=sign)
        data = prepare(point, field)
        r, direction = positions[disc, index], directions[disc]
        start = perf_counter()
        adaptive = [
            [int(accelerated_surrogate.integrate(r, speed * direction, .2, data, refinement)[0])
             for speed in speeds]
            for refinement in (1., 2.)
        ]
        adaptive_seconds = perf_counter() - start
        start = perf_counter()
        fixed = [
            [int(accelerated_surrogate.classify(r, speed * direction, step, .2, data)[0])
             for speed in speeds]
            for step in (5e-6, 2.5e-6)
        ]
        fixed_seconds = perf_counter() - start
        outcomes = np.asarray(adaptive + fixed)
        different = np.flatnonzero(np.any(outcomes != outcomes[0], axis=0))
        unresolved = np.flatnonzero(np.any(outcomes < 0, axis=0))
        probe = dict(
            point=asdict(point), axial_helicity=sign, launch_position_m=r.tolist(),
            direction=direction.tolist(), adaptive_seconds=adaptive_seconds,
            fixed_seconds=fixed_seconds, speeds_m_per_s=speeds.tolist(),
            adaptive_codes=adaptive, fixed_codes=fixed,
            differing_nodes_m_per_s=speeds[different].tolist(),
            timeout_nodes_m_per_s=speeds[unresolved].tolist(),
            status="agreement" if not len(different) and not len(unresolved) else "requires_escalation",
        )
        print(point, sign, f"adaptive {adaptive_seconds:.2f}s, fixed {fixed_seconds:.2f}s,",
              f"{len(different)} disagreements, {len(unresolved)} timeout nodes", flush=True)
        probes.append(probe)
        write_json("trajectory_probes.json", probes)
    return probes


def validate_core_stability(geometry):
    points, _ = scan_points()
    zero = np.zeros(3)
    z_grid = np.linspace(-.002, .002, 81)
    records = []
    for point in points:
        field = field_config(point, geometry, isotropic_axial_helicity=-1)
        data = prepare(point, field)

        def acceleration_z(z):
            return accelerated_surrogate.acceleration(z * AXIS, zero, AXIS, data)[2]

        values = [acceleration_z(z) for z in z_grid]
        roots = sorted({
            brentq(acceleration_z, a, b, xtol=1e-14)
            for a, b, fa, fb in zip(z_grid[:-1], z_grid[1:], values[:-1], values[1:], strict=True)
            if fa * fb <= 0
        })
        equilibria = []
        for z in roots:
            center = z * AXIS
            residual = np.linalg.norm(accelerated_surrogate.acceleration(center, zero, AXIS, data))
            spectra = stability_spectra(center, data)
            equilibria.append({
                "z_m": z,
                "acceleration_residual_m_per_s2": float(residual),
                "stable": bool(residual <= ROOT_RESIDUAL_M_PER_S2 and stable_spectra(spectra)),
                "spectra_per_s": [
                    [[float(e.real), float(e.imag)] for e in spectrum] for spectrum in spectra
                ],
            })
        records.append({"point": asdict(point), "axial_core_equilibria": equilibria})
    write_json("gravity_core_stability.json", records)
    stable_count = sum(any(root["stable"] for root in record["axial_core_equilibria"])
                       for record in records)
    print(f"Found a gravity-on stable core equilibrium for {stable_count}/{len(points)} settings.",
          flush=True)


def write_json(name, data):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    geometry = current_symmetric_surrogate()
    local = validate_local(geometry)
    write_json("local_reference_validation.json", {
        "status": "passed", "surrogate": geometry.metadata(),
        "reference_cases_per_setting": len(REFERENCE_STATES), "settings": local,
    })
    print(f"Passed intensity, field and full 24-state force checks at {len(local)} settings.", flush=True)
    validate_core_stability(geometry)
    if not args.local_only:
        trajectory_probe(geometry)


if __name__ == "__main__":
    main()

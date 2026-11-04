"""Save calibration, vapor audit, and in-cell pilot geometry; run no dynamics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann

from pmot.configuration import RB87_MASS_KG
from pmot.loading import (
    LOADING_RATE_PREFACTOR, SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3,
    THERMAL_SCALE_M2_PER_S2,
)
from pmot.mot_strange.surrogate_capture_campaign import CampaignConfig, build_launch_tasks
from pmot.mot_strange.surrogate_effective_field import (
    calibrated_surrogate_field_config, numerical_field_jacobian_t_per_m,
    surrogate_component_intensities_w_per_m2,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]
                        / "outputs/diagnostics/mot_strange/loading_rerun_10_10_20Gpcm_20261025_preflight")
    args = parser.parse_args()
    field = calibrated_surrogate_field_config()
    # One speed only materializes the geometry once; this is not a speed grid.
    campaign = CampaignConfig(seed=2026102501, direction_discs=32, points_per_disc=20,
                              disc_radius_m=0.015, speeds_m_per_s=(1.0,),
                              in_cell_launches=True, effective_field_config=field)
    tasks = build_launch_tasks(campaign)
    jacobians = {str(step): numerical_field_jacobian_t_per_m(step_m=step, config=field)*100
                 for step in (1e-6, 1e-7, 1e-8)}
    for jac in jacobians.values():
        np.testing.assert_allclose(jac, np.diag((10.0, 10.0, -20.0)), atol=1e-6, rtol=1e-7)
    center = surrogate_component_intensities_w_per_m2((0, 0, 0), field)
    powers = np.repeat([field.x_path_power_w, field.y_path_power_w, field.z_path_power_w], 2)
    ranges = np.tile([field.forward_rayleigh_range_m, field.return_rayleigh_range_m], 3)
    waist_squared = field.wavelength_m * ranges / np.pi
    components = [{"component": label, "power_w": float(power),
                   "origin_intensity_w_per_m2": float(intensity),
                   "origin_intensity_w_per_cm2": float(intensity/1e4),
                   "waist_radius_m": float(np.sqrt(w2)),
                   "waist_peak_intensity_w_per_m2": float(2*power/(np.pi*w2))}
                  for label, power, intensity, w2 in zip(
                      ("+x", "-x", "+y", "-y", "+z", "-z"), powers, center, waist_squared)]
    thermal = THERMAL_SCALE_M2_PER_S2
    normalized = 4/(np.sqrt(np.pi)*thermal**1.5)
    temperature = RB87_MASS_KG*thermal/(2*Boltzmann)
    density = LOADING_RATE_PREFACTOR/normalized
    nominal_density = LOADING_RATE_PREFACTOR/SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3
    vapor = {
        "historical_loading_prefactor": LOADING_RATE_PREFACTOR,
        "thermal_scale_m2_per_s2": thermal,
        "implied_temperature_k": temperature,
        "implied_temperature_c": temperature-273.15,
        "stated_speed_distribution_prefactor_s3_per_m3": SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3,
        "normalized_maxwell_prefactor_s3_per_m3": float(normalized),
        "integral_of_stated_speed_distribution": float(SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3/normalized),
        "equivalent_normalized_maxwell_n87_per_m3": float(density),
        "equivalent_rb87_partial_pressure_pa": float(density*Boltzmann*temperature),
        "equivalent_rb87_partial_pressure_torr": float(density*Boltzmann*temperature/133.32236842105263),
        "density_from_historical_unnormalized_prefactor_per_m3": float(nominal_density),
        "status": "inferred normalization, not a calibrated experimental vapor density",
        "decision": "preserve historical prefactor pending vapor decision; also report R/n87",
    }
    geometry = [asdict(t) for t in tasks]
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__),
                    root/"src/pmot/mot_strange/surrogate_effective_field.py",
                    root/"src/pmot/mot_strange/cell_launch_sampling.py",
                    root/"src/pmot/mot_strange/surrogate_capture_campaign.py",
                    root/"src/pmot/launch_geometry.py", root/"src/pmot/loading.py"]
    areas = np.asarray([tasks[d*20].launch_area_m2 for d in range(32)])
    payload = {
        "status": "preflight only; no trajectories or loading estimate",
        "field_configuration": field.metadata(),
        "gradient_jacobians_g_per_cm_by_step_m": {k: v.tolist() for k, v in jacobians.items()},
        "trapping_components": components,
        "incident_three_path_total_power_w": float(np.sum(powers)/2),
        "origin_incoherent_summed_intensity_w_per_m2": float(np.sum(center)),
        "vapor_audit": vapor,
        "geometry_preview_configuration": asdict(campaign),
        "geometry_sha256": hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(),
        "disc_inside_cell_area_min_mean_max_m2": [float(areas.min()), float(areas.mean()), float(areas.max())],
        "precision_plan": {"confidence": 0.95, "relative_half_width_target": 0.05,
                           "independent_units": "direction discs",
                           "allocation": "independent pilot determines a fixed production count before production"},
        "source_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_files},
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"preflight.json"
    encoded = json.dumps(payload, indent=2)+"\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("preflight differs from existing output; choose a fresh directory")
    manifest.write_text(encoded, encoding="utf-8")
    with (output/"launch_geometry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(geometry[0]))
        writer.writeheader()
        writer.writerows(geometry)
    print(json.dumps({"preflight": str(manifest), "powers_mw": (powers[::2]*1e3).tolist(),
                      "gradient_g_per_cm": np.diag(jacobians["1e-08"]).tolist(),
                      "launch_points": len(tasks), "vapor_audit": vapor}, indent=2))


if __name__ == "__main__":
    main()

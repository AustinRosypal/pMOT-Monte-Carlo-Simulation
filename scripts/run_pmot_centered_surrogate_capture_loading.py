"""Run the centered-surrogate pMOT capture/loading campaign.

The run deliberately uses a velocity-resolved direct mask rather than fitting a
single capture velocity to every ray.  Direction discs are the independent
clusters for confidence intervals.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.simulation import build_multilevel_mot_beams
from pmot.pmot.surrogate_capture_campaign import (
    CampaignConfig,
    run_campaign,
    run_representative_audits,
)
from pmot.pmot.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
)


RUN_NAME = "centered field full-sphere capture loading 25x10 r15mm 20261002"

CAMPAIGN = CampaignConfig(
    seed=20261002,
    direction_discs=25,
    points_per_disc=10,
    launch_radius_m=10.0e-3,
    disc_radius_m=15.0e-3,
    speeds_m_per_s=(
        *(float(value) for value in range(1, 26, 2)),
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
        60.0,
        70.0,
        80.0,
        100.0,
    ),
    duration_s=60.0e-3,
    time_step_s=5.0e-6,
    workers=8,
    # Stop launching new work after 10.5 h and reserve 1.5 h for draining,
    # aggregation, plots, and representative convergence checks.
    submission_budget_s=10.5 * 3600.0,
    absolute_wall_budget_s=12.0 * 3600.0,
)


def _write_preflight_metadata(output_dir: Path) -> None:
    mot = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=mot)
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]
    if len(cooling) != 6 or len(repump) != 6:
        raise RuntimeError("expected six cooling and six repump traveling components")
    if not all(np.isclose(beam.power_w, 27.0e-3) for beam in cooling):
        raise RuntimeError("the requested 27 mW cooling power is not active")
    if not all(np.isclose(2.0 * beam.beam_radius_m, 12.7e-3) for beam in cooling):
        raise RuntimeError("the requested 12.7 mm cooling-beam diameter is not active")
    if not np.isclose(CAMPAIGN.disc_radius_m, 15.0e-3):
        raise RuntimeError("the requested 15 mm sampling-disc radius is not active")

    payload = {
        "schema": "pmot.centered-surrogate-capture-loading-request.v1",
        "campaign": asdict(CAMPAIGN),
        "mot_configuration": asdict(mot),
        "cooling_beams": [
            {
                "label": beam.label,
                "power_w": beam.power_w,
                "one_over_e2_diameter_m": 2.0 * beam.beam_radius_m,
                "wavelength_m": beam.wavelength_m,
                "detuning_hz": beam.detuning_hz,
                "polarization": beam.circular_polarization,
            }
            for beam in cooling
        ],
        "repump_beams": [
            {
                "label": beam.label,
                "power_w": beam.power_w,
                "one_over_e2_diameter_m": 2.0 * beam.beam_radius_m,
                "wavelength_m": beam.wavelength_m,
                "detuning_hz": beam.detuning_hz,
                "polarization": beam.circular_polarization,
            }
            for beam in repump
        ],
        "effective_field_configuration": (
            DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.metadata()
        ),
        "method": (
            "direct velocity mask; full-sphere direction discs; uniform-area "
            "points; direction discs are independent Student-t clusters"
        ),
        "scientific_status": (
            "preliminary stretched-transition-equivalent surrogate-field "
            "diagnostic; not a state-resolved pMOT prediction"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "requested_configuration.json"
    encoded = json.dumps(payload, indent=2)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_speeds = prior.get("campaign", {}).pop("speeds_m_per_s", None)
        current_for_comparison = json.loads(encoded)
        current_speeds = current_for_comparison["campaign"].pop("speeds_m_per_s")
        speed_extension = (
            prior == current_for_comparison
            and prior_speeds is not None
            and set(prior_speeds).issubset(current_speeds)
        )
        if not speed_extension and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("saved preflight configuration does not match this run")
    path.write_text(encoded, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = (
        root
        / "outputs"
        / "diagnostics"
        / "pmot"
        / "MOT testing with a strange defined magnetic field"
        / RUN_NAME
    )
    _write_preflight_metadata(output_dir)
    run_campaign(output_dir, CAMPAIGN)
    run_representative_audits(output_dir, CAMPAIGN)


if __name__ == "__main__":
    main()

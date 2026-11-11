"""Reuse the independent 24-state force and core-stability checks at new powers."""

import os
from pathlib import Path

os.environ["NUMBA_CACHE_DIR"] = str(
    Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / "strange-gradient-extension-validation" / f"process_{os.getpid()}")

import extend_mot_strange_gradient_loading as campaign
import validate_mot_strange_loading_scans as validation
from pmot.mot_strange.loading_scan_inputs import current_symmetric_surrogate


def main():
    document = campaign.read(campaign.OUTPUT / "manifest.json")
    points = tuple(campaign.physical_point(row) for row in document["points"])
    validation.scan_points = lambda: (points, {})
    saved = {}

    def save(name, value):
        saved[name] = value
        campaign.write_json(campaign.OUTPUT / "validation" / name, value)

    validation.write_json = save
    geometry = current_symmetric_surrogate()
    local = validation.validate_local(geometry)
    validation.validate_core_stability(geometry)
    stability = saved["gravity_core_stability.json"]
    if len(stability) != 20 or any(not any(root["stable"] for root in row["axial_core_equilibria"])
                                   for row in stability):
        raise ValueError("An extension setting has no validated stable core equilibrium")
    campaign.write_json(campaign.OUTPUT / "physical_validation.json", {
        "passed": True, "independent_local_force_checks": local,
        "core_stability_checks": stability,
        "scope": "Original public 24-state reference and compiled force at the extension powers.",
    })
    print(f"Passed {len(local)} local checks and {len(stability)} gravity-on core stability checks", flush=True)


if __name__ == "__main__":
    main()

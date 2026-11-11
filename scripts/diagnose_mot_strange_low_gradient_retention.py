"""Check low-gradient core equilibria and matched zero-field trajectory controls."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import root

from pmot.mot_strange import accelerated_surrogate as solver
from pmot.mot_strange import loading_scan_inputs as inputs
from pmot.mot_strange.loading_scan_retention import observe, agreed_category
from run_mot_strange_loading_pilot import ROOT, write_json


OUTPUT = ROOT / "extension_40G_20261111" / "retention_diagnosis.json"


def jacobian(center, data, scale):
    state = np.r_[center, np.zeros(3)]
    steps = np.array([1e-7]*3 + [1e-4]*3) / scale

    def derivative(z):
        return np.r_[z[3:], solver.acceleration(z[:3], z[3:], np.array([0., 0., 1.]), data)]

    return np.column_stack([(derivative(state+d)-derivative(state-d))/(2*h)
                           for d, h in zip(np.diag(steps), steps, strict=True)])


def observation(row):
    return {
        "category": int(row[0]), "actual_duration_s": float(row[1]),
        "endpoint_mm": (row[2]*1e3).tolist(), "end_velocity_m_s": row[3].tolist(),
        "first_core_s": float(row[4]), "max_final_radius_mm": float(row[7]*1e3),
    }


def main():
    previous = json.loads((ROOT / "revisions/gradient_capture_diagnosis_20261109/matched_dense_masks.json").read_bytes())
    results = {}
    for geometry, scan in (("isotropic", "trap_power_isotropic"), ("axial", "trap_power_axial")):
        point = inputs.ScanPoint(geometry, 1.)
        field = inputs.field_config(point, inputs.current_symmetric_surrogate(), isotropic_axial_helicity=-1)
        data = inputs.prepare(point, field)
        equilibrium = root(lambda r: solver.acceleration(r, np.zeros(3), np.array([0.,0.,1.]), data),
                           [0.,0.,-5e-5], tol=1e-10)
        center = equilibrium.x
        residual = float(np.linalg.norm(solver.acceleration(center, np.zeros(3), np.array([0.,0.,1.]), data)))
        spectra = [np.linalg.eigvals(jacobian(center, data, scale)) for scale in (1, 2)]
        if residual > 1e-5 or any(np.max(v.real) >= 0 for v in spectra):
            raise ValueError("Low-gradient core stability failed")
        returns = []
        for component in range(6):
            for sign in (-1, 1):
                r, v = center.copy(), np.zeros(3)
                (r if component < 3 else v)[component % 3] += sign * (.00025 if component < 3 else .05)
                pair = [observe(r, v, data, .5, .1, refinement) for refinement in (1., 2.)]
                distance = [float(np.linalg.norm(row[2]-center)) for row in pair]
                returns.append({"component": component, "sign": sign, "final_distance_m": distance,
                                "agreed_category": agreed_category(*pair)})
        example = previous["diagnostics"][scan]["examples"][0]
        r = np.asarray(example["position_mm"])*1e-3
        v = np.asarray(example["direction"])*example["speed_m_s"]
        controls = {}
        for power, name, duration in ((1., "original_low_gradient", 2.),
                                      (1., "longer_low_gradient", 4.),
                                      (0., "zero_surrogate_field", 2.),
                                      (5., "intermediate_gradient", 2.)):
            control_point = replace(point, trap_power_mw=power)
            control_field = inputs.field_config(control_point, inputs.current_symmetric_surrogate(),
                                                isotropic_axial_helicity=-1)
            control = list(data)
            control[11] = data[11].copy()
            control[11][5:8] = [control_field.x_path_power_w, control_field.y_path_power_w, control_field.z_path_power_w]
            pair = [observe(r, v, tuple(control), duration, .1, refinement) for refinement in (1.,2.)]
            if agreed_category(*pair) is None:
                raise ValueError("Matched trajectory control did not converge")
            controls[name] = [observation(row) for row in pair]
        results[geometry] = {
            "gradient_xyz_g_cm": np.diag(inputs.field_axes(field)["origin_surrogate_field_jacobian_g_per_cm"]).tolist(),
            "core_equilibrium_mm": (center*1e3).tolist(), "gravity_on_residual_m_s2": residual,
            "spectral_abscissae_s_inv": [float(max(v.real)) for v in spectra],
            "perturbed_returns": returns,
            "matched_initial_position_mm": example["position_mm"],
            "matched_initial_velocity_m_s": v.tolist(), "matched_controls": controls,
        }
        write_json(OUTPUT, {"status": "diagnostic_in_progress", "results": results})
        print(geometry, "core equilibrium and matched controls completed", flush=True)
    write_json(OUTPUT, {
        "status": "passed", "scope": "Representative diagnostics, original production rates unchanged.",
        "core_rule": "5 ms residence or two entries, plus continuous final 100 ms inside 2 mm at 2 s.",
        "lowest_actual_gradient_g_cm": abs(results["isotropic"]["gradient_xyz_g_cm"][2]),
        "zero_gradient_in_original_scan": False, "results": results,
    })


if __name__ == "__main__":
    main()

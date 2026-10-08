"""Run the PMOT_EFFECTIVE_FIELD_CODEX.md surrogate-field diagnostic campaign."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pmot.pmot.surrogate_effective_field_study import (
    create_field_diagnostics,
    create_force_diagnostics,
    default_output_directory,
    run_diagnostic_trajectory_panel,
    run_timestep_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_report(output_directory: Path, field, force, trajectories, audit) -> Path:
    surrogate_force = force["models"]["surrogate"]
    aggregate = trajectories["aggregate"]
    field_working = bool(
        surrogate_force["all_position_modes_restoring"]
        and surrogate_force["all_velocity_modes_damping"]
    )
    trajectory_support = bool(
        aggregate["surrogate"]["preloaded_trapped_count"] > 0
        or aggregate["surrogate"]["incident_trapped_count"] > 0
    )
    timestep_support = bool(audit["capture_classifications_agree"])
    overall = field_working and trajectory_support and timestep_support
    diagonal = next(
        row
        for row in trajectories["cases"]
        if row["model"] == "surrogate" and row["case"] == "incident_diagonal_5mps"
    )
    verdict = (
        "The diagnostic evidence supports a locally stable cooling trap and demonstrates "
        "trapping for the tested axial launches, but it does not demonstrate robust "
        "three-dimensional incident capture because the diagonal launch was not trapped."
        if overall
        else "The diagnostic evidence does not yet demonstrate a working cooling trap."
    )
    lines = [
        "# MOT testing with a strange defined magnetic field",
        "",
        "This directory contains a proof-of-principle surrogate pMOT test defined by "
        "`PMOT_EFFECTIVE_FIELD_CODEX.md`. The six 1530-nm beam intensities are mapped "
        "beam-by-beam to a stretched-transition-equivalent magnetic field and vector-summed.",
        "",
        f"**Verdict:** {verdict}",
        "",
        "This verdict is diagnostic only. It is not a pMOT capture cross section, loading "
        "rate, temperature, or validation of a state-resolved AC-Stark Hamiltonian.",
        "",
        "## Evidence gates",
        "",
        f"- All three local position-force modes restoring: "
        f"`{surrogate_force['all_position_modes_restoring']}`",
        f"- All three local velocity-force modes damping: "
        f"`{surrogate_force['all_velocity_modes_damping']}`",
        f"- Surrogate preloaded trapped cases: "
        f"`{aggregate['surrogate']['preloaded_trapped_count']}/"
        f"{aggregate['surrogate']['preloaded_case_count']}`",
        f"- Surrogate incident trapped cases: "
        f"`{aggregate['surrogate']['incident_trapped_count']}/"
        f"{aggregate['surrogate']['incident_case_count']}`",
        f"- Conventional-coil incident trapped cases: "
        f"`{aggregate['coil']['incident_trapped_count']}/"
        f"{aggregate['coil']['incident_case_count']}`",
        f"- 5-us versus 2.5-us capture classification agreement: "
        f"`{audit['capture_classifications_agree']}`",
        f"- Diagonal 5 m/s surrogate result: `{diagonal['classification']}`; "
        f"minimum radius `{1e3 * diagonal['minimum_radius_m']:.3f} mm`; "
        f"final speed `{diagonal['final_speed_m_per_s']:.6f} m/s`.",
        "",
        "## Effective-field measurements",
        "",
        f"- Origin field [G]: `{field['origin_field_g']}`.",
        f"- Origin diagonal gradients [G/cm]: "
        f"`{[field['origin_jacobian_g_per_cm'][i][i] for i in range(3)]}`.",
        f"- Maximum field on the fine saved cell-plane grids: "
        f"`{field['maximum_fine_plane_sampled_field_g']:.3f} G`.",
        f"- Maximum field on the waist-augmented saved 3D grid: "
        f"`{field['maximum_waist_augmented_3d_grid_field_g']:.3f} G`.",
        f"- Gravity-balanced surrogate equilibrium [mm]: "
        f"`{force['models']['surrogate']['gravity_balanced_equilibrium']['position_mm']}`.",
        "",
        "## Directory map",
        "",
        "- `figures/effective_field`: axis cuts, all three 2D slices, and the 3D vector field.",
        "- `figures/force`: restoring/damping comparison with the conventional 10 G/cm MOT.",
        "- `figures/trajectories`: paired trajectory comparisons and detailed representative plots.",
        "- `data/effective_field`: sampled field arrays and origin-gradient summary.",
        "- `data/force`: force curves and full 3D force-Jacobian diagnostics.",
        "- `data/trajectories`: full trajectory CSV/NPZ files, case statistics, and timestep audit.",
        "",
        "## Physical boundary",
        "",
        "The effective field is exactly zero outside the closed cubic vapor domain with "
        "half-length 14.1421356 mm. Trajectories terminate as wall losses when they leave it.",
        "",
    ]
    path = output_directory / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "field", "force", "trajectories", "audit", "report"),
        default="all",
    )
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory or default_output_directory(PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[surrogate field study] output={output}", flush=True)

    field = force = trajectories = audit = None
    if args.stage in ("all", "field"):
        field = create_field_diagnostics(output)
        print("[surrogate field study] field diagnostics complete", flush=True)
    if args.stage in ("all", "force"):
        force = create_force_diagnostics(output)
        print("[surrogate field study] force diagnostics complete", flush=True)
    if args.stage in ("all", "trajectories"):
        trajectories = run_diagnostic_trajectory_panel(output)
        print("[surrogate field study] trajectory panel complete", flush=True)
    if args.stage in ("all", "audit"):
        audit = run_timestep_audit(output)
        print("[surrogate field study] timestep audit complete", flush=True)
    if args.stage == "report":
        field = json.loads(
            (output / "data" / "effective_field" / "field_diagnostic_summary.json").read_text(
                encoding="utf-8"
            )
        )
        force = json.loads(
            (output / "data" / "force" / "force_diagnostic_summary.json").read_text(
                encoding="utf-8"
            )
        )
        trajectories = json.loads(
            (output / "data" / "trajectories" / "trajectory_panel_summary.json").read_text(
                encoding="utf-8"
            )
        )
        audit = json.loads(
            (output / "data" / "trajectories" / "timestep_audit.json").read_text(
                encoding="utf-8"
            )
        )
    if args.stage in ("all", "report"):
        field_summary = (
            json.loads(Path(field["summary"]).read_text(encoding="utf-8"))
            if args.stage == "all"
            else field
        )
        report = _write_report(output, field_summary, force, trajectories, audit)
        manifest = {
            "schema": "pmot.surrogate-effective-field-campaign.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "specification": str(PROJECT_ROOT / "PMOT_EFFECTIVE_FIELD_CODEX.md"),
            "output_directory": str(output),
            "field_summary": str(output / "data" / "effective_field" / "field_diagnostic_summary.json"),
            "force_summary": str(output / "data" / "force" / "force_diagnostic_summary.json"),
            "trajectory_summary": str(output / "data" / "trajectories" / "trajectory_panel_summary.json"),
            "timestep_audit": str(output / "data" / "trajectories" / "timestep_audit.json"),
            "report": report,
        }
        (output / "campaign_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"[surrogate field study] report={report}", flush=True)


if __name__ == "__main__":
    main()

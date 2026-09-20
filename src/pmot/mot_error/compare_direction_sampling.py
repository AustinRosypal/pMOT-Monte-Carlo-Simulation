"""Compare octant and full-sphere multilevel-MOT loading estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .configuration import multilevel_mot_paths


def compare_loading_summaries(octant_path: Path, full_sphere_path: Path, output_path: Path) -> dict[str, object]:
    octant = json.loads(octant_path.read_text(encoding="utf-8"))
    full = json.loads(full_sphere_path.read_text(encoding="utf-8"))
    octant_rate = float(octant["loading_rate_atoms_per_s"])
    full_rate = float(full["loading_rate_atoms_per_s"])
    octant_se = float(octant["loading_rate_monte_carlo_disc_cluster_standard_error_atoms_per_s"])
    full_se = float(full["loading_rate_monte_carlo_disc_cluster_standard_error_atoms_per_s"])
    difference = full_rate - octant_rate
    combined_se = float(np.hypot(octant_se, full_se))
    payload = {
        "octant_summary": str(octant_path),
        "full_sphere_summary": str(full_sphere_path),
        "octant_loading_rate_atoms_per_s": octant_rate,
        "full_sphere_loading_rate_atoms_per_s": full_rate,
        "full_sphere_over_octant": full_rate / octant_rate,
        "relative_difference": difference / octant_rate,
        "difference_atoms_per_s": difference,
        "combined_disc_cluster_standard_error_atoms_per_s": combined_se,
        "difference_z_score": difference / combined_se if combined_se else float("inf"),
        "statistically_consistent_at_95_percent": abs(difference) <= 1.96 * combined_se,
        "interpretation": (
            "The estimates are treated as direction-clustered Monte Carlo samples. "
            "Agreement is assessed with the quadrature sum of their disc-cluster standard errors."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.errorbar(
        ["one octant", "full sphere"],
        [octant_rate / 1e6, full_rate / 1e6],
        yerr=[1.96 * octant_se / 1e6, 1.96 * full_se / 1e6],
        fmt="o", markersize=8, capsize=6, color="#0f766e",
    )
    axis.set(ylabel="loading rate [10⁶ atoms/s]", title="Direction-domain loading-rate comparison (95% MC intervals)")
    axis.grid(axis="y", alpha=0.25)
    plot_path = output_path.with_suffix(".png")
    figure.savefig(plot_path, dpi=190)
    plt.close(figure)
    payload["comparison_plot"] = str(plot_path)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    base = multilevel_mot_paths()["statistics"]
    parser = argparse.ArgumentParser(description="Compare octant and full-sphere loading estimates")
    parser.add_argument("--octant-summary", type=Path, default=base / "loading_rate_50_discs_25_points" / "full_mot_loading_summary.json")
    parser.add_argument("--full-sphere-summary", type=Path, default=base / "loading_rate_full_sphere_50_discs_50_points" / "full_mot_loading_summary.json")
    parser.add_argument("--output", type=Path, default=base / "loading_rate_direction_comparison.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    payload = compare_loading_summaries(args.octant_summary, args.full_sphere_summary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


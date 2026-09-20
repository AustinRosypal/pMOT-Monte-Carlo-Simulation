"""Analysis plots for fixed-speed multilevel disk-sampling CSV output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NUMERIC_COLUMNS = {
    "disc_index": int,
    "point_index": int,
    "theta_rad": float,
    "phi_rad": float,
    "theta_prime_rad": float,
    "s_m": float,
    "radial_distance_m": float,
    "launch_speed_m_per_s": float,
    "initial_state_index": int,
    "initial_f": int,
    "initial_m_f": int,
    "core_entries": int,
    "lifetime_s": float,
    "minimum_radius_m": float,
    "final_radius_m": float,
    "absorption_events": int,
    "spontaneous_emissions": int,
    "stimulated_emissions": int,
    "photons_before_dark": int,
    "dark_entry_time_s": float,
    "dark_parent_excited_f": int,
    "x0_m": float,
    "y0_m": float,
    "z0_m": float,
    "vx_hat": float,
    "vy_hat": float,
    "vz_hat": float,
    "xf_m": float,
    "yf_m": float,
    "zf_m": float,
    "vxf_m_per_s": float,
    "vyf_m_per_s": float,
    "vzf_m_per_s": float,
}

CLASSIFICATION_COLORS = {
    "trapped": "#15803d",
    "candidate_trapped_two_core_entries": "#15803d",
    "escaped": "#b91c1c",
    "untrapped_dark": "#111827",
    "untrapped_no_reentry": "#b91c1c",
    "indeterminate_event_cap": "#7c3aed",
    "indeterminate_duration": "#f97316",
}

CLASSIFICATION_LABELS = {
    "candidate_trapped_two_core_entries": "two-core-entry candidate",
    "escaped": "escaped outward",
    "untrapped_dark": "visited F=1",
    "indeterminate_event_cap": "hit event cap",
    "untrapped_no_reentry": "no core reentry",
}


def _convert_value(key: str, value: str) -> object:
    if value == "":
        return np.nan
    converter = NUMERIC_COLUMNS.get(key)
    if converter is None:
        return value
    return converter(float(value)) if converter is int else converter(value)


def load_sampling_csv(csv_path: Path) -> list[dict[str, object]]:
    """Load a multilevel fixed-speed sampling CSV into typed row dictionaries."""

    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = [{key: _convert_value(key, value) for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")
    return rows


def paired_analysis_figure_directory(csv_path: Path) -> Path:
    """Return the parallel figures directory for a statistics CSV."""

    statistics_directory = csv_path.resolve().parent
    parts = statistics_directory.parts
    if "outputs" in parts and "statistics" in parts:
        statistics_index = parts.index("statistics")
        prefix = Path(*parts[:statistics_index])
        suffix = Path(*parts[statistics_index + 1 :])
        return prefix / "figures" / suffix.with_name(f"{suffix.name}_analysis")
    return statistics_directory.with_name(f"{statistics_directory.name}_figures")


def _array(rows: list[dict[str, object]], key: str, dtype=float) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=dtype)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _display_label(classification: str) -> str:
    return CLASSIFICATION_LABELS.get(classification, classification.replace("_", " "))


def _classification_order(rows: list[dict[str, object]]) -> list[str]:
    priority = ["candidate_trapped_two_core_entries", "trapped", "escaped", "untrapped_no_reentry", "untrapped_dark", "indeterminate_event_cap", "indeterminate_duration"]
    present = set(_array(rows, "classification", dtype=str))
    return [label for label in priority if label in present] + sorted(present - set(priority))


def plot_classification_counts(rows: list[dict[str, object]], output_directory: Path) -> Path:
    counts = Counter(_array(rows, "classification", dtype=str))
    labels = _classification_order(rows)
    figure, axis = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    axis.bar(
        [_display_label(label) for label in labels],
        [counts[label] for label in labels],
        color=[CLASSIFICATION_COLORS.get(label, "#64748b") for label in labels],
    )
    axis.set(title="Fixed-speed launch outcomes", ylabel="Launches")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(alpha=0.22, axis="y")
    return _save(figure, output_directory / "classification_counts.png")


def plot_classification_by_disc(rows: list[dict[str, object]], output_directory: Path) -> Path:
    labels = _classification_order(rows)
    disc_indices = sorted(set(_array(rows, "disc_index", dtype=int)))
    counts_by_disc: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts_by_disc[int(row["disc_index"])][str(row["classification"])] += 1
    figure, axis = plt.subplots(figsize=(13.0, 5.6), constrained_layout=True)
    bottom = np.zeros(len(disc_indices), dtype=float)
    for label in labels:
        values = np.asarray([counts_by_disc[disc][label] for disc in disc_indices], dtype=float)
        totals = np.asarray([sum(counts_by_disc[disc].values()) for disc in disc_indices], dtype=float)
        fractions = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0.0)
        axis.bar(
            disc_indices,
            fractions,
            bottom=bottom,
            color=CLASSIFICATION_COLORS.get(label, "#64748b"),
            label=_display_label(label),
            width=0.88,
        )
        bottom += fractions
    axis.set(title="Outcome fraction by incident disk", xlabel="Disc index", ylabel="Fraction of points")
    axis.set_ylim(0.0, 1.0)
    axis.legend(ncols=min(4, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 1.14))
    axis.grid(alpha=0.18, axis="y")
    return _save(figure, output_directory / "classification_fraction_by_disc.png")


def plot_lifetime_histograms(rows: list[dict[str, object]], output_directory: Path) -> Path:
    lifetime_us = 1.0e6 * _array(rows, "lifetime_s")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    for axis, log_scale in zip(axes, (False, True), strict=True):
        axis.hist(lifetime_us, bins=60, color="#2563eb", edgecolor="#0f172a", linewidth=0.35)
        axis.set(title="Lifetime distribution" + ("; log y" if log_scale else ""), xlabel="Lifetime [µs]", ylabel="Launches")
        if log_scale:
            axis.set_yscale("log")
        axis.grid(alpha=0.22)
    return _save(figure, output_directory / "lifetime_histograms_linear_and_log.png")


def plot_dark_entry_histogram(rows: list[dict[str, object]], output_directory: Path) -> Path | None:
    dark_entry = _array(rows, "dark_entry_time_s")
    finite = np.isfinite(dark_entry)
    if not np.any(finite):
        return None
    dark_entry_us = 1.0e6 * dark_entry[finite]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    for axis, log_scale in zip(axes, (False, True), strict=True):
        axis.hist(dark_entry_us, bins=60, color="#111827", edgecolor="#cbd5e1", linewidth=0.35)
        axis.set(title="First F=1 visit time" + ("; log y" if log_scale else ""), xlabel="First F=1 visit [µs]", ylabel="Launches")
        if log_scale:
            axis.set_yscale("log")
        axis.grid(alpha=0.22)
    return _save(figure, output_directory / "first_f1_visit_time_histograms_linear_and_log.png")


def plot_impact_parameter_lifetime(rows: list[dict[str, object]], output_directory: Path) -> Path:
    labels = _classification_order(rows)
    s_mm = 1.0e3 * _array(rows, "s_m")
    lifetime_us = 1.0e6 * _array(rows, "lifetime_s")
    classifications = _array(rows, "classification", dtype=str)
    figure, axis = plt.subplots(figsize=(8.8, 5.7), constrained_layout=True)
    for label in labels:
        mask = classifications == label
        axis.scatter(
            s_mm[mask],
            lifetime_us[mask],
            s=18,
            alpha=0.72,
            color=CLASSIFICATION_COLORS.get(label, "#64748b"),
            label=_display_label(label),
        )
    axis.set(title="Lifetime versus impact parameter", xlabel="Impact parameter s [mm]", ylabel="Lifetime [µs]")
    axis.legend(loc="best")
    axis.grid(alpha=0.22)
    return _save(figure, output_directory / "lifetime_vs_impact_parameter.png")


def plot_minimum_radius_by_impact_parameter(rows: list[dict[str, object]], output_directory: Path) -> Path:
    labels = _classification_order(rows)
    s_mm = 1.0e3 * _array(rows, "s_m")
    rmin_mm = 1.0e3 * _array(rows, "minimum_radius_m")
    classifications = _array(rows, "classification", dtype=str)
    figure, axis = plt.subplots(figsize=(8.8, 5.7), constrained_layout=True)
    for label in labels:
        mask = classifications == label
        axis.scatter(
            s_mm[mask],
            rmin_mm[mask],
            s=18,
            alpha=0.72,
            color=CLASSIFICATION_COLORS.get(label, "#64748b"),
            label=_display_label(label),
        )
    axis.axhline(2.0, color="#15803d", linestyle="--", linewidth=1.0, label="2 mm core")
    axis.set(title="Closest approach versus impact parameter", xlabel="Impact parameter s [mm]", ylabel="Minimum radius [mm]")
    axis.legend(loc="best")
    axis.grid(alpha=0.22)
    return _save(figure, output_directory / "minimum_radius_vs_impact_parameter.png")


def plot_disc_direction_map(rows: list[dict[str, object]], output_directory: Path) -> Path:
    disc_indices = sorted(set(_array(rows, "disc_index", dtype=int)))
    theta, phi, f1_fraction, event_cap_fraction, minimum_radius = [], [], [], [], []
    for disc in disc_indices:
        disc_rows = [row for row in rows if int(row["disc_index"]) == disc]
        classifications = [str(row["classification"]) for row in disc_rows]
        theta.append(float(disc_rows[0]["theta_rad"]))
        phi.append(float(disc_rows[0]["phi_rad"]))
        f1_fraction.append(classifications.count("untrapped_dark") / len(disc_rows))
        event_cap_fraction.append(classifications.count("indeterminate_event_cap") / len(disc_rows))
        minimum_radius.append(1.0e3 * min(float(row["minimum_radius_m"]) for row in disc_rows))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    panels = (
        (f1_fraction, "Fraction with first F=1 visit", "viridis"),
        (event_cap_fraction, "Fraction hitting event cap", "magma"),
        (minimum_radius, "Best minimum radius [mm]", "cividis_r"),
    )
    for axis, (values, title, cmap) in zip(axes, panels, strict=True):
        scatter = axis.scatter(phi, theta, c=values, s=72, cmap=cmap, edgecolor="#0f172a", linewidth=0.35)
        figure.colorbar(scatter, ax=axis)
        axis.set(title=title, xlabel=r"$\phi$ [rad]", ylabel=r"$\theta$ [rad]")
        axis.grid(alpha=0.2)
    return _save(figure, output_directory / "incident_direction_octant_maps.png")


def plot_event_counts(rows: list[dict[str, object]], output_directory: Path) -> Path:
    lifetime_us = 1.0e6 * _array(rows, "lifetime_s")
    absorptions = _array(rows, "absorption_events")
    spontaneous = _array(rows, "spontaneous_emissions")
    stimulated = _array(rows, "stimulated_emissions")
    figure, axis = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    axis.scatter(lifetime_us, absorptions, s=14, alpha=0.55, color="#2563eb", label="absorption")
    axis.scatter(lifetime_us, spontaneous, s=14, alpha=0.55, color="#0f766e", label="spontaneous")
    axis.scatter(lifetime_us, stimulated, s=14, alpha=0.55, color="#c2410c", label="stimulated")
    axis.set(title="Optical event counts versus lifetime", xlabel="Lifetime [µs]", ylabel="Events")
    axis.legend()
    axis.grid(alpha=0.22)
    return _save(figure, output_directory / "event_counts_vs_lifetime.png")


def plot_final_vs_minimum_radius(rows: list[dict[str, object]], output_directory: Path) -> Path:
    labels = _classification_order(rows)
    final_mm = 1.0e3 * _array(rows, "final_radius_m")
    minimum_mm = 1.0e3 * _array(rows, "minimum_radius_m")
    classifications = _array(rows, "classification", dtype=str)
    figure, axis = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    for label in labels:
        mask = classifications == label
        axis.scatter(
            minimum_mm[mask],
            final_mm[mask],
            s=18,
            alpha=0.72,
            color=CLASSIFICATION_COLORS.get(label, "#64748b"),
            label=_display_label(label),
        )
    axis.axvline(2.0, color="#15803d", linestyle="--", linewidth=1.0)
    axis.axhline(2.0, color="#15803d", linestyle="--", linewidth=1.0)
    axis.set(title="Final radius versus closest approach", xlabel="Minimum radius [mm]", ylabel="Final radius [mm]")
    axis.legend(loc="best")
    axis.grid(alpha=0.22)
    return _save(figure, output_directory / "final_radius_vs_minimum_radius.png")


def summarize_rows(rows: list[dict[str, object]], csv_path: Path, output_files: list[Path]) -> dict[str, object]:
    classifications = _array(rows, "classification", dtype=str)
    lifetimes = _array(rows, "lifetime_s")
    dark_entry = _array(rows, "dark_entry_time_s")
    minimum_radius = _array(rows, "minimum_radius_m")
    event_cap_mask = classifications == "indeterminate_event_cap"
    f1_mask = np.isfinite(dark_entry)
    return {
        "source_csv": str(csv_path),
        "sample_count": len(rows),
        "disc_count": int(len(set(_array(rows, "disc_index", dtype=int)))),
        "points_per_disc_min": int(min(Counter(_array(rows, "disc_index", dtype=int)).values())),
        "points_per_disc_max": int(max(Counter(_array(rows, "disc_index", dtype=int)).values())),
        "classification_counts": dict(Counter(classifications)),
        "event_cap_fraction": float(np.mean(event_cap_mask)),
        "first_f1_visit_fraction": float(np.mean(f1_mask)),
        "mean_lifetime_us": float(1.0e6 * np.mean(lifetimes)),
        "median_lifetime_us": float(1.0e6 * np.median(lifetimes)),
        "mean_first_f1_visit_time_us": float(1.0e6 * np.nanmean(dark_entry)) if np.any(f1_mask) else None,
        "median_first_f1_visit_time_us": float(1.0e6 * np.nanmedian(dark_entry)) if np.any(f1_mask) else None,
        "minimum_radius_best_mm": float(1.0e3 * np.min(minimum_radius)),
        "minimum_radius_median_mm": float(1.0e3 * np.median(minimum_radius)),
        "outputs": [str(path) for path in output_files],
        "classification_note": "For repump-on data, untrapped_dark means the trajectory visited F=1 at least once; F=1 is no longer terminal when repumper_enabled=True.",
    }


def run_sampling_csv_analysis(
    csv_path: Path,
    output_directory: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, object]:
    """Generate fixed-speed multilevel sampling analysis plots from a CSV."""

    rows = load_sampling_csv(csv_path)
    figures = output_directory or paired_analysis_figure_directory(csv_path)
    output_files: list[Path] = [
        plot_classification_counts(rows, figures),
        plot_classification_by_disc(rows, figures),
        plot_lifetime_histograms(rows, figures),
        plot_impact_parameter_lifetime(rows, figures),
        plot_minimum_radius_by_impact_parameter(rows, figures),
        plot_disc_direction_map(rows, figures),
        plot_event_counts(rows, figures),
        plot_final_vs_minimum_radius(rows, figures),
    ]
    dark_entry_plot = plot_dark_entry_histogram(rows, figures)
    if dark_entry_plot is not None:
        output_files.append(dark_entry_plot)

    summary = summarize_rows(rows, csv_path, output_files)
    target_summary = summary_path or csv_path.parent / "multilevel_sampling_analysis_summary.json"
    target_summary.parent.mkdir(parents=True, exist_ok=True)
    target_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["outputs"].append(str(target_summary))
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze fixed-speed multilevel disk-sampling CSV output")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    summary = run_sampling_csv_analysis(args.csv_path, args.figures_dir, args.summary_json)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

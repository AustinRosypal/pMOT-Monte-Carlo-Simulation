"""Disk/impact-parameter sampling framework for multilevel trajectories.

This module deliberately stops at launch geometry and fixed-speed trajectory
classification. It is not a validated multilevel capture-velocity estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ..mot.magnetic_fields import default_anti_helmholtz_config
from ..mot_simple.sampling import DiscSample
from ..mot_simple.sampling import PointSample
from ..mot_simple.sampling import sample_disc_points
from ..mot_simple.sampling import sample_incident_disc
from ..mot_simple.sampling import scale
from .atomic_structure import AtomicStructure, build_atomic_structure
from .configuration import DarkStateBehavior, MultilevelMOTConfig
from .configuration import default_multilevel_mot_config, multilevel_mot_paths
from .screening import capture_classification, core_entry_count
from .screening import resampled_radii_m, trajectory_lifetime_s
from .simulation import build_multilevel_mot_beams, simulate_multilevel_trajectory
from .trajectory import MultilevelAtomState, sample_initial_internal_state


Vec3 = tuple[float, float, float]

SAMPLE_FIELDNAMES = [
    "disc_index",
    "point_index",
    "theta_rad",
    "phi_rad",
    "theta_prime_rad",
    "s_m",
    "radial_distance_m",
    "launch_speed_m_per_s",
    "initial_state_index",
    "initial_f",
    "initial_m_f",
    "classification",
    "core_entries",
    "lifetime_s",
    "minimum_radius_m",
    "final_radius_m",
    "termination",
    "absorption_events",
    "spontaneous_emissions",
    "stimulated_emissions",
    "photons_before_dark",
    "dark_entry_time_s",
    "dark_parent_excited_f",
    "x0_m",
    "y0_m",
    "z0_m",
    "vx_hat",
    "vy_hat",
    "vz_hat",
    "xf_m",
    "yf_m",
    "zf_m",
    "vxf_m_per_s",
    "vyf_m_per_s",
    "vzf_m_per_s",
]


@dataclass(frozen=True, slots=True)
class MultilevelSamplingConfig:
    """Configuration for pre-statistical multilevel launch sampling."""

    radial_distance_m: float = 15.0e-3
    disc_radius_m: float = 12.0e-3
    launch_speed_m_per_s: float = 8.0
    disc_count: int = 4
    points_per_disc: int = 8
    duration_s: float = 5.0e-3
    max_events: int = 5_000
    include_center_point: bool = True
    seed: int = 0
    save_every: int = 25


@dataclass(frozen=True, slots=True)
class MultilevelTrajectorySample:
    """One fixed-speed multilevel trajectory launched from an incident disk."""

    disc_index: int
    point_index: int
    theta_rad: float
    phi_rad: float
    theta_prime_rad: float
    s_m: float
    radial_distance_m: float
    initial_position_m: Vec3
    incident_unit_vector: Vec3
    launch_speed_m_per_s: float
    initial_state_index: int
    initial_f: int
    initial_m_f: int
    classification: str
    core_entries: int
    lifetime_s: float
    minimum_radius_m: float
    final_radius_m: float
    final_position_m: Vec3
    final_velocity_m_per_s: Vec3
    termination: str
    absorption_events: int
    spontaneous_emissions: int
    stimulated_emissions: int
    photons_before_dark: int
    dark_entry_time_s: float | None
    dark_parent_excited_f: int | None


def generate_launch_samples(search: MultilevelSamplingConfig, rng: np.random.Generator) -> list[tuple[DiscSample, list[PointSample]]]:
    """Generate incident disks and uniformly area-sampled launch points."""

    launches: list[tuple[DiscSample, list[PointSample]]] = []
    for disc_index in range(search.disc_count):
        disc = sample_incident_disc(disc_index, search.radial_distance_m, rng)
        points = sample_disc_points(
            disc,
            search.points_per_disc,
            search.disc_radius_m,
            search.include_center_point,
            rng,
        )
        launches.append((disc, points))
    return launches


def simulate_launch_sample(
    point: PointSample,
    sample_index: int,
    structure: AtomicStructure,
    beams,
    coil_config,
    config: MultilevelMOTConfig,
    search: MultilevelSamplingConfig,
) -> MultilevelTrajectorySample:
    """Run and classify one fixed-speed multilevel launch."""

    rng = np.random.default_rng(search.seed + 100_000 + sample_index)
    initial_state_index = sample_initial_internal_state(structure, config.initialization_mode, rng)
    initial_internal = structure.states[initial_state_index]
    initial = MultilevelAtomState(
        position_m=point.initial_position_m,
        velocity_m_per_s=scale(search.launch_speed_m_per_s, point.incident_unit_vector),
        internal_state_index=initial_state_index,
    )
    record = simulate_multilevel_trajectory(
        initial,
        search.duration_s,
        coil_config,
        beams=beams,
        structure=structure,
        config=config,
        seed=search.seed + sample_index,
        max_events=search.max_events,
    )
    radii = resampled_radii_m(record)
    counters = asdict(record.counters)
    return MultilevelTrajectorySample(
        disc_index=point.disc_index,
        point_index=point.point_index,
        theta_rad=point.theta_rad,
        phi_rad=point.phi_rad,
        theta_prime_rad=point.theta_prime_rad,
        s_m=point.s_m,
        radial_distance_m=point.radial_distance_m,
        initial_position_m=point.initial_position_m,
        incident_unit_vector=point.incident_unit_vector,
        launch_speed_m_per_s=search.launch_speed_m_per_s,
        initial_state_index=initial_state_index,
        initial_f=initial_internal.f,
        initial_m_f=initial_internal.m_f,
        classification=capture_classification(record),
        core_entries=core_entry_count(record),
        lifetime_s=trajectory_lifetime_s(record),
        minimum_radius_m=float(np.min(radii)),
        final_radius_m=float(np.linalg.norm(record.positions_m[-1])),
        final_position_m=record.positions_m[-1],
        final_velocity_m_per_s=record.velocities_m_per_s[-1],
        termination=record.termination_reason,
        absorption_events=counters["absorption_events"],
        spontaneous_emissions=counters["spontaneous_emissions"],
        stimulated_emissions=counters["stimulated_emissions"],
        photons_before_dark=counters["photons_before_dark"],
        dark_entry_time_s=counters["dark_entry_time_s"],
        dark_parent_excited_f=counters["dark_parent_excited_f"],
    )


def _sample_to_row(sample: MultilevelTrajectorySample) -> dict[str, object]:
    row = asdict(sample)
    row.update(
        {
            "x0_m": sample.initial_position_m[0],
            "y0_m": sample.initial_position_m[1],
            "z0_m": sample.initial_position_m[2],
            "vx_hat": sample.incident_unit_vector[0],
            "vy_hat": sample.incident_unit_vector[1],
            "vz_hat": sample.incident_unit_vector[2],
            "xf_m": sample.final_position_m[0],
            "yf_m": sample.final_position_m[1],
            "zf_m": sample.final_position_m[2],
            "vxf_m_per_s": sample.final_velocity_m_per_s[0],
            "vyf_m_per_s": sample.final_velocity_m_per_s[1],
            "vzf_m_per_s": sample.final_velocity_m_per_s[2],
        }
    )
    row.pop("initial_position_m")
    row.pop("incident_unit_vector")
    row.pop("final_position_m")
    row.pop("final_velocity_m_per_s")
    return row


def summarize_samples(samples: list[MultilevelTrajectorySample], search: MultilevelSamplingConfig) -> dict[str, object]:
    """Return compact metadata for a fixed-speed sampling run."""

    classifications = [sample.classification for sample in samples]
    lifetimes = np.asarray([sample.lifetime_s for sample in samples], dtype=float)
    return {
        "sample_count": len(samples),
        "disc_count": search.disc_count,
        "points_per_disc": search.points_per_disc,
        "launch_speed_m_per_s": search.launch_speed_m_per_s,
        "duration_s": search.duration_s,
        "classification_counts": {label: classifications.count(label) for label in sorted(set(classifications))},
        "mean_lifetime_s": float(np.mean(lifetimes)) if len(lifetimes) else None,
        "median_lifetime_s": float(np.median(lifetimes)) if len(lifetimes) else None,
        "sampling_config": asdict(search),
        "interpretation": "Fixed-speed multilevel disk sampling framework; not a validated capture-velocity estimate.",
    }


def save_sampling_results(samples: list[MultilevelTrajectorySample], search: MultilevelSamplingConfig, output_directory: Path) -> tuple[Path, Path]:
    """Save fixed-speed sampling rows and summary metadata."""

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "multilevel_launch_samples.csv"
    json_path = output_directory / "multilevel_launch_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDNAMES)
        writer.writeheader()
        for sample in samples:
            writer.writerow(_sample_to_row(sample))
    json_path.write_text(json.dumps(summarize_samples(samples, search), indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def run_multilevel_sampling(
    search_config: MultilevelSamplingConfig | None = None,
    output_directory: Path | None = None,
    coil_config=None,
    config: MultilevelMOTConfig | None = None,
) -> list[MultilevelTrajectorySample]:
    """Run the fixed-speed multilevel disk sampling framework."""

    search = search_config or MultilevelSamplingConfig()
    output = output_directory or multilevel_mot_paths()["statistics"] / "sampling"
    coil = coil_config or default_anti_helmholtz_config()
    cfg = replace(config or default_multilevel_mot_config(), dark_state_behavior=DarkStateBehavior.BALLISTIC)
    structure = build_atomic_structure()
    beams = build_multilevel_mot_beams(config=cfg)
    rng = np.random.default_rng(search.seed)
    samples: list[MultilevelTrajectorySample] = []
    sample_index = 0
    total_samples = search.disc_count * search.points_per_disc
    for disc, points in generate_launch_samples(search, rng):
        for point in points:
            samples.append(simulate_launch_sample(point, sample_index, structure, beams, coil, cfg, search))
            sample_index += 1
            if sample_index % max(1, search.save_every) == 0:
                save_sampling_results(samples, search, output)
                print(f"[multilevel sampling] saved partial results at {sample_index}/{total_samples} samples", flush=True)
        print(f"[multilevel sampling] completed disc {disc.disc_index + 1}/{search.disc_count}", flush=True)
    save_sampling_results(samples, search, output)
    return samples


def build_argument_parser() -> argparse.ArgumentParser:
    defaults = MultilevelSamplingConfig()
    parser = argparse.ArgumentParser(description="Fixed-speed multilevel MOT disk launch sampler")
    parser.add_argument("--disc-count", type=int, default=defaults.disc_count)
    parser.add_argument("--points-per-disc", type=int, default=defaults.points_per_disc)
    parser.add_argument("--radial-distance-mm", type=float, default=1e3 * defaults.radial_distance_m)
    parser.add_argument("--disc-radius-mm", type=float, default=1e3 * defaults.disc_radius_m)
    parser.add_argument("--launch-speed", type=float, default=defaults.launch_speed_m_per_s)
    parser.add_argument("--duration-ms", type=float, default=1e3 * defaults.duration_s)
    parser.add_argument("--max-events", type=int, default=defaults.max_events)
    parser.add_argument("--include-center-point", action="store_true", default=defaults.include_center_point)
    parser.add_argument("--no-include-center-point", dest="include_center_point", action="store_false")
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def search_config_from_args(args: argparse.Namespace) -> MultilevelSamplingConfig:
    return MultilevelSamplingConfig(
        radial_distance_m=1e-3 * args.radial_distance_mm,
        disc_radius_m=1e-3 * args.disc_radius_mm,
        launch_speed_m_per_s=args.launch_speed,
        disc_count=args.disc_count,
        points_per_disc=args.points_per_disc,
        duration_s=1e-3 * args.duration_ms,
        max_events=args.max_events,
        include_center_point=args.include_center_point,
        seed=args.seed,
        save_every=args.save_every,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    search = search_config_from_args(args)
    samples = run_multilevel_sampling(search_config=search, output_directory=args.output_dir)
    print(json.dumps(summarize_samples(samples, search), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

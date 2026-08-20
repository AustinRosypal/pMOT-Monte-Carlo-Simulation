"""Command-line runner for the efficient full multilevel MOT model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from ..mot.magnetic_fields import default_anti_helmholtz_config
from .configuration import default_multilevel_mot_config, multilevel_mot_paths
from .rate_diagnostics import (
    plot_rate_equation_performance,
    plot_rate_equation_trajectory_3d,
    save_rate_equation_trajectory,
    summarize_rate_equation_trajectory,
)
from .rate_equations import (
    RateEquationAtomState,
    RateEquationTrajectoryConfig,
    build_rate_equation_model,
    simulate_rate_equation_trajectory,
)
from .simulation import build_multilevel_mot_beams


def run_rate_equation_atom(
    *,
    duration_s: float = 25.0e-3,
    time_step_s: float = 5.0e-6,
    initial_position_m=(1.0e-3, 0.0, 0.0),
    initial_velocity_m_per_s=(-1.0, 0.2, 0.0),
    seed: int = 20260819,
    include_diffusion: bool = True,
    output_directory: Path | None = None,
) -> dict[str, object]:
    """Run one atom and save numerical data plus 3D/performance figures."""

    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    beams = build_multilevel_mot_beams(config=config)
    numerical = RateEquationTrajectoryConfig(
        time_step_s=time_step_s,
        include_diffusion=include_diffusion,
        seed=seed,
        escape_radius_m=30.0e-3,
    )
    output = output_directory or multilevel_mot_paths()["trajectories"] / "rate_equation"
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"[rate-equation MOT] running trajectory 1/1: T={1e3*duration_s:.3f} ms, "
        f"dt={1e6*time_step_s:.3f} us, steps={int(round(duration_s/time_step_s)):,}",
        flush=True,
    )
    start = perf_counter()
    record = simulate_rate_equation_trajectory(
        RateEquationAtomState(tuple(initial_position_m), tuple(initial_velocity_m_per_s)),
        duration_s,
        default_anti_helmholtz_config(),
        beams=beams,
        model=model,
        config=config,
        trajectory_config=numerical,
    )
    wall_time_s = perf_counter() - start
    stem = output / "trajectory_000"
    summary = summarize_rate_equation_trajectory(record)
    files = save_rate_equation_trajectory(
        record,
        model,
        beams,
        stem,
        metadata={
            "duration_s": duration_s,
            "time_step_s": time_step_s,
            "initial_position_m": list(initial_position_m),
            "initial_velocity_m_per_s": list(initial_velocity_m_per_s),
            "seed": seed,
            "include_diffusion": include_diffusion,
            "multilevel_config": asdict(config),
            "summary": summary,
        },
    )
    files.extend([
        plot_rate_equation_trajectory_3d(record, beams, output / "trajectory_000_3d.png"),
        plot_rate_equation_trajectory_3d(
            record,
            beams,
            output / "trajectory_000_3d_zoom.png",
            title="Full multilevel MOT rate-equation trajectory — trap-region zoom",
            show_beams=False,
        ),
        plot_rate_equation_performance(record, model, output / "trajectory_000_performance.png"),
    ])
    result = {
        **summary,
        "wall_time_s": wall_time_s,
        "duration_s": duration_s,
        "time_step_s": time_step_s,
        "seed": seed,
        "include_diffusion": include_diffusion,
        "outputs": [str(path) for path in files],
    }
    summary_path = output / "trajectory_000_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["outputs"].append(str(summary_path))
    print(
        f"[rate-equation MOT] completed trajectory 1/1: "
        f"termination={record.termination_reason}, "
        f"elapsed={1e3*record.times_s[-1]:.3f} ms, wall={wall_time_s:.3f} s",
        flush=True,
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Efficient fixed-dt multilevel MOT trajectory")
    parser.add_argument("--duration-ms", type=float, default=25.0)
    parser.add_argument("--dt-us", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--no-diffusion", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_rate_equation_atom(
        duration_s=1e-3 * args.duration_ms,
        time_step_s=1e-6 * args.dt_us,
        seed=args.seed,
        include_diffusion=not args.no_diffusion,
        output_directory=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

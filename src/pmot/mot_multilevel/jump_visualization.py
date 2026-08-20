"""Isolated photon-jump visualization; not the main trajectory algorithm.

The expensive Gillespie engine is retained here for short pedagogical runs and
cross-checks against the efficient rate-equation model.
"""

from __future__ import annotations

from pathlib import Path

from ..mot.magnetic_fields import default_anti_helmholtz_config
from .atomic_structure import build_atomic_structure
from .configuration import MultilevelMOTConfig, default_multilevel_mot_config
from .screening import animate_hyperfine_walk
from .simulation import build_multilevel_mot_beams, simulate_multilevel_trajectory
from .trajectory import MultilevelAtomState


def create_hyperfine_jump_animation(
    path: Path,
    *,
    duration_s: float = 100.0e-6,
    seed: int = 20260819,
    initial_f: int = 2,
    initial_m_f: int = 0,
    config: MultilevelMOTConfig | None = None,
    max_events: int = 20_000,
) -> Path:
    """Run a deliberately short legacy jump trace and save its animation."""

    cfg = config or default_multilevel_mot_config()
    structure = build_atomic_structure()
    beams = build_multilevel_mot_beams(config=cfg)
    initial = MultilevelAtomState(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        structure.state_index("ground", initial_f, initial_m_f),
    )
    record = simulate_multilevel_trajectory(
        initial,
        duration_s,
        default_anti_helmholtz_config(),
        beams=beams,
        structure=structure,
        config=cfg,
        seed=seed,
        max_events=max_events,
    )
    return animate_hyperfine_walk(record, structure, path)


__all__ = ["create_hyperfine_jump_animation"]

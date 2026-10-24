"""Deterministic vector-only effective two-level pMOT diagnostic."""

from __future__ import annotations

from typing import Sequence

from ..mot_simple.configuration import SimpleMOTConfig
from ..mot_simple.simulation import SimpleMOTBeam
from ..mot_strange.trapping_beams import TrappingBeam
from ..pMOT_2Level.simulation import PMOT2LevelLightBundle
from ..pMOT_2Level.simulation import PMOT2LevelLocalObservable
from ..pMOT_2Level.simulation import PMOT2LevelTrajectoryRecord
from ..pMOT_2Level.simulation import build_pmot_2level_cooling_beams
from ..pMOT_2Level.simulation import build_pmot_2level_mot_light
from ..pMOT_2Level.simulation import build_pmot_2level_repump_beams
from ..pMOT_2Level.simulation import evaluate_pmot_2level_local_observable
from ..pMOT_2Level.simulation import rk4_step_pmot_2level
from ..pMOT_2Level.simulation import simulate_pmot_2level_trajectory
from ..state import AtomState
from .ac_stark import load_vector_only_polarizability_table


def build_pmot_vec_2level_mot_light(*args, **kwargs) -> PMOT2LevelLightBundle:
    """Build cooling beams and passive, resonance-tuned repump scaffolding."""

    return build_pmot_2level_mot_light(*args, **kwargs)


def build_pmot_vec_2level_cooling_beams(*args, **kwargs) -> list[SimpleMOTBeam]:
    return build_pmot_2level_cooling_beams(*args, **kwargs)


def build_pmot_vec_2level_repump_beams(*args, **kwargs):
    return build_pmot_2level_repump_beams(*args, **kwargs)


def evaluate_pmot_vec_2level_local_observable(
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    position_m,
    velocity_m_per_s,
    *,
    path_powers_w,
    previous_axis=(0.0, 0.0, 1.0),
    simple_config: SimpleMOTConfig | None = None,
) -> PMOT2LevelLocalObservable:
    return evaluate_pmot_2level_local_observable(
        cooling_beams,
        trapping_beams,
        position_m,
        velocity_m_per_s,
        path_powers_w=path_powers_w,
        previous_axis=previous_axis,
        simple_config=simple_config,
        polarizability_table=load_vector_only_polarizability_table(),
    )


def rk4_step_pmot_vec_2level(
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    atom_state: AtomState,
    time_step_s: float,
    *,
    path_powers_w,
    previous_axis=(0.0, 0.0, 1.0),
    simple_config: SimpleMOTConfig | None = None,
):
    return rk4_step_pmot_2level(
        cooling_beams,
        trapping_beams,
        atom_state,
        time_step_s,
        path_powers_w=path_powers_w,
        previous_axis=previous_axis,
        simple_config=simple_config,
        polarizability_table=load_vector_only_polarizability_table(),
    )


def simulate_pmot_vec_2level_trajectory(
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    initial_state: AtomState,
    duration_s: float,
    time_step_s: float,
    *,
    path_powers_w,
    initial_quantization_axis=(0.0, 0.0, 1.0),
    simple_config: SimpleMOTConfig | None = None,
) -> PMOT2LevelTrajectoryRecord:
    return simulate_pmot_2level_trajectory(
        cooling_beams,
        trapping_beams,
        initial_state,
        duration_s,
        time_step_s,
        path_powers_w=path_powers_w,
        initial_quantization_axis=initial_quantization_axis,
        simple_config=simple_config,
        polarizability_table=load_vector_only_polarizability_table(),
    )


__all__ = [
    "build_pmot_vec_2level_cooling_beams",
    "build_pmot_vec_2level_mot_light",
    "build_pmot_vec_2level_repump_beams",
    "evaluate_pmot_vec_2level_local_observable",
    "rk4_step_pmot_vec_2level",
    "simulate_pmot_vec_2level_trajectory",
]

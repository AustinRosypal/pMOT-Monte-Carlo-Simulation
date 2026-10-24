"""Vector-only differential AC-Stark closure for the effective two-level pMOT."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..mot_strange.polarizability import DifferentialPolarizabilityTable
from ..mot_strange.trapping_beams import TrappingBeam
from ..pMOT_2Level.ac_stark import PMOT2LevelStarkSample
from ..pMOT_2Level.ac_stark import evaluate_differential_ac_stark_shift
from ..pMOT_2Level.ac_stark import load_pmot_2level_polarizability_table


@lru_cache(maxsize=1)
def load_vector_only_polarizability_table() -> DifferentialPolarizabilityTable:
    """Return the Arora table with scalar and tensor columns set identically zero."""

    source = load_pmot_2level_polarizability_table()
    scalar = np.zeros_like(source.scalar_si)
    tensor = np.zeros_like(source.tensor_si)
    scalar.setflags(write=False)
    tensor.setflags(write=False)
    return DifferentialPolarizabilityTable(
        wavelengths_nm=source.wavelengths_nm,
        scalar_si=scalar,
        vector_si=source.vector_si,
        tensor_si=tensor,
        source_path=source.source_path,
    )


def evaluate_vector_only_ac_stark_shift(
    trapping_beams: tuple[TrappingBeam, ...] | list[TrappingBeam],
    position_m,
    velocity_m_per_s,
    *,
    path_powers_w,
    previous_axis=(0.0, 0.0, 1.0),
    relative_axis_zero_tolerance: float = 1.0e-12,
) -> PMOT2LevelStarkSample:
    """Evaluate the same geometry/axis prescription with only rank-1 shift."""

    return evaluate_differential_ac_stark_shift(
        trapping_beams,
        position_m,
        velocity_m_per_s,
        path_powers_w=path_powers_w,
        previous_axis=previous_axis,
        polarizability_table=load_vector_only_polarizability_table(),
        relative_axis_zero_tolerance=relative_axis_zero_tolerance,
    )


__all__ = [
    "evaluate_vector_only_ac_stark_shift",
    "load_vector_only_polarizability_table",
]

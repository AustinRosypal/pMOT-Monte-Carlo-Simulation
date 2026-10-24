"""Vector-only effective two-level pseudo-MOT diagnostic.

This branch shares the pMOT trapping geometry, Doppler-shifted polarizability
lookup, local quantization-axis prescription, six cooling beams, and
deterministic force machinery with :mod:`pmot.pMOT_2Level`.  It sets the scalar
and tensor differential polarizabilities identically to zero, leaving only the
vector AC-Stark term.

Six repump components are exposed at the same resonance used by the multilevel
MOT. They are passive metadata in this strict F=2 -> F'=3 model because it has
no F=1 population; treating them as cooling-transition scatterers would be
unphysical.
"""

from ..pMOT_2Level.trapping_geometry import build_pmot_2level_trapping_beams
from .ac_stark import (
    evaluate_vector_only_ac_stark_shift,
    load_vector_only_polarizability_table,
)
from .simulation import (
    build_pmot_vec_2level_cooling_beams,
    build_pmot_vec_2level_mot_light,
    build_pmot_vec_2level_repump_beams,
    evaluate_pmot_vec_2level_local_observable,
    rk4_step_pmot_vec_2level,
    simulate_pmot_vec_2level_trajectory,
)

build_pmot_vec_2level_trapping_beams = build_pmot_2level_trapping_beams

__all__ = [
    "build_pmot_vec_2level_cooling_beams",
    "build_pmot_vec_2level_mot_light",
    "build_pmot_vec_2level_repump_beams",
    "build_pmot_vec_2level_trapping_beams",
    "evaluate_pmot_vec_2level_local_observable",
    "evaluate_vector_only_ac_stark_shift",
    "load_vector_only_polarizability_table",
    "rk4_step_pmot_vec_2level",
    "simulate_pmot_vec_2level_trajectory",
]

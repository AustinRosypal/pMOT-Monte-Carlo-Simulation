"""Plotting-free ideal-magic vector-only pMOT observable.

Keeping this physics core independent of Matplotlib is important: interactive
notebooks can import it without a batch plotting module changing their backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..configuration import HBAR_J_S
from ..mot_multilevel.rate_equations import rate_equation_observable_from_local_environment
from .ac_stark import ProvisionalStarkConfig
from .ac_stark import build_physics_trapping_beams
from .ac_stark import provisional_transition_stark_shifts


BARE_AXIS = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class VectorOnlyContext:
    """Inputs needed by the vector-only observable, without plotting state."""

    apparatus: Any
    multilevel_config: Any
    model: Any
    polarizability_table: Any
    cooling_repump_beams: tuple[Any, ...]
    power_w_per_path: float


@dataclass(frozen=True, slots=True)
class VectorOnlyObservable:
    """One vector-only local environment and its inherited 780-nm response."""

    rate_equation: Any
    stark_diagnostic: Any
    applied_transition_shift_rad_per_s: np.ndarray


def decode_sigma_helicity_code(
    code: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Decode ``ix,iy,iz,rx,ry,rz`` symbols into sigma labels."""

    if len(code) != 6 or any(symbol not in "+-" for symbol in code):
        raise ValueError("helicity code must contain six '+'/'-' symbols")
    labels = tuple("sigma+" if symbol == "+" else "sigma-" for symbol in code)
    return labels[:3], labels[3:]


def prepare_vector_only_trapping_environment(
    context: VectorOnlyContext,
    code: str,
) -> tuple[ProvisionalStarkConfig, list[Any]]:
    """Resolve the six trapping components for one sigma-only code."""

    incident, retro = decode_sigma_helicity_code(code)
    stark_config = ProvisionalStarkConfig.uniform_power(
        context.power_w_per_path,
        incident_helicities_by_axis=incident,
        retro_helicities_by_axis=retro,
    )
    beams = build_physics_trapping_beams(
        context.apparatus.trapping_laser,
        stark_config,
    )
    return stark_config, beams


def vector_only_pmot_observable(
    context: VectorOnlyContext,
    code: str,
    position_m,
    velocity_m_per_s,
    *,
    previous_axis=BARE_AXIS,
    prepared: tuple[ProvisionalStarkConfig, list[Any]] | None = None,
) -> VectorOnlyObservable:
    """Apply only the vector Stark transition shift to fixed 780-nm beams."""

    stark_config, trapping_beams = (
        prepared
        if prepared is not None
        else prepare_vector_only_trapping_environment(context, code)
    )
    stark = provisional_transition_stark_shifts(
        context.model,
        trapping_beams,
        position_m,
        velocity_m_per_s,
        context.apparatus.trapping_laser,
        stark_config,
        previous_axis,
        polarizability_table=context.polarizability_table,
    )
    applied_shift = np.asarray(stark.vector_transition_energy_j, dtype=float) / HBAR_J_S
    rate = rate_equation_observable_from_local_environment(
        context.model,
        list(context.cooling_repump_beams),
        tuple(float(value) for value in position_m),
        tuple(float(value) for value in velocity_m_per_s),
        (0.0, 0.0, 0.0),
        stark.quantization_axis,
        context.multilevel_config,
        transition_resonance_shift_rad_per_s=applied_shift,
    )
    return VectorOnlyObservable(rate, stark, applied_shift)


def bare_780_observable(
    context: VectorOnlyContext,
    position_m,
    velocity_m_per_s,
):
    """Evaluate fixed cooling/repump light with no external or Stark shift."""

    return rate_equation_observable_from_local_environment(
        context.model,
        list(context.cooling_repump_beams),
        tuple(float(value) for value in position_m),
        tuple(float(value) for value in velocity_m_per_s),
        (0.0, 0.0, 0.0),
        BARE_AXIS,
        context.multilevel_config,
        transition_resonance_shift_rad_per_s=np.zeros(
            len(context.model.transition_ground),
            dtype=float,
        ),
    )


__all__ = [name for name in globals() if not name.startswith("_")]

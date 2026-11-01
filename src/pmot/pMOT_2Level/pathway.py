"""Named wavelength pathways for the effective two-level pMOT.

The two pathways deliberately share the same trapping-beam geometry, Stark
evaluator, cooling-force model, and trajectory integrator.  Only the selected
trapping wavelength and its documented polarizability condition differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from types import MappingProxyType
from typing import Mapping
from typing import Sequence

from ..mot_strange.trapping_beams import TrappingBeam
from ..mot_strange.trapping_beams import TrappingLaserConfig
from .trapping_geometry import DEFAULT_INCIDENT_HELICITIES_BY_AXIS
from .trapping_geometry import DEFAULT_RETRO_HELICITIES_BY_AXIS
from .trapping_geometry import build_pmot_2level_trapping_beams


SCALAR_TENSOR_CANCELLATION_WAVELENGTH_NM = 1529.268881
"""Single-beam stretched-reference scalar--tensor cancellation wavelength."""

SCALAR_ZERO_WAVELENGTH_NM = 1529.273195181438
"""Nearby smooth zero of the tabulated differential scalar polarizability."""


@dataclass(frozen=True, slots=True)
class PMOT2LevelPathway:
    """A fixed trapping-wavelength prescription for the shared pMOT engine."""

    key: str
    display_name: str
    trapping_wavelength_nm: float
    polarizability_condition: str
    geometry_note: str

    @property
    def trapping_wavelength_m(self) -> float:
        """Return the fixed pathway wavelength in metres."""

        return 1.0e-9 * self.trapping_wavelength_nm

    def build_laser_config(
        self,
        base_config: TrappingLaserConfig | None = None,
        **geometry_overrides,
    ) -> TrappingLaserConfig:
        """Return a geometry config pinned to this pathway's wavelength.

        ``geometry_overrides`` may alter waist positions, input diameter,
        focal length, retro power fraction, or other non-wavelength geometry.
        A wavelength override is rejected so that saved pathway names remain
        scientifically meaningful.
        """

        if "wavelength_m" in geometry_overrides:
            raise ValueError(
                "wavelength_m is fixed by the selected pMOT pathway; choose a "
                "different pathway instead"
            )
        base = base_config or TrappingLaserConfig()
        return replace(
            base,
            wavelength_m=self.trapping_wavelength_m,
            **geometry_overrides,
        )

    def build_trapping_beams(
        self,
        base_config: TrappingLaserConfig | None = None,
        *,
        incident_helicities_by_axis: Sequence[str] = (
            DEFAULT_INCIDENT_HELICITIES_BY_AXIS
        ),
        retro_helicities_by_axis: Sequence[str] = DEFAULT_RETRO_HELICITIES_BY_AXIS,
        **geometry_overrides,
    ) -> tuple[TrappingBeam, ...]:
        """Build the six shared-geometry components for this wavelength path."""

        laser_config = self.build_laser_config(base_config, **geometry_overrides)
        return build_pmot_2level_trapping_beams(
            laser_config,
            incident_helicities_by_axis=incident_helicities_by_axis,
            retro_helicities_by_axis=retro_helicities_by_axis,
        )


SCALAR_TENSOR_CANCELLATION_PATHWAY = PMOT2LevelPathway(
    key="scalar_tensor_cancellation",
    display_name="Single-beam scalar--tensor cancellation",
    trapping_wavelength_nm=SCALAR_TENSOR_CANCELLATION_WAVELENGTH_NM,
    polarizability_condition=(
        "For a stretched reference with the quantization axis parallel to one "
        "circular beam, the differential scalar and tensor energy shifts cancel."
    ),
    geometry_note=(
        "In the balanced three-axis core the tensor sum cancels geometrically, "
        "so a differential scalar residue remains."
    ),
)

SCALAR_ZERO_PATHWAY = PMOT2LevelPathway(
    key="scalar_zero",
    display_name="Differential-scalar zero",
    trapping_wavelength_nm=SCALAR_ZERO_WAVELENGTH_NM,
    polarizability_condition=(
        "The tabulated 5P3/2 minus 5S1/2 differential scalar polarizability "
        "vanishes while the vector and tensor polarizabilities remain nonzero."
    ),
    geometry_note=(
        "In the balanced three-axis core the tensor sum also cancels "
        "geometrically, leaving the locally linear vector shift as the leading "
        "nonzero differential shift near the origin."
    ),
)

PMOT_2LEVEL_PATHWAYS: Mapping[str, PMOT2LevelPathway] = MappingProxyType(
    {
        SCALAR_TENSOR_CANCELLATION_PATHWAY.key: (
            SCALAR_TENSOR_CANCELLATION_PATHWAY
        ),
        SCALAR_ZERO_PATHWAY.key: SCALAR_ZERO_PATHWAY,
    }
)


def get_pmot_2level_pathway(name: str) -> PMOT2LevelPathway:
    """Return one named pathway, raising with the valid keys when unknown."""

    try:
        return PMOT_2LEVEL_PATHWAYS[name]
    except KeyError as error:
        choices = ", ".join(PMOT_2LEVEL_PATHWAYS)
        raise ValueError(
            f"unknown pMOT_2Level pathway {name!r}; choose one of: {choices}"
        ) from error


__all__ = [
    "PMOT2LevelPathway",
    "PMOT_2LEVEL_PATHWAYS",
    "SCALAR_TENSOR_CANCELLATION_PATHWAY",
    "SCALAR_TENSOR_CANCELLATION_WAVELENGTH_NM",
    "SCALAR_ZERO_PATHWAY",
    "SCALAR_ZERO_WAVELENGTH_NM",
    "get_pmot_2level_pathway",
]

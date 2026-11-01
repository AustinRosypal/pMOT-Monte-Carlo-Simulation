"""pMOT pathway using the differential-scalar-zero wavelength.

This pathway fixes the trapping wavelength at 1529.273195181438 nm. It reuses
the shared :mod:`pmot.pMOT_2Level` Stark, force, and trajectory implementation.
"""

from ..pathway import SCALAR_ZERO_PATHWAY as PATHWAY

build_trapping_laser_config = PATHWAY.build_laser_config
build_trapping_beams = PATHWAY.build_trapping_beams

__all__ = ["PATHWAY", "build_trapping_beams", "build_trapping_laser_config"]

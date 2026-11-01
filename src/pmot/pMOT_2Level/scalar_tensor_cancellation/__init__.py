"""pMOT pathway using single-beam scalar--tensor cancellation.

This pathway fixes the trapping wavelength at 1529.268881 nm.  It reuses the
shared :mod:`pmot.pMOT_2Level` Stark, force, and trajectory implementation.
"""

from ..pathway import SCALAR_TENSOR_CANCELLATION_PATHWAY as PATHWAY

build_trapping_laser_config = PATHWAY.build_laser_config
build_trapping_beams = PATHWAY.build_trapping_beams

__all__ = ["PATHWAY", "build_trapping_beams", "build_trapping_laser_config"]

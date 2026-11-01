# Single-beam scalar--tensor-cancellation pathway

This pathway uses a trapping wavelength of **1529.268881 nm**. For one
circular trapping beam, a stretched reference transition, and a quantization
axis parallel to the beam propagation direction, the differential scalar and
tensor energy shifts cancel.

That cancellation is not global in the three-axis pMOT. At balanced equal
x/y/z path intensity, the three rank-2 tensor contributions cancel one another
geometrically, leaving the scalar contribution nonzero. This pathway is kept
as a separate, explicit simulation branch so that its behavior can be compared
without changing the shared Stark, force, or trajectory implementation.

```python
from pmot.pMOT_2Level.scalar_tensor_cancellation import (
    PATHWAY,
    build_trapping_beams,
)

trapping_beams = build_trapping_beams()
print(PATHWAY.trapping_wavelength_nm)
```

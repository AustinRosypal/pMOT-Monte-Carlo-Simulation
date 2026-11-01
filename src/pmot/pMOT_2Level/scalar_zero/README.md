# Differential-scalar-zero pathway

This pathway uses a trapping wavelength of **1529.273195181438 nm**, the nearby
smooth zero of the differential scalar polarizability in
`Arora_CCSD_Differential_Polarizabilities.csv`.

The vector and tensor polarizabilities remain nonzero. In the ideal balanced
three-axis geometry, the tensor sum vanishes at every point where the three
round-trip path intensities are equal. Near the symmetric origin, the signed
vector shift is locally odd and linear in displacement, while the tensor
residue caused by path-intensity anisotropy begins at higher order.

```python
from pmot.pMOT_2Level.scalar_zero import PATHWAY, build_trapping_beams

trapping_beams = build_trapping_beams()
print(PATHWAY.trapping_wavelength_nm)
```

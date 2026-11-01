# Effective two-level pMOT notebooks

`trajectory_simulation.ipynb` is the shared interactive trajectory front end.
The source package now provides two fixed, named trapping-wavelength pathways:

- `scalar_tensor_cancellation`: 1529.268881 nm;
- `scalar_zero`: 1529.273195181438 nm.

Both paths use the same beam geometry, Doppler-shifted polarizability lookup,
effective Stark closure, cooling-force equation, and RK4 integrator. Only the
trapping wavelength differs. Programmatic notebook cells should select a path
with:

```python
from pmot.pMOT_2Level import get_pmot_2level_pathway

PATHWAY = get_pmot_2level_pathway("scalar_zero")
trapping_beams = PATHWAY.build_trapping_beams()
```

The notebook's historical 1529.268881-nm value corresponds to
`scalar_tensor_cancellation`. When comparing saved trajectories, include the
pathway key and exact wavelength in the run metadata and output directory.

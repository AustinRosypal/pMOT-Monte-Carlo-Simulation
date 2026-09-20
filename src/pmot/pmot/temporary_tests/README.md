# Temporary pMOT tests

This package is an isolation boundary for exploratory pMOT calculations. It is
not imported by the production pMOT workflow.

`fixed_stretched_shift.py` evaluates the signed scalar, vector, and tensor
differential AC-Stark shift of only the fixed-basis stretched cycling
transition, `5S1/2 F=2,mF=+2 -> 5P3/2 F'=3,mF'=+3`.

`run_fixed_stretched_shift_diagnostic.py` runs beamwise Doppler, unit,
cancellation, position-lineout, and shifted-resonance checks and writes a
self-contained result bundle under `outputs/diagnostics/pmot`. Its default
absolute power is explicitly a historical diagnostic scale, not a pMOT power
recommendation. Use `--power-mw-per-path` to test another scale.

Run from the repository root with the project interpreter:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.pmot.temporary_tests.run_fixed_stretched_shift_diagnostic
```

The raw Arora CSV has no unit declaration and supplies only differential
reference-transition coefficients. These tests therefore cannot create a
24-state Stark Hamiltonian, separate upper/lower level shifts, or a physical
pMOT trajectory.

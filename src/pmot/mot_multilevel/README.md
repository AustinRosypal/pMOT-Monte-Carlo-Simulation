# Multilevel MOT package

This package is the clean replacement for the deleted preliminary multilevel
implementation. It is isolated from `pmot.mot_simple`, which remains the
validated two-level regression model.

Current layers:

- `configuration.py`: angular-frequency conventions and no-repumper settings.
- `atomic_structure.py`: 23 indexed states and precomputed dipole/decay graphs.
- `polarization.py`: propagation-frame polarization and local spherical basis.
- `coupling.py`: Doppler, Zeeman, detuning, and laser-driven rates.
- `events.py`: spontaneous/stimulated channels and Gillespie sampling.
- `trajectory.py`: internal/classical state records, initialization, and recoil.
- `validation.py`: concise report for the implemented Stage A-C foundation.

Run the focused tests with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python -m pytest -q \
  tests/test_multilevel_atomic_structure.py \
  tests/test_multilevel_coupling.py \
  tests/test_multilevel_events.py
```

Run the concise validation report with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.mot_multilevel.validation
```

Do not run multilevel capture or loading calculations yet. Optical-pumping,
dark-leakage ensemble, trajectory-coupling, and two-level-limit tests are the
next gated stages described by `MULTILEVEL_MOT.md`.

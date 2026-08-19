# Multilevel MOT package

This package is the clean replacement for the deleted preliminary multilevel
implementation. It is isolated from `pmot.mot_simple`, which remains the
validated two-level regression model.

Current layers:

- `configuration.py`: angular-frequency conventions, cooling, and repumper settings.
- `atomic_structure.py`: 24 indexed states and precomputed dipole/decay graphs.
- `polarization.py`: propagation-frame polarization and local spherical basis.
- `coupling.py`: Doppler, Zeeman, detuning, and laser-driven rates.
- `events.py`: spontaneous/stimulated channels and Gillespie sampling.
- `trajectory.py`: internal/classical state records, initialization, and recoil.
- `simulation.py`: event-driven trajectories and conditional mean-force diagnostics.
- `diagnostics.py`: reproducible plots, trajectory CSV files, and ensemble summaries.
- `sampling.py`: fixed-speed disk/impact-parameter launch sampling framework.
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

Generate the multilevel force, field, trajectory, and dark-state diagnostic
bundle with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.mot_multilevel.diagnostics
```

Its outputs are isolated under
`outputs/{figures,trajectories,statistics}/mot_multilevel`.

The repumper is explicit and toggleable. It is off by default for dark-state
regression work:

```python
from dataclasses import replace
from pmot.mot_multilevel import default_multilevel_mot_config

config = replace(default_multilevel_mot_config(), repumper_enabled=True)
```

The baseline repumper uses the D2 `F=1 -> F'=2` resonance with
0.1 mW per beam and includes the dipole-allowed off-resonant
`F=1 -> F'=0,1,2` channels. It shares the six cooling-beam paths and
propagation-frame helicities; `F=1 -> F'=3` remains forbidden. Generate the
first-pass repumper diagnostic bundle with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python - <<'PY'
from pmot.mot_multilevel.diagnostics import run_repump_diagnostics
print(run_repump_diagnostics(trajectory_count=8))
PY
```

Repumper diagnostics are written under
`outputs/trajectories/mot_multilevel/repump_diagnostics` and
`outputs/figures/mot_multilevel/repump_diagnostics`.

Run the ten-launch pre-statistical capture screen with per-trajectory plots,
per-trajectory hyperfine walk GIFs, and lifetime-tagged animation filenames
with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.mot_multilevel.screening
```

Run the fixed-speed disk/impact-parameter sampling framework with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.mot_multilevel.sampling --disc-count 4 --points-per-disc 8
```

Do not run multilevel capture or loading calculations yet. Optical-pumping,
dark-leakage ensemble, trajectory-coupling, and two-level-limit tests are the
next gated stages described by `MULTILEVEL_MOT.md`.

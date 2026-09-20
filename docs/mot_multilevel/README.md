# Multilevel MOT population-rate solver

The governing derivation and required equations are in the project-root
`PopulationRateEq_Instructions.md`, with Section 12 taking precedence. The
implemented solver is documented in `src/pmot/mot_multilevel/README.md`.

The previous documents were moved to `docs/mot_error`. They describe an
archived implementation whose transition rates incorrectly reused a saturated
two-level scattering formula inside the multilevel population equations; they
are historical records, not specifications for this solver.

The implemented validity boundary is an incoherent, population-only,
adiabatically eliminated model. It includes ARC dipole matrix elements,
hyperfine splittings, Zeeman shifts, local spherical-polarization projections,
cooling and repump optical pumping, spontaneous branching, and net
radiation-pressure force. It excludes optical/ground coherences, coherent
standing-wave interference, sub-Doppler cooling, and a validated recoil
diffusion model.

The reconstructed simulation interfaces are:

- `notebooks/mot_multilevel/trajectory_explorer.ipynb`;
- `notebooks/mot_multilevel/trajectory_animation.ipynb`;
- `notebooks/mot_multilevel/capture_loading_explorer.ipynb`;
- `src/pmot/mot_multilevel/diagnostics.py` for plots, persistence, and
  animation;
- `src/pmot/mot_multilevel/capture.py` for capture velocity, capture cross
  section, and loading rate.

These are working analysis tools, not a claim that a production capture or
loading campaign has passed convergence. Capture timeouts fail closed, and a
scalar boundary is accepted only with trapped and escaped endpoints.

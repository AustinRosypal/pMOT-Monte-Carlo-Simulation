# Interactive pMOT visualization suite

This directory contains browser-based diagnostics for the provisional
effective two-level pMOT. The pages call the live Python implementation in
`src/pmot/pMOT_2Level`; they do not maintain a second force model in
JavaScript.

## Start the suite

From the repository root:

```powershell
uv run python visualizations/pmot_interactive/serve.py --open
```

If the browser does not open automatically, visit:

```text
http://127.0.0.1:8765/
```

Stop the server with `Ctrl+C`. To use another port:

```powershell
uv run python visualizations/pmot_interactive/serve.py --port 8877 --open
```

Do not open the HTML files directly with a `file://` URL. The force, Stark,
and polarizability data are supplied by localhost API endpoints in `serve.py`.
The server binds only to `127.0.0.1` and does not expose the diagnostics to the
network.

## Pages

- `polarizability-explorer.html`: zoomable scalar/vector/tensor differential
  polarizabilities with markers at both configured pMOT wavelengths. It sends
  every CSV row in the selected wavelength interval to the browser; wavelength
  data are not downsampled by the interactive page. Optical frequency is the
  primary horizontal coordinate, wavelength is the secondary coordinate,
  coefficient magnitudes use a logarithmic axis, and solid/dashed segments
  encode positive/negative coefficients.
- `force-gradient-explorer.html`: force and same-axis spatial derivative with
  adjustable pathway, powers, detuning, axis, offsets, velocity, range, and
  independent scalar/vector/tensor checkboxes. The displayed model name uses
  the canonical `2Level_pMOT_<components>` convention.
- `model-comparison-explorer.html`: live 2-by-3 force and force-gradient
  overlays for the multilevel MOT, `mot_strange`, and two independently
  configured effective two-level pMOT curves. Each pMOT curve has its own
  scalar/vector/tensor switches, powers, detuning, and wavelength pathway, so
  configurations such as `2Level_pMOT_svt` and `2Level_pMOT_sv` can be
  compared directly. Model-specific controls also expose multilevel cooling
  and repump powers, the conventional field gradient, and surrogate-field
  path powers.
- `stark-geometry-explorer.html`: Cartesian-plane maps of scalar, vector,
  tensor, scalar-plus-tensor, and total-intensity fields with quantization-axis
  arrows and out-of-plane symbols.
- `phase-space-explorer.html`: force as a function of position and velocity,
  useful for identifying restoring, damping, anti-restoring, and barrier
  regions.

All force visualizations exclude gravity and remain provisional. They do not
include a state-resolved Stark Hamiltonian, conservative trapping-light force,
1529-nm scattering/heating, recoil diffusion, or validated capture/loading
physics.

The component code is always written in scalar/vector/tensor order: `s`, `v`,
and `t`. Clearing all three checkboxes produces the `2Level_pMOT_none`
no-Stark control. If `v` is disabled while `t` remains enabled, the vector
shift cannot define a local quantization axis; the tensor diagnostic therefore
uses the existing previous/fallback-axis prescription and is explicitly
axis-prescription dependent.

## Validation

Exercise every API without starting a server:

```powershell
uv run python visualizations/pmot_interactive/serve.py --smoke-test
```

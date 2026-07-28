# MOT Simulation Validation Test Specification

## Goal

Implement, run, and report five automated validation tests for the magneto-optical trap simulation:

1. Zero-field optical molasses
2. Zero-velocity restoring force
3. Beam-selection probability
4. Heating versus cooling under detuning reversal
5. Doppler-temperature scale

Use `pytest` unless the repository already has another test framework. A failing physical criterion must produce a failing test and a nonzero exit code.

Run with:

```bash
pytest -v -s tests/test_mot_validation.py
```

Use the production simulation functions rather than reimplementing the physics inside the tests.


## General requirements

- Put the tests in `tests/test_mot_validation.py`, or follow the repository's existing test layout.
- Use fixed random seeds, for example `np.random.default_rng(12345)`.
- Disable unrelated physics in each test.
- State whether frequency quantities use Hz or rad/s. Do not mix them.
- Use SI units unless the project consistently uses another convention.
- Print quantitative measurements, expected behavior, tolerances, and `PASS`, `FAIL`, or `ERROR`.
- Save optional plots only as supplementary diagnostics. A plot is not a pass criterion.
- Do not modify the simulation merely to force a test to pass.
- If a tolerance is changed, explain why and preserve the original measurements.

Use a report block similar to:

```text
============================================================
TEST: Zero-field optical molasses
STATUS: PASS
Measured:
    F_x(+v) = -1.23e-21 N
    F_x(-v) = +1.22e-21 N
Expected:
    F_x(+v) < 0
    F_x(-v) > 0
    odd-symmetry error < 2%
Reason:
    Force opposes motion and is antisymmetric within tolerance.
============================================================
```


# Test 1 — Zero-Field Optical Molasses

## Objective

Verify that equal red-detuned counterpropagating beams produce velocity damping when the magnetic field is disabled.

## Configuration

- Set `B(r) = 0`.
- Use equal counterpropagating cooling beams.
- Set `r = (0, 0, 0)`.
- Disable gravity and other external forces.
- Use red detuning, `Delta < 0`.
- Choose a small velocity satisfying approximately `|k v_test| <= 0.2 Gamma`.

Evaluate the deterministic mean force at:

```text
v = (0, 0, 0)
v = (+v_test, 0, 0)
v = (-v_test, 0, 0)
```

## Pass criteria

Define a characteristic force:

```text
F_scale = hbar * k * Gamma / 2
force_tolerance = 1e-8 * F_scale
```

Adjust only if numerical precision requires it, and document the reason.

Require:

```text
|F_x(0)| <= force_tolerance
|F_y(0)| <= force_tolerance
|F_z(0)| <= force_tolerance

F_x(+v_test) < 0
F_x(-v_test) > 0
```

Define:

```text
symmetry_error =
    |F_x(+v_test) + F_x(-v_test)|
    / max(|F_x(+v_test)|, |F_x(-v_test)|)
```

Require:

```text
symmetry_error < 0.02
```

For motion along `x`, require negligible transverse force:

```text
|F_y| / |F_x| < 0.01
|F_z| / |F_x| < 0.01
```

## Required output

Print:

- detuning;
- test velocity;
- force at zero velocity;
- force at positive and negative velocity;
- symmetry error;
- transverse-force ratios;
- `PASS` or `FAIL`.


# Test 2 — Zero-Velocity Restoring Force

## Objective

Verify that the magnetic quadrupole field and MOT beam helicities produce a force toward the trap center.

## Configuration

- Enable the normal MOT quadrupole field.
- Set `v = (0, 0, 0)`.
- Use equal beam intensities and intended MOT polarizations.
- Disable gravity.
- Choose small displacements inside both the linear magnetic-field region and the central part of the beam profile.

Evaluate:

```text
(+x_test, 0, 0), (-x_test, 0, 0)
(0, +y_test, 0), (0, -y_test, 0)
(0, 0, +z_test), (0, 0, -z_test)
```

## Pass criteria

For each axis `i`, require:

```text
F_i(+r_test) < 0
F_i(-r_test) > 0
```

Define:

```text
symmetry_error_i =
    |F_i(+r_test) + F_i(-r_test)|
    / max(|F_i(+r_test)|, |F_i(-r_test)|)
```

Require:

```text
symmetry_error_i < 0.05
```

Estimate the spring constant:

```text
kappa_i =
    -[F_i(+r_test) - F_i(-r_test)] / (2 r_test)
```

Require:

```text
kappa_i > 0
```

For displacement along axis `i`, require:

```text
|F_transverse| / |F_i| < 0.05
```

Strongly recommended: evaluate `r_i = {-2r_test, -r_test, 0, +r_test, +2r_test}`, fit

```text
F_i = -kappa_i r_i + b
```

and require:

```text
R_squared > 0.98
```

## Required output

For each axis print:

- displacements;
- all force components;
- estimated `kappa_i`;
- symmetry error;
- transverse-force fraction;
- linear-fit result, if implemented;
- axis-specific `PASS` or `FAIL`.

The complete test passes only if every tested axis passes.


# Test 4 — Beam-Selection Probability

## Objective

Verify that the Monte Carlo beam selector chooses each beam with conditional probability

```text
P(j | scattering event) = R_j / sum_l R_l.
```

Test the selector independently of trajectory propagation.

## Configuration

- Disable the magnetic field.
- Set `r = (0, 0, 0)`.
- Set `v = (0, 0, -v_test)`.
- Use red-detuned light.
- Keep the atom's state fixed while drawing samples.
- Calculate the rates once and do not update position, velocity, or internal state.

The `+z` beam should be Doppler shifted closer to resonance, so first require:

```text
R_(+z) > R_(-z)
```

Calculate:

```text
R_total = sum_j R_j
p_j = R_j / R_total
```

Run at least:

```text
N = 100_000
```

conditional beam-selection draws. Do not include the no-scattering branch in this test.

## Pass criteria

For each beam, measure:

```text
n_j = selected count
f_j = n_j / N
sigma_j = sqrt(N * p_j * (1 - p_j))
z_j = |n_j - N p_j| / sigma_j
```

For categories with sufficiently large expected counts, require:

```text
z_j < 5
```

Also require:

```text
f_(+z) > f_(-z)
|f_j - p_j| < max(5 * sqrt(p_j * (1 - p_j) / N), 1e-3)
```

If an expected count is below roughly 20, combine rare categories or state that the normal approximation is unreliable.

Optional: perform a chi-square goodness-of-fit test and require `p_value > 0.001`.

## Required output

Print:

```text
Beam    Detuning    Rate (1/s)    Expected p    Observed f    Count    z-score
+x      ...
-x      ...
+y      ...
-y      ...
+z      ...
-z      ...
```

Also print:

- `R_(+z) / R_(-z)`;
- `p_(+z) / p_(-z)`;
- sample count;
- random seed;
- `PASS` or `FAIL`.


# Test 7 — Heating Versus Cooling Under Detuning Reversal

## Objective

Verify that red detuning produces damping, blue detuning produces anti-damping, and zero detuning produces little or no linear damping.

## Configuration

- Disable the magnetic field.
- Set `r = (0, 0, 0)`.
- Use equal counterpropagating beams.
- Choose a small test speed.
- Use three detunings:

```text
Delta_red < 0
Delta_zero = 0
Delta_blue = -Delta_red > 0
```

Evaluate forces at `+v_test` and `-v_test`.

Estimate:

```text
alpha(Delta) =
    -[F_x(+v_test, Delta) - F_x(-v_test, Delta)]
    / (2 v_test)
```

## Pass criteria

Require:

```text
alpha_red > 0
alpha_blue < 0
sign(alpha_red) = -sign(alpha_blue)
```

Equivalently, at positive velocity:

```text
F_x_red < 0
F_x_blue > 0
```

Require weak damping at zero detuning:

```text
|alpha_zero| < 0.1 * |alpha_red|
```

For an ideal two-level model with equal-magnitude red and blue detuning, require approximately:

```text
|alpha_red + alpha_blue|
/ max(|alpha_red|, |alpha_blue|)
< 0.1
```

Use a looser tolerance only if a documented multilevel asymmetry justifies it.

## Required output

Print:

- all detunings;
- forces at positive and negative test velocities;
- damping coefficients;
- sign checks;
- `|alpha_zero| / |alpha_red|`;
- red-blue symmetry error;
- `PASS` or `FAIL`.


# Test 14 — Doppler-Temperature Scale

## Objective

Verify that stochastic recoil plus Doppler damping reaches a finite equilibrium temperature on the expected Doppler scale.

This test is valid only when spontaneous-emission recoil, or an equivalent momentum-diffusion model, is enabled.

## Reference

Calculate from the simulation constants:

```text
T_D = hbar * Gamma / (2 k_B)
```

Use a consistent linewidth convention. If `Gamma` is angular frequency in the scattering equations, use the same `Gamma` here.

## Configuration

Use optical molasses for the cleanest comparison:

- `B(r) = 0`;
- six equal red-detuned beams;
- low saturation;
- `Delta approximately -Gamma / 2`;
- no gravity;
- initial ensemble temperature clearly above `T_D`;
- at least `N_atoms >= 1000`, if computationally practical;
- run long enough to reach a stationary velocity distribution.

## Temperature calculation

Subtract the center-of-mass velocity and calculate:

```text
T_x = m * Var(v_x) / k_B
T_y = m * Var(v_y) / k_B
T_z = m * Var(v_z) / k_B
T = (T_x + T_y + T_z) / 3
```

Do not estimate temperature from mean speed.

## Equilibration check

Divide the final part of the run into at least four windows. Calculate the mean temperature in each window.

Define:

```text
stationarity_metric =
    (max(window_means) - min(window_means))
    / mean(window_means)
```

Require:

```text
stationarity_metric < 0.15
```

Also verify that the final windows do not show a monotonic collapse toward zero.

## Pass criteria

Require cooling:

```text
T_final < T_initial
```

Require a finite temperature:

```text
T_final > 0
```

Initially use the broad Doppler-scale requirement:

```text
0.3 * T_D <= T_final <= 3.0 * T_D
```

After independent convergence and recoil checks pass, optionally tighten to:

```text
0.7 * T_D <= T_final <= 1.5 * T_D
```

For symmetric beams, require approximate isotropy:

```text
max(T_x, T_y, T_z) / min(T_x, T_y, T_z) < 1.25
```

Repeat using `dt` and `dt / 2` and require:

```text
|T(dt) - T(dt/2)| / T(dt/2) < 0.15
```

Failure of timestep convergence means the test fails even if the absolute temperature is near `T_D`.

## Required output

Print:

- atomic mass;
- linewidth and convention;
- saturation;
- detuning;
- timestep;
- number of atoms;
- initial temperature;
- `T_D`;
- final `T_x`, `T_y`, `T_z`;
- final average temperature;
- `T_final / T_D`;
- stationarity metric;
- anisotropy ratio;
- timestep-convergence metric;
- `PASS` or `FAIL`.

Optionally save `temperature_vs_time.png`.


# Suggested reusable helpers

Use production physics functions wherever possible.

```python
def evaluate_mean_force(position, velocity, config):
    ...
```

```python
def calculate_beam_rates(position, velocity, internal_state, config):
    ...
```

```python
def sample_beam_from_rates(rates, rng):
    probabilities = rates / rates.sum()
    return rng.choice(len(rates), p=probabilities)
```

```python
def calculate_temperature(velocities, atomic_mass, boltzmann_constant):
    centered = velocities - velocities.mean(axis=0, keepdims=True)
    return atomic_mass * centered.var(axis=0, ddof=1) / boltzmann_constant
```

```python
def report_test_result(name, passed, measured, expected, reason):
    ...
```


# Required final report

After implementation:

1. Run the entire validation suite.
2. Capture the terminal output.
3. Report every test as `PASS`, `FAIL`, or `ERROR`.
4. For each failure, identify the exact failed criterion.
5. State the likely physical or numerical cause.
6. Point to the relevant source file and function.
7. Do not conceal failures or silently loosen tolerances.
8. Save optional plots in a validation-results directory.

Use this final summary format:

```text
MOT VALIDATION SUMMARY

1. Zero-field optical molasses: PASS
2. Zero-velocity restoring force: FAIL
3. Beam-selection probability: PASS
4. Detuning reversal: PASS
5. Doppler-temperature scale: ERROR

Passed: 3
Failed: 1
Errors: 1

Failure details:
- Zero-velocity restoring force:
  The force at x > 0 points outward.
  Likely causes: incorrect Zeeman-shift sign, reversed beam helicity,
  or inconsistent quantization-axis convention.

Error details:
- Doppler-temperature scale:
  The simulation does not expose an ensemble evolution path with
  stochastic recoil enabled.
```

The task is complete only after the tests have been implemented, executed, and reported.

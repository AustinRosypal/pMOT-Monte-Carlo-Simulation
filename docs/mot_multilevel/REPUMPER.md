# REPUMPER.md

## Purpose

Implement realistic repumper-light physics in the existing multilevel \(^{87}\mathrm{Rb}\) D2 MOT Monte Carlo simulation.

The repumper must prevent atoms that leak from the bright ground manifold

\[
5S_{1/2},F=2
\]

into the dark ground manifold

\[
5S_{1/2},F=1
\]

from remaining optically inactive. The repumper should excite \(F=1\) atoms back into the \(5P_{3/2}\) manifold, after which spontaneous decay can return them to \(F=2\), where normal cooling resumes.

Do **not** implement repumping as an artificial state reset such as

```python
if F == 1:
    F = 2
```

The repumper must be simulated using the same state-resolved absorption, Zeeman, Doppler, polarization, Clebsch-Gordan, recoil, and spontaneous-decay machinery as the cooling light.

---

# 1. Primary Repump Transition

Use the standard \(^{87}\mathrm{Rb}\) D2 repump transition

\[
\boxed{
5S_{1/2},F=1
\rightarrow
5P_{3/2},F'=2
}
\]

as the dominant repumping transition.

The cooling transition remains

\[
5S_{1/2},F=2
\rightarrow
5P_{3/2},F'=3.
\]

The repumper returns atoms to the bright manifold through

\[
F=1
\rightarrow
F'=2
\rightarrow
F=2.
\]

Because \(F'=2\) can also decay back to \(F=1\), more than one repump scattering event may be required before an atom successfully returns to \(F=2\).

---

# 2. Frequencies and Wavelengths

Use frequencies internally whenever possible rather than rounded wavelengths.

Approximate resonant values:

| Transition | Resonant Frequency | Vacuum Wavelength |
|---|---:|---:|
| Cooling \(F=2\rightarrow F'=3\) | \(384.228115~\mathrm{THz}\) | \(780.246021~\mathrm{nm}\) |
| Repump \(F=1\rightarrow F'=2\) | \(384.234683~\mathrm{THz}\) | \(780.232684~\mathrm{nm}\) |

The resonant frequency separation is approximately

\[
\boxed{
\nu_{\rm rep,res}
-
\nu_{\rm cool,res}
\approx
6.56803~\mathrm{GHz}.
}
\]

The ground-state hyperfine splitting is approximately

\[
\Delta_{\rm ground}
=
6.8346826~\mathrm{GHz},
\]

while the excited-state separation

\[
F'=3-F'=2
\]

is approximately

\[
266.650~\mathrm{MHz}.
\]

Therefore

\[
6.8346826~\mathrm{GHz}
-
0.266650~\mathrm{GHz}
=
6.56803~\mathrm{GHz}.
\]

The present cooling laser is

\[
\Delta_{\rm cool}/2\pi=-15~\mathrm{MHz}
\]

relative to \(F=2\rightarrow F'=3\).

For the initial repump implementation use

\[
\boxed{
\Delta_{\rm rep}/2\pi=0~\mathrm{MHz}
}
\]

relative to the \(F=1\rightarrow F'=2\) resonance.

Therefore, relative to the actual \(-15~\mathrm{MHz}\)-detuned cooling-laser frequency,

\[
\boxed{
\nu_{\rm rep}
-
\nu_{\rm cool,laser}
\approx
6.58303~\mathrm{GHz}.
}
\]

---

# 3. Repumper Geometry

For the baseline implementation, give every cooling-beam path a repump-frequency component.

Use:

- the same 6 propagation directions as the MOT cooling beams,
- the same beam axes,
- the same Gaussian transverse profile,
- the same beam radius,
- the same modeled beam length,
- the same laboratory-frame polarization handedness as the corresponding cooling beam.

Current beam radius:

\[
\boxed{
w_{\rm rep}=w_{\rm cool}=6.35~\mathrm{mm}.
}
\]

Thus each physical beam should conceptually contain two optical-frequency components:

\[
\text{beam }j:
\qquad
\left\{
\begin{array}{l}
\omega_{\rm cool}, P_{\rm cool} \\
\omega_{\rm rep}, P_{\rm rep}
\end{array}
\right.
\]

with the same \(\mathbf{k}_j\), spatial profile, and lab-frame polarization.

Do not create a separate repump geometry unless needed later for a specific experimental comparison.

---

# 4. Initial Repump Power

Use a modest repump power initially.

Recommended starting range:

\[
\boxed{
P_{\rm rep}=0.1\text{--}0.2~\mathrm{mW/beam}.
}
\]

A useful power scan is

\[
0.02,\;
0.05,\;
0.10,\;
0.20,\;
0.50~\mathrm{mW/beam}.
\]

The purpose of the scan is to find where the simulated capture statistics and \(F=1\) dwell times become insensitive to additional repump power.

Do not assume there is one universal experimental repump power.

---

# 5. Polarization Treatment

Do not assign a fixed transition label \(q\) to a repump beam.

Use the same local-polarization decomposition already used for the cooling light.

At each atom position:

\[
\hat{\mathbf b}
=
\frac{\mathbf B}{|\mathbf B|}
\]

defines the local quantization axis.

Project the lab-frame repump polarization vector into the local spherical basis:

\[
|\epsilon_{+1}|^2,
\qquad
|\epsilon_0|^2,
\qquad
|\epsilon_{-1}|^2.
\]

These correspond to:

\[
q=+1 \quad (\sigma^+),
\]

\[
q=0 \quad (\pi),
\]

\[
q=-1 \quad (\sigma^-).
\]

For every candidate transition require

\[
m_F'=m_F+q.
\]

The repumper must therefore use the exact same local-basis polarization logic as the cooling beams.

---

# 6. Ground and Excited Repump Manifolds

The dark ground manifold is

\[
F=1,
\qquad
m_F=-1,0,+1.
\]

The dominant repump excited manifold is

\[
F'=2,
\qquad
m_F'=-2,-1,0,+1,+2.
\]

For a more complete implementation, allow repump excitation from \(F=1\) into all dipole-allowed D2 hyperfine excited manifolds:

\[
\boxed{
F=1\rightarrow F'=0,1,2.
}
\]

Do not allow

\[
F=1\rightarrow F'=3
\]

because \(\Delta F=2\) is forbidden for an electric-dipole transition.

Approximate excited-state offsets relative to \(F'=2\):

\[
F'=1:
\quad
-156.947~\mathrm{MHz},
\]

\[
F'=0:
\quad
-229.165~\mathrm{MHz}.
\]

Equivalently, a repumper resonant with \(F'=2\) is approximately

\[
+156.947~\mathrm{MHz}
\]

blue-detuned from \(F'=1\), and

\[
+229.165~\mathrm{MHz}
\]

blue-detuned from \(F'=0\).

The first implementation may use only \(F'=2\), but the preferred final implementation should include \(F'=0,1,2\) consistently.

---

# 7. Zeeman Shifts

Use the same Zeeman-shift machinery as the cooling transition.

Approximate Landé factors:

\[
g_F(F=1)\approx-\frac12,
\]

\[
g_{F'}(F'=2)\approx+\frac23.
\]

The state-dependent transition shift is

\[
\Delta\omega_Z
=
\frac{\mu_B B}{\hbar}
\left(
g_{F'}m_F'
-
g_Fm_F
\right).
\]

If the code defines laser detuning as

\[
\Delta
=
\omega_L-\omega_0,
\]

then the effective repump detuning for beam \(j\) is

\[
\boxed{
\Delta_{\rm eff}^{(j)}
=
\Delta_{\rm rep}
-
\mathbf k_j\cdot\mathbf v
-
\frac{\mu_BB}{\hbar}
\left(
g_{F'}m_F'
-
g_Fm_F
\right).
}
\]

Use the exact same sign convention that has already passed the cooling-beam symmetry tests.

---

# 8. Doppler Shift

Repump transitions must include Doppler shifts.

For every repump beam,

\[
\Delta_{\rm Doppler}
=
-\mathbf k_j\cdot\mathbf v.
\]

Do not model repumping with a constant position-independent and velocity-independent rate.

At \(780~\mathrm{nm}\), an atom moving at \(20~\mathrm{m/s}\) experiences a Doppler shift of order

\[
25.6~\mathrm{MHz},
\]

which is several natural linewidths.

This can substantially alter which repump beam is closest to resonance.

---

# 9. Transition Strengths

For the dominant

\[
F=1\rightarrow F'=2
\]

repump transition, use state-resolved dipole strengths.

The squared dipole strengths below include the hyperfine reduced-matrix-element factor.

## Absolute strengths in units of \(d_J^2\)

| Initial \(m_F\) | \(\sigma^-\) | \(\pi\) | \(\sigma^+\) |
|---:|---:|---:|---:|
| \(-1\) | \(1/4\) | \(1/8\) | \(1/24\) |
| \(0\)  | \(1/8\) | \(1/6\) | \(1/8\) |
| \(+1\) | \(1/24\) | \(1/8\) | \(1/4\) |

If the simulation normalizes all dipole strengths to the stretched cooling transition

\[
|F=2,m_F=\pm2\rangle
\rightarrow
|F'=3,m_F'=\pm3\rangle,
\]

whose squared strength is

\[
\frac12 d_J^2,
\]

then use the following relative weights.

## Strengths normalized to the stretched cooling transition

| Initial \(m_F\) | \(\sigma^-\) | \(\pi\) | \(\sigma^+\) |
|---:|---:|---:|---:|
| \(-1\) | \(1/2\) | \(1/4\) | \(1/12\) |
| \(0\)  | \(1/4\) | \(1/3\) | \(1/4\) |
| \(+1\) | \(1/12\) | \(1/4\) | \(1/2\) |

Prefer computing these weights from a unified Wigner-3j / Wigner-6j / reduced-matrix-element implementation rather than hardcoding isolated tables, provided the existing atomic-physics utilities support this reliably.

---

# 10. Saturation Intensity Convention

The present cooling model uses approximately

\[
I_{\rm sat,cool}
=
16.69~\mathrm{W/m^2}
=
1.669~\mathrm{mW/cm^2}
\]

for the maximally stretched cooling transition.

The strongest \(F=1\rightarrow F'=2\) repump transition has half the squared dipole moment of the stretched cooling transition.

Therefore an equivalent two-level saturation intensity for the strongest repump line is approximately

\[
\boxed{
I_{\rm sat,rep,max}
\approx
3.338~\mathrm{mW/cm^2}.
}
\]

However, for the multilevel implementation, prefer a single reference saturation-intensity convention together with explicit relative dipole-strength factors rather than assigning a different empirical \(I_{\rm sat}\) to every Zeeman transition.

Be internally consistent with the existing cooling scattering-rate implementation.

---

# 11. Repump Scattering Rate

Use the same scattering formalism as the cooling light.

For each candidate repump channel \(c\),

\[
c=(j,F',m_F',q),
\]

calculate a channel rate of the same form already used for cooling:

\[
R_c
=
R
\left(
I_j,
|\epsilon_{j,q}|^2,
|d_c|^2,
\Delta_{\rm eff,c},
\Gamma
\right).
\]

The implementation must preserve the same conventions already validated for:

- total saturation in the denominator,
- local intensity,
- transition-strength weighting,
- Doppler shift,
- Zeeman shift,
- polarization decomposition.

Then sum all allowed repump channels:

\[
R_{\rm rep,total}
=
\sum_c R_c.
\]

For timestep \(\Delta t\),

\[
\boxed{
P_{\rm rep,scatter}
=
1-
e^{-R_{\rm rep,total}\Delta t}.
}
\]

If scattering occurs, choose the absorption channel probabilistically:

\[
\boxed{
P(c|\text{scatter})
=
\frac{R_c}
{R_{\rm rep,total}}.
}
\]

Do not select a channel independently from the others after a scattering event has already been chosen.

---

# 12. When Cooling and Repump Scattering Are Evaluated

A clean first implementation is:

## If the atom is in \(F=2\)

Evaluate the cooling-light scattering channels.

The repump laser is approximately \(6.8~\mathrm{GHz}\) away from the \(F=2\) ground-state transitions and may be neglected initially.

## If the atom is in \(F=1\)

Evaluate the repump-light scattering channels.

The cooling laser is several GHz away from the \(F=1\) transitions and may be neglected initially.

This separation is physically appropriate and keeps the initial implementation efficient.

A later precision implementation may evaluate all optical frequencies against all allowed transitions, but this is not necessary for the first validated repumper implementation.

---

# 13. Absorption Recoil

Every repump absorption must impart momentum

\[
\boxed{
\Delta\mathbf p_{\rm abs}
=
+\hbar\mathbf k_{\rm rep,j}.
}
\]

The recoil magnitude is essentially the same as for the cooling transition because the wavelengths differ by only about \(0.013~\mathrm{nm}\).

Do not ignore repump absorption recoil.

---

# 14. Spontaneous Decay After Repump Excitation

After repump excitation into

\[
|F',m_F'\rangle,
\]

sample spontaneous decay into all allowed ground states

\[
|F_g,m_g\rangle,
\qquad
F_g=1,2.
\]

For each allowed decay channel,

\[
W_i
\propto
\left|
\langle F_g,m_g|
d_q
|F',m_F'\rangle
\right|^2.
\]

Normalize:

\[
P_i
=
\frac{W_i}
{\sum_k W_k}.
\]

Sample the final state using these normalized branching probabilities.

Then:

1. update \(F\),
2. update \(m_F\),
3. apply spontaneous-emission recoil.

For \(F'=2\), summing over all Zeeman sublevels should reproduce the total hyperfine branching fractions

\[
\boxed{
P(F'=2\rightarrow F=1)=\frac12,
}
\]

\[
\boxed{
P(F'=2\rightarrow F=2)=\frac12.
}
\]

This must be explicitly tested.

If decay returns the atom to \(F=1\), continue repumping.

If decay places the atom in \(F=2\), the atom resumes ordinary cooling-light dynamics.

---

# 15. Spontaneous-Emission Recoil

Every spontaneous decay must impart a recoil momentum corresponding to the emitted photon.

Use the same spontaneous-emission recoil model already implemented for cooling.

If emission directions are sampled isotropically, sample a random unit vector

\[
\hat{\mathbf n}_{\rm emit}
\]

and apply

\[
\Delta\mathbf p_{\rm spont}
=
-\hbar k \hat{\mathbf n}_{\rm emit}.
\]

Maintain the same convention already used in the cooling simulation.

---

# 16. Suggested Atomic-State Logic

The atom state should remain explicitly represented as

```text
(F, mF)
```

with

```text
F = 1 or 2
```

for the ground state.

The simulation loop should behave conceptually as follows:

```python
if atom.F == 2:
    channels = build_cooling_channels(atom, beams, B_field)
else:
    channels = build_repump_channels(atom, repump_beams, B_field)

R_total = sum(channel.rate for channel in channels)

P_scatter = 1 - exp(-R_total * dt)

if random() < P_scatter:
    channel = weighted_random_choice(
        channels,
        weights=[c.rate for c in channels],
    )

    apply_absorption_recoil(channel.k)

    excited_state = channel.final_excited_state

    ground_state = sample_spontaneous_decay(excited_state)

    apply_spontaneous_recoil()

    atom.F = ground_state.F
    atom.mF = ground_state.mF
```

Use existing project abstractions rather than duplicating logic wherever possible.

The ideal architecture is one generic optical-scattering engine that can operate on either cooling or repump frequencies.

---

# 17. Recommended Beam Representation

Prefer extending the current beam data structure rather than creating an unrelated repump-beam implementation.

A physical MOT beam may be represented conceptually as

```python
beam = {
    "k_hat": ...,
    "waist": ...,
    "origin": ...,
    "lab_polarization": ...,
    "frequency_components": [
        {
            "kind": "cooling",
            "frequency": omega_cool,
            "power": P_cool,
        },
        {
            "kind": "repump",
            "frequency": omega_rep,
            "power": P_rep,
        },
    ],
}
```

Equivalent object-oriented designs are acceptable.

The important principle is that cooling and repump light may share the same physical optical path while remaining distinct frequency components.

---

# 18. Required Validation Tests

The repumper implementation must not be considered complete until the following tests pass.

## Test 1 — Repump Selection Rules

For every \(F=1,m_F\) state:

- verify only \(q=-1,0,+1\) are allowed,
- verify \(m_F'=m_F+q\),
- verify invalid \(m_F'\) values are rejected,
- verify \(F'=3\) is forbidden.

Expected dominant manifold:

\[
F=1\rightarrow F'=2.
\]

---

## Test 2 — Repump CG / Dipole-Strength Table

Numerically reproduce the expected \(F=1\rightarrow F'=2\) strengths.

Normalized to the stretched cooling transition:

| \(m_F\) | \(\sigma^-\) | \(\pi\) | \(\sigma^+\) |
|---:|---:|---:|---:|
| \(-1\) | \(1/2\) | \(1/4\) | \(1/12\) |
| \(0\)  | \(1/4\) | \(1/3\) | \(1/4\) |
| \(+1\) | \(1/12\) | \(1/4\) | \(1/2\) |

Fail if these are not reproduced within numerical tolerance.

---

## Test 3 — Hyperfine Branching Ratio

Prepare many atoms in random \(F'=2,m_F'\) excited states and sample spontaneous decay.

After summing over Zeeman substates, verify approximately

\[
50\%
\rightarrow F=1,
\]

\[
50\%
\rightarrow F=2.
\]

Use enough events that the statistical uncertainty is small.

---

## Test 4 — Pure \(\sigma^+\) Repump Optical Pumping

Use:

- uniform nonzero magnetic field,
- stationary atom,
- one pure \(\sigma^+\) repump beam,
- atom initially in \(F=1\).

Verify the transition pattern follows the allowed positive-\(q\) channels and that the state populations are consistent with the CG coefficients.

Repeat with pure \(\sigma^-\) and verify the mirrored behavior.

---

## Test 5 — Zeeman-Sign Test

For known \(m_F,m_F'\) and \(B\), compare the calculated repump transition shift to

\[
\Delta\omega_Z
=
\frac{\mu_BB}{\hbar}
\left(
g_{F'}m_F'
-
g_Fm_F
\right).
\]

Explicitly verify the negative sign of

\[
g_F(F=1).
\]

---

## Test 6 — Doppler-Sign Test

Launch atoms parallel and antiparallel to a repump beam.

Verify that

\[
-\mathbf k\cdot\mathbf v
\]

moves one case toward resonance and the other away from resonance according to the existing detuning convention.

---

## Test 7 — Resonant Repumping Time

Prepare atoms in \(F=1\) at or near the MOT center with the repumper enabled.

Measure the time until each atom first returns to \(F=2\).

Expected qualitative result:

\[
\boxed{
\text{repumping time}
\sim
\text{microseconds},
}
\]

substantially shorter than the previously observed no-repump dark-state entry time of approximately

\[
54~\mu\mathrm{s}.
\]

The exact mean depends on repump power and polarization.

---

## Test 8 — Repump-Power Scan

Run the same initial conditions for several repump powers:

\[
0.02,\;
0.05,\;
0.10,\;
0.20,\;
0.50~\mathrm{mW/beam}.
\]

Record:

- mean \(F=1\) dwell time,
- median \(F=1\) dwell time,
- number of repump photons per dark-state visit,
- capture probability,
- capture velocity.

Confirm that stronger repump light reduces \(F=1\) dwell time until the result begins to saturate.

---

## Test 9 — No-Repump Regression

Set

\[
P_{\rm rep}=0.
\]

Verify that the simulation reproduces the previous dark-state behavior rather than silently repumping atoms.

This ensures the repumper is the only new mechanism returning \(F=1\) atoms to \(F=2\).

---

## Test 10 — MOT Symmetry Test

Repeat the existing simultaneous symmetry transformation:

\[
\mathbf B\rightarrow-\mathbf B,
\]

and

\[
\sigma^+\leftrightarrow\sigma^-.
\]

With repumping enabled, the mechanical physics and capture statistics should remain invariant within Monte Carlo uncertainty.

---

# 19. Required Diagnostics to Save

For each atom, record enough information to diagnose repump behavior without rerunning the full simulation.

At minimum save:

- total time spent in \(F=1\),
- number of visits to \(F=1\),
- duration of each \(F=1\) visit,
- number of repump absorption events,
- number of repump photons required for each return to \(F=2\),
- repump beam ID for every repump absorption,
- \(F,m_F\) before repump absorption,
- \(F',m_F'\) after absorption,
- \(F,m_F\) after spontaneous decay,
- local magnetic field,
- local velocity,
- effective detuning,
- local \(\sigma^+,\pi,\sigma^-\) weights,
- absorption-channel rate,
- position of each repump event.

Save numerical data in `.npz`, `.parquet`, or another analysis-friendly format in addition to any `.png` figures.

---

# 20. Recommended Output Plots

After implementing the repumper, generate at least:

1. histogram of individual \(F=1\) dwell times,
2. survival curve for \(F=1\) dwell time,
3. histogram of repump photons required per dark-state visit,
4. repump event count by beam,
5. repump event count by \(m_F\rightarrow m_F'\) channel,
6. fraction of simulation time spent in \(F=1\) versus \(F=2\),
7. capture probability with and without repumping,
8. capture velocity with and without repumping,
9. capture probability versus repump power,
10. mean \(F=1\) dwell time versus repump power.

Always save the raw numerical data that generated these plots.

---

# 21. Expected Physical Behavior

Without repumping:

\[
F=2
\rightarrow
F'=2
\rightarrow
F=1
\]

eventually leaves the atom dark to the cooling laser.

The previously observed mean time to enter \(F=1\) is approximately

\[
54~\mu\mathrm{s}.
\]

With repumping enabled, expect repeated behavior of the form

\[
F=2
\rightarrow
\cdots
\rightarrow
F=1
\]

followed after a much shorter interval by

\[
F=1
\rightarrow
F'=2
\rightarrow
F=1
\]

possibly one or more times, and eventually

\[
\boxed{
F=1
\rightarrow
F'=2
\rightarrow
F=2.
}
\]

The atom then resumes cooling.

A successful repumper should therefore produce:

- much shorter dark-state dwell times,
- repeated cycling between \(F=2\) and occasional brief \(F=1\) visits,
- substantially more total cooling-photon scattering,
- higher capture probability,
- higher attainable capture velocity,
- longer optically active trajectories.

---

# 22. Baseline Configuration

Use the following initial configuration before any optimization:

\[
\boxed{
\begin{aligned}
\text{species} &: {}^{87}\mathrm{Rb} \\
\text{repump transition} &: F=1\rightarrow F'=2 \\
\Delta_{\rm rep}/2\pi &: 0~\mathrm{MHz} \\
\nu_{\rm rep} &: \approx384.234683~\mathrm{THz} \\
\lambda_{\rm rep} &: \approx780.232684~\mathrm{nm} \\
P_{\rm rep} &: 0.1\text{--}0.2~\mathrm{mW/beam} \\
w_{\rm rep} &: 6.35~\mathrm{mm} \\
\text{beam geometry} &: \text{same six paths as cooling} \\
\text{lab polarization} &: \text{same as corresponding cooling beam} \\
\text{quantization axis} &: \hat{\mathbf B}(\mathbf r) \\
\text{ground states} &: F=1,2 \\
\text{repump excited states} &: F'=0,1,2 \\
\text{dominant repump manifold} &: F'=2.
\end{aligned}
}
\]

Do not tune detuning, geometry, or polarization until this baseline implementation passes the validation tests above.

---

# 23. Implementation Priority

Implement in this order:

1. add repump frequency and power parameters,
2. extend each MOT beam to include a repump frequency component,
3. enable \(F=1\rightarrow F'=2\) state-resolved excitation,
4. include local polarization decomposition,
5. include Doppler and Zeeman shifts,
6. include absorption recoil,
7. include full spontaneous decay into \(F=1\) and \(F=2\),
8. validate branching ratios,
9. validate repump timing,
10. add \(F'=0,1\) off-resonant channels,
11. run power scans,
12. compare capture statistics with and without repumping.

The priority is physical correctness and validation before performance optimization.

---

# 24. Completion Criteria

The repumper implementation is complete only when all of the following are true:

- \(F=1\) is no longer terminal.
- Repump excitation obeys correct selection rules.
- CG / dipole strengths are correct.
- Doppler and Zeeman shifts are included.
- Polarization is resolved relative to the local magnetic field.
- Repump absorption recoil is included.
- Spontaneous-emission recoil is included.
- \(F'=2\) branching to \(F=1\) and \(F=2\) is correct.
- Dark-state dwell times become microsecond-scale at reasonable repump power.
- Turning repump power to zero reproduces the previous no-repump result.
- Capture probability and capture velocity increase when repumping is enabled.
- Existing MOT symmetry tests continue to pass.
- All diagnostic data are saved in numerical form, not only as images.

The resulting model should then represent a conventional multilevel \(^{87}\mathrm{Rb}\) MOT with explicit cooling, dark-state leakage, and repumping dynamics.

# MULTILEVEL MOT SIMULATION

## Purpose

This document specifies the next stage of the validated Rubidium-87 magneto-optical trap (MOT) simulation.

The existing simulation treats each Rb-87 atom as an effective two-level system with:

- one ground state,
- one excited state,
- six 780 nm cooling beams,
- a red detuning of approximately 15 MHz,
- Doppler shifts,
- an effective Zeeman-like shift,
- stochastic photon scattering,
- recoil,
- classical atomic trajectories,
- capture cross-section calculations,
- atom flux calculations,
- and a validated MOT loading-rate calculation.

The next model must retain this validated classical trajectory framework while replacing the effective two-level internal structure with a state-resolved multilevel model of the Rb-87 D2 transition.

The purpose of this stage is to simulate:

- the Rb-87 ground hyperfine structure,
- the Rb-87 excited D2 hyperfine structure,
- the Zeeman sublevels within those manifolds,
- state-dependent laser couplings,
- Clebsch-Gordan transition strengths,
- local polarization decomposition into sigma-plus, pi, and sigma-minus components,
- optical pumping among Zeeman states,
- off-resonant excitation to nearby excited hyperfine states,
- spontaneous decay with correct branching probabilities,
- photon recoil,
- and leakage into the dark \(F=1\) ground manifold.

A repumper laser is **not** included at this stage.

The simulation must explicitly track the dark-state population over time.

---

# 1. Physical Model

The simulated cooling transition is based on the Rb-87 D2 line:

\[
5S_{1/2} \rightarrow 5P_{3/2}.
\]

The MOT cooling laser is tuned approximately 15 MHz red of the

\[
5S_{1/2},F=2 \rightarrow 5P_{3/2},F'=3
\]

transition.

The model should not include the D1 \(5P_{1/2}\) manifold at approximately 795 nm.

The relevant internal structure is therefore the hyperfine and Zeeman structure of the D2 system.

---

# 2. Internal States

## 2.1 Ground States

Include both ground hyperfine manifolds.

### \(F=2\)

\[
m_F=-2,-1,0,+1,+2.
\]

This gives 5 ground states.

### \(F=1\)

\[
m_F=-1,0,+1.
\]

This gives 3 ground states.

Total ground states:

\[
N_g=8.
\]

The \(F=1\) manifold will be treated as dark in this no-repumper model.

---

## 2.2 Excited States

For excitation from \(F=2\), include the allowed D2 excited hyperfine manifolds

\[
F'=1,2,3.
\]

Their Zeeman states are:

### \(F'=1\)

\[
m_{F'}=-1,0,+1.
\]

### \(F'=2\)

\[
m_{F'}=-2,-1,0,+1,+2.
\]

### \(F'=3\)

\[
m_{F'}=-3,-2,-1,0,+1,+2,+3.
\]

Total excited states:

\[
3+5+7=15.
\]

Therefore the active model contains

\[
8+15=23
\]

internal states.

The code architecture should still be sufficiently general that \(F'=0\) could be added later if desired.

---

# 3. Recommended Simulation Strategy

The recommended model is:

\[
\boxed{
\text{classical atomic trajectory}
+
\text{stochastic internal-state Markov process}
}
\]

The atom should possess a definite internal state at any instant.

The atom's translational degrees of freedom are treated classically:

\[
\mathbf r(t),\qquad \mathbf v(t).
\]

The internal state evolves through stochastic laser excitation, stimulated emission, and spontaneous emission events.

This is preferable to solving the complete optical Bloch equations for every atom.

---

# 4. Why Rate Equations / Quantum Jumps Are Preferred

A full optical Bloch equation treatment would propagate a density matrix

\[
\rho
\]

with roughly

\[
N^2
\]

complex elements.

For 23-24 states, this would require hundreds of density-matrix elements per atom.

That is unnecessary for the present goals of:

- MOT capture,
- MOT loading,
- optical pumping,
- dark-state leakage,
- recoil,
- and population transfer.

For this stage, use an incoherent rate-equation / quantum-jump approach.

The model will retain populations and stochastic state transitions while discarding optical coherences.

This model is appropriate unless the simulation later needs to treat effects such as:

- coherent population trapping,
- Raman coherences,
- detailed sub-Doppler cooling,
- coherent dark states,
- Rabi oscillations,
- or other explicitly phase-coherent internal-state dynamics.

---

# 5. State Representation

Every atom should carry an internal-state label.

A convenient representation is:

```python
state = {
    "manifold": "ground" or "excited",
    "F": ...,
    "mF": ...
}
```

Alternatively, states may be represented by integer indices into precomputed arrays.

The integer-index approach will generally be faster.

Every state should contain or map to:

- manifold type,
- \(F\),
- \(m_F\),
- state energy offset,
- Landé \(g_F\),
- list of allowed outgoing transitions.

---

# 6. Hyperfine Transition Frequencies

Reference the cooling laser to the

\[
F=2 \rightarrow F'=3
\]

transition.

Define

\[
\Delta_{\rm cooling}\approx -15~{\rm MHz}.
\]

The nearby excited hyperfine manifolds must also be included because off-resonant excitation can produce leakage into \(F=1\).

Approximately,

\[
\nu_{F'=2}-\nu_{F'=3}\approx -266.650~{\rm MHz},
\]

and

\[
\nu_{F'=1}-\nu_{F'=3}\approx -423.597~{\rm MHz}.
\]

Therefore the cooling laser has approximately

\[
\Delta_{23}=-15~{\rm MHz},
\]

\[
\Delta_{22}=+251.650~{\rm MHz},
\]

\[
\Delta_{21}=+408.597~{\rm MHz}.
\]

The code should define these using a central transition-frequency table rather than hardcoding them inside the scattering engine.

---

# 7. Dark-State Physics

The nominal cooling transition

\[
F=2 \rightarrow F'=3
\]

is a closed hyperfine transition with respect to spontaneous decay because

\[
F'=3\rightarrow F=1
\]

would require

\[
\Delta F=-2,
\]

which is electric-dipole forbidden.

However, the cooling light can weakly excite

\[
F=2\rightarrow F'=2
\]

and

\[
F=2\rightarrow F'=1
\]

off resonance.

Those states can spontaneously decay into \(F=1\).

Therefore atoms can eventually be optically pumped into the dark ground-state hyperfine manifold.

The primary observable for this stage is

\[
\boxed{
P_{\rm dark}(t)
=
\sum_{m_F=-1}^{+1}P(F=1,m_F,t)
}
\]

or, in trajectory Monte Carlo form,

\[
P_{\rm dark}(t)
=
\frac{N_{F=1}(t)}{N_{\rm total}}.
\]

Once an atom reaches \(F=1\), it should be treated as dark because there is no repumper laser.

---

# 8. Electric-Dipole Selection Rules

For an electric-dipole transition:

\[
\Delta F=0,\pm1
\]

except that

\[
F=0\not\leftrightarrow F'=0.
\]

For Zeeman sublevels:

\[
\Delta m_F=q,
\]

where

\[
q=
\begin{cases}
+1 & \sigma^+,\\
0 & \pi,\\
-1 & \sigma^-.
\end{cases}
\]

Therefore,

\[
m_{F'}=m_F+q.
\]

Reject every channel for which

\[
|m_{F'}|>F'.
\]

---

# 9. Clebsch-Gordan Transition Strengths

The transition strength for

\[
|F,m_F\rangle
\rightarrow
|F',m_{F'}\rangle
\]

depends on the hyperfine reduced matrix element and the Zeeman Clebsch-Gordan coefficient.

Conceptually,

\[
\langle F'm'|d_q|Fm\rangle
\propto
\begin{pmatrix}
F' & 1 & F\\
-m' & q & m
\end{pmatrix}
\begin{Bmatrix}
J' & F' & I\\
F & J & 1
\end{Bmatrix}.
\]

The simulation needs only the squared transition strengths:

\[
C_{ge}^2
\propto
|\langle e|d_q|g\rangle|^2.
\]

Normalize the strengths to the stretched cycling transition:

\[
|F=2,m_F=+2\rangle
\rightarrow
|F'=3,m_{F'}=+3\rangle.
\]

Define

\[
C_{\rm cycling}^2=1.
\]

All Clebsch-Gordan and Wigner-symbol calculations must be performed once during initialization.

Do **not** evaluate Wigner \(3j\), Wigner \(6j\), or Clebsch-Gordan coefficients inside the trajectory loop.

---

# 10. Precomputed Transition Table

Construct an absorption-transition table containing, at minimum,

```text
ground_state_index
excited_state_index
F
mF
F_prime
mF_prime
q
C_squared
hyperfine_frequency_offset
```

Construct a spontaneous-decay table containing, at minimum,

```text
excited_state_index
ground_state_index
q
branch_weight
branch_probability
```

Each state should store a direct adjacency list of allowed outgoing transitions.

This prevents scanning all 23 states during every scattering calculation.

---

# 11. Local Magnetic Field and Quantization Axis

At each atom position calculate the magnetic field:

\[
\mathbf B(\mathbf r).
\]

For nonzero field define the local quantization axis:

\[
\hat{\mathbf b}
=
\frac{\mathbf B}{|\mathbf B|}.
\]

The local spherical basis is

\[
\mathbf e_{+1},
\qquad
\mathbf e_0,
\qquad
\mathbf e_{-1},
\]

with

\[
\mathbf e_0=\hat{\mathbf b}.
\]

The laser polarization must be projected into this local basis.

This replaces the simplified two-level treatment in which beam handedness alone determined the sign of the Zeeman-like contribution.

---

# 12. Local Polarization Decomposition

Every beam \(j\) has a complex polarization vector

\[
\boldsymbol{\epsilon}_j.
\]

Project it into the local spherical basis:

\[
a_{j,q}
=
\mathbf e_q^*
\cdot
\boldsymbol{\epsilon}_j.
\]

Then define

\[
p_{j,q}
=
|a_{j,q}|^2.
\]

These satisfy

\[
p_{j,+1}
+
p_{j,0}
+
p_{j,-1}
=
1.
\]

The interpretation is:

\[
p_{j,+1}
\rightarrow
\sigma^+ \text{ fraction},
\]

\[
p_{j,0}
\rightarrow
\pi \text{ fraction},
\]

\[
p_{j,-1}
\rightarrow
\sigma^- \text{ fraction}.
\]

A beam that is circular in its own propagation frame will generally be a mixture of all three components relative to the local magnetic-field direction.

---

# 13. Behavior Near Zero Magnetic Field

At the exact MOT center,

\[
|\mathbf B|\rightarrow 0,
\]

so the local quantization axis becomes undefined.

Use a numerical threshold

\[
B_{\epsilon}.
\]

For

\[
B>B_{\epsilon},
\]

use

\[
\hat{\mathbf b}=\mathbf B/B.
\]

For

\[
B\le B_{\epsilon},
\]

use a stable fallback strategy.

Recommended first implementation:

- retain the atom's last well-defined quantization axis.

Possible alternatives for later study:

- use a fixed laboratory axis,
- smoothly interpolate the quantization axis,
- or introduce a more complete coherent treatment near \(B=0\).

The implementation must avoid abrupt numerical singularities.

---

# 14. Zeeman Shifts

In the weak-field regime use

\[
E_{F,m_F}(B)
=
E_F^{(0)}
+
g_F\mu_Bm_FB.
\]

For an optical transition

\[
|F,m\rangle\rightarrow|F',m'\rangle,
\]

the Zeeman contribution to the transition angular frequency is

\[
\Delta\omega_Z
=
\frac{\mu_BB}{\hbar}
\left(
g_{F'}m'
-
g_Fm
\right).
\]

The effective detuning for beam \(j\) is therefore

\[
\boxed{
\delta_{j,ge}
=
\omega_L
-
\omega_{ge}^{(0)}
-
\mathbf k_j\cdot\mathbf v
-
\frac{\mu_BB}{\hbar}
\left(
g_{F'}m'
-
g_Fm
\right)
}
\]

in angular-frequency units.

Equivalent frequency-unit expressions may be used, but the code must remain internally consistent.

Do not mix Hz and rad/s.

Calculate the actual hyperfine Landé \(g_F\) values once during initialization using the standard Landé formula.

Approximate expectations are:

\[
g_{F=2}\approx +\frac12,
\]

\[
g_{F=1}\approx -\frac12,
\]

and the D2 excited hyperfine manifolds have \(g_{F'}\) values near \(2/3\).

---

# 15. Doppler Shift

The Doppler contribution must remain beam specific.

For beam \(j\),

\[
\Delta\omega_{D,j}
=
\mathbf k_j\cdot\mathbf v.
\]

The sign convention must match the existing validated two-level model.

A useful validation condition is:

- an atom moving toward a counterpropagating red-detuned beam should experience that beam as closer to resonance.

---

# 16. Laser Coupling Strength

For a transition

\[
g\rightarrow e
\]

driven by beam \(j\), define an effective transition-specific saturation parameter:

\[
s_{ge,j}
=
\frac{I_j}{I_{\rm sat,cyc}}
C_{ge}^2
p_{j,q}.
\]

Here:

- \(I_j\) is the local beam intensity,
- \(I_{\rm sat,cyc}\) is the cycling-transition saturation intensity,
- \(C_{ge}^2\) is the relative transition strength,
- \(p_{j,q}\) is the local polarization fraction.

For the Rb-87 D2 cycling transition, use approximately

\[
I_{\rm sat,cyc}
\approx
1.669~{\rm mW/cm^2}.
\]

The natural linewidth is approximately

\[
\Gamma
=
2\pi\times6.07~{\rm MHz}.
\]

The excited-state lifetime is approximately

\[
\tau
\approx
26.2~{\rm ns}.
\]

---

# 17. Laser-Driven Transition Rate

For the incoherent rate-equation model, define

\[
\boxed{
R_{ge,j}
=
\frac{\Gamma}{2}
\frac{s_{ge,j}}
{1+4\delta_{ge,j}^2/\Gamma^2}
}
\]

for each allowed laser-driven channel.

The exact implementation should preserve the rate-equation logic and should not independently apply a saturated two-level scattering denominator to every channel and then simply sum the resulting saturated rates.

Saturation should emerge from the explicit competition between:

- excitation,
- stimulated emission,
- spontaneous emission,
- and state population.

---

# 18. Event-Driven Monte Carlo / Gillespie Method

The recommended implementation is event driven.

Do not reduce the global trajectory timestep to the nanosecond scale.

For an atom in internal state \(i\), calculate all allowed outgoing rates:

\[
R_1,R_2,\ldots,R_N.
\]

Then calculate

\[
R_{\rm total}
=
\sum_c R_c.
\]

Draw a random number

\[
u\in(0,1)
\]

and sample the waiting time:

\[
\boxed{
\tau
=
-\frac{\ln u}{R_{\rm total}}
}
\]

Then choose the event channel using

\[
P(c)
=
\frac{R_c}{R_{\rm total}}.
\]

This is the standard continuous-time stochastic process / Gillespie approach.

---

# 19. Ground-State Events

If the atom is currently in

\[
|F=2,m_F\rangle,
\]

the outgoing laser channels are all allowed transitions

\[
|F=2,m_F\rangle
\rightarrow
|F',m_F'\rangle
\]

from all six cooling beams.

For every beam:

1. compute local intensity,
2. compute Doppler shift,
3. compute local polarization fractions,
4. loop only over allowed precomputed internal transitions,
5. calculate transition-specific effective detuning,
6. calculate transition-specific rate.

Then sample one excitation event from the total set of possible beam-transition channels.

If the atom is in \(F=1\), no cooling-laser excitation is performed in this model.

---

# 20. Absorption Recoil

If absorption occurs from beam \(j\),

\[
\Delta\mathbf p_{\rm abs}
=
+\hbar\mathbf k_j.
\]

Therefore

\[
\Delta\mathbf v_{\rm abs}
=
\frac{\hbar\mathbf k_j}{m_{\rm Rb}}.
\]

Then update the internal state to the selected excited state.

---

# 21. Excited-State Events

If the atom is in an excited state,

\[
|F',m_{F'}\rangle,
\]

allow two classes of outgoing events:

1. spontaneous emission,
2. stimulated emission.

The total outgoing rate is

\[
R_e
=
\Gamma
+
\sum_{g,j}R_{eg,j}.
\]

The spontaneous-emission contribution is resolved into individual final ground states using precomputed branching fractions.

The stimulated-emission contribution is resolved by beam and ground state.

Sample the next event using the same Gillespie procedure.

---

# 22. Spontaneous Emission

Allowed spontaneous decays satisfy the same dipole selection rules.

For

\[
|F',m'\rangle
\rightarrow
|F,m\rangle,
\]

the branch probability is proportional to

\[
|\langle F,m|d_q|F',m'\rangle|^2.
\]

Normalize all allowed decay branches from a given excited state such that

\[
\sum_g P(e\rightarrow g)=1.
\]

The total spontaneous decay rate from the excited state must remain

\[
\Gamma.
\]

---

# 23. Hyperfine Branching Ratios

The implementation should reproduce the known manifold-level branching behavior.

In particular:

### \(F'=3\)

\[
F'=3\rightarrow F=2
\]

with probability 1.

There must be no decay to \(F=1\).

### \(F'=2\)

The total manifold branching should satisfy approximately

\[
P(F'=2\rightarrow F=2)=\frac12,
\]

\[
P(F'=2\rightarrow F=1)=\frac12.
\]

### \(F'=1\)

The total manifold branching should satisfy approximately

\[
P(F'=1\rightarrow F=2)=\frac16,
\]

\[
P(F'=1\rightarrow F=1)=\frac56.
\]

These values must emerge from the state-resolved dipole strengths.

---

# 24. Spontaneous-Emission Recoil

For spontaneous emission, apply a recoil

\[
\Delta\mathbf p_{\rm em}
=
-\hbar k\hat{\mathbf n}.
\]

For the first implementation, choose

\[
\hat{\mathbf n}
\]

isotropically.

This is sufficient for the first state-resolved MOT model.

A later refinement may use the \(q\)-dependent dipole radiation patterns:

\[
f_{\sigma^\pm}(\theta)
=
\frac{3}{16\pi}
(1+\cos^2\theta),
\]

\[
f_{\pi}(\theta)
=
\frac{3}{8\pi}
\sin^2\theta.
\]

Do not make anisotropic spontaneous emission a prerequisite for the first implementation.

---

# 25. Entering the Dark State

If spontaneous emission places the atom in any state

\[
|F=1,m_F\rangle,
\]

set

```python
dark = True
```

and record:

- time of dark-state entry,
- position,
- velocity,
- parent excited state,
- number of photons scattered before dark-state entry.

No cooling-light scattering should occur afterward.

The atom may either:

- continue ballistically if trajectory history is desired,
- or be classified as lost if that is consistent with the capture analysis.

The behavior must be explicitly configurable.

---

# 26. Initial Internal-State Population

Use two separate initialization modes.

## 26.1 Validation Mode

For initial implementation and testing, use

\[
P(F=2)=1.
\]

Distribute the initial \(F=2\) population uniformly:

\[
P(m_F|F=2)=\frac15.
\]

This isolates optical pumping and dark-state leakage caused by the cooling light.

---

## 26.2 Vapor Mode

For a thermal, unpolarized Rb-87 vapor, the ground-state hyperfine splitting is tiny compared with \(k_BT\).

Therefore approximate the initial hyperfine populations by degeneracy:

\[
P(F=2)
\approx
\frac{5}{8},
\]

\[
P(F=1)
\approx
\frac{3}{8}.
\]

Within each hyperfine manifold, initialize the \(m_F\) states uniformly unless a different physical source preparation is specified.

Because no repumper is present, atoms initially in \(F=1\) begin dark.

This distinction must be accounted for when calculating a physically representative loading rate.

---

# 27. Optical Pumping

The simulation must naturally produce optical pumping among \(m_F\) states.

For example, under pure \(\sigma^+\) excitation on the

\[
F=2\rightarrow F'=3
\]

transition, repeated scattering should preferentially pump the atom toward

\[
|F=2,m_F=+2\rangle.
\]

The stretched transition

\[
|2,+2\rangle
\rightarrow
|3,+3\rangle
\]

should behave as a closed cycling transition when only the resonant \(F'=3\) manifold is enabled.

This provides one of the most important validation tests.

---

# 28. Relationship to the Existing Two-Level Simulation

The validated two-level model should remain available.

The multilevel model should reuse as much validated infrastructure as possible:

- MOT beam geometry,
- Gaussian beam intensities,
- anti-Helmholtz magnetic field,
- source geometry,
- atom flux sampling,
- classical trajectory integration,
- capture criteria,
- capture cross-section calculation,
- loading-rate integration.

Only the internal-state/scattering engine should be replaced.

Maintain the ability to select between:

```text
two_level
multilevel
```

through configuration.

The two-level implementation should remain a regression benchmark.

---

# 29. Recommended Software Architecture

Separate the implementation into the following modules or logical layers.

## 29.1 Atomic Structure

Responsibilities:

- define internal states,
- define hyperfine offsets,
- define \(g_F\) values,
- calculate/precompute Clebsch-Gordan coefficients,
- generate allowed transition tables,
- generate spontaneous branching tables,
- provide \(\Gamma\),
- provide \(I_{\rm sat}\),
- provide recoil parameters.

---

## 29.2 Light and Magnetic Coupling

Inputs:

\[
\mathbf r,\quad
\mathbf v,\quad
\text{internal state}.
\]

Responsibilities:

- evaluate \(\mathbf B(\mathbf r)\),
- determine the local quantization axis,
- evaluate beam intensities,
- evaluate Doppler shifts,
- decompose polarization,
- calculate Zeeman shifts,
- calculate effective detunings,
- calculate laser-driven transition rates.

---

## 29.3 Internal-State Event Engine

Responsibilities:

- gather all allowed outgoing channels,
- calculate total transition rate,
- sample event time,
- sample event type,
- update internal state,
- determine spontaneous branch,
- mark dark-state entry.

---

## 29.4 Trajectory Engine

Responsibilities:

- propagate \(\mathbf r\),
- propagate \(\mathbf v\),
- apply absorption recoil,
- apply emission recoil,
- enforce capture/loss criteria,
- record diagnostics.

---

# 30. Performance Requirements

The multilevel model must be implemented with computational efficiency as a primary design goal.

Use:

- integer state indices,
- precomputed state arrays,
- precomputed transition adjacency lists,
- precomputed Clebsch-Gordan strengths,
- precomputed branching probabilities,
- vectorized beam quantities where practical,
- event-driven internal-state dynamics,
- early exit for dark atoms.

Avoid:

- symbolic algebra inside trajectory loops,
- Wigner-symbol calculations inside trajectory loops,
- scanning all states for every event,
- nanosecond global timesteps,
- repeated allocation of large Python objects inside tight loops.

---

# 31. Quantities to Record

For each trajectory, record at minimum:

- initial position,
- initial velocity,
- initial internal state,
- final position,
- final velocity,
- capture/loss result,
- dark-state result,
- time of dark-state entry,
- position of dark-state entry,
- velocity at dark-state entry,
- number of absorption events,
- number of spontaneous emissions,
- number of stimulated emissions,
- number of photons absorbed from each beam,
- number of excitations through \(F'=1\),
- number of excitations through \(F'=2\),
- number of excitations through \(F'=3\),
- parent excited state responsible for dark-state leakage,
- number of photons scattered before entering \(F=1\).

Optional trajectory-history output:

\[
\mathbf r(t),
\]

\[
\mathbf v(t),
\]

\[
F(t),
\]

\[
m_F(t),
\]

\[
F'(t),
\]

\[
m_{F'}(t).
\]

Do not store complete time histories by default for very large Monte Carlo runs.

Use a configurable diagnostics mode.

---

# 32. Ensemble Diagnostics

The simulation should be able to calculate:

\[
P(F=2,m_F,t),
\]

\[
P(F=1,m_F,t),
\]

\[
P_{\rm dark}(t),
\]

\[
P(F',t),
\]

and

\[
\langle N_\gamma^{\rm before\,dark}\rangle.
\]

Also calculate:

- distribution of dark-state pumping times,
- distribution of dark-state pumping positions,
- distribution of photon counts before shelving,
- fraction captured before entering \(F=1\),
- fraction entering \(F=1\) before capture,
- capture cross section versus initial velocity,
- modified MOT loading rate.

---

# 33. Preliminary Validation Tests

The multilevel model must **not** immediately be connected to the full loading-rate simulation.

First implement and run isolated atomic-physics tests.

Each test should produce a clear:

```text
PASS
```

or

```text
FAIL
```

with useful diagnostic values.

The following tests are required.

---

## TEST 1 — State Count

Verify that the state generator produces:

- 5 states for \(F=2\),
- 3 states for \(F=1\),
- 3 states for \(F'=1\),
- 5 states for \(F'=2\),
- 7 states for \(F'=3\).

Expected total:

\[
23.
\]

### PASS criterion

```text
ground_states = 8
excited_states = 15
total_states = 23
```

---

## TEST 2 — Selection Rules

Generate all laser-allowed and spontaneous-emission transitions.

For every transition verify:

\[
\Delta F=0,\pm1,
\]

subject to the usual electric-dipole restrictions, and

\[
\Delta m_F=-1,0,+1.
\]

Verify that

\[
m_{F'}=m_F+q.
\]

Verify that no generated transition violates

\[
|m_F|\le F
\]

or

\[
|m_{F'}|\le F'.
\]

### Specific required checks

The transition

\[
|2,+2\rangle\rightarrow|3,+3\rangle
\]

with \(q=+1\) must exist.

The transition

\[
|2,+2\rangle\rightarrow|3,-3\rangle
\]

must not exist.

The decay

\[
F'=3\rightarrow F=1
\]

must never appear.

### PASS criterion

No forbidden channels are generated and all expected allowed channels are present.

---

## TEST 3 — Clebsch-Gordan Normalization

For every ground state and chosen excited hyperfine manifold, verify that the relative strengths of all allowed \(q\) channels are internally consistent.

Normalize the stretched cycling transition:

\[
|2,+2\rangle
\rightarrow
|3,+3\rangle
\]

to

\[
C^2=1.
\]

Verify that no transition has

\[
C^2<0
\]

or

\[
C^2>1
\]

under this normalization.

Where applicable, compare generated values against trusted tabulated Rb-87 D2 transition strengths.

### PASS criterion

- stretched cycling transition has \(C^2=1\),
- all strengths are nonnegative,
- symmetry-related channels agree where expected,
- normalization identities are satisfied to numerical tolerance.

Suggested numerical tolerance:

\[
10^{-12}
\]

for purely algebraic identities.

---

## TEST 4 — Polarization Decomposition

Choose simple magnetic-field and beam geometries with analytically obvious results.

### Case A

Let

\[
\hat{\mathbf B}=\hat{\mathbf z}
\]

and use pure \(\sigma^+\) light defined relative to \(+\hat{\mathbf z}\).

Expected:

\[
p_{+1}=1,
\]

\[
p_0=0,
\]

\[
p_{-1}=0.
\]

### Case B

Use linear polarization parallel to \(\hat{\mathbf B}\).

Expected:

\[
p_0=1.
\]

### Case C

Use linear polarization perpendicular to \(\hat{\mathbf B}\).

Expected equal or analytically appropriate decomposition into \(\sigma^+\) and \(\sigma^-\).

### Universal requirement

\[
p_{+1}+p_0+p_{-1}=1.
\]

### PASS criterion

All cases match analytic expectations to floating-point tolerance.

---

## TEST 5 — Hyperfine Branching Ratios

For every excited Zeeman state, sum all state-resolved spontaneous-emission branches.

The total must satisfy

\[
\sum_g P(e\rightarrow g)=1.
\]

Then aggregate over final hyperfine manifolds.

### \(F'=3\)

Expected:

\[
P(F'=3\rightarrow F=2)=1,
\]

\[
P(F'=3\rightarrow F=1)=0.
\]

### \(F'=2\)

Expected:

\[
P(F'=2\rightarrow F=2)\approx\frac12,
\]

\[
P(F'=2\rightarrow F=1)\approx\frac12.
\]

### \(F'=1\)

Expected:

\[
P(F'=1\rightarrow F=2)\approx\frac16,
\]

\[
P(F'=1\rightarrow F=1)\approx\frac56.
\]

### PASS criterion

Numerical branching ratios agree with these expected manifold values to a tight tolerance.

---

## TEST 6 — Spontaneous-Decay Normalization

For every excited state

\[
|F',m_{F'}\rangle,
\]

verify that the sum of all spontaneous decay rates equals

\[
\Gamma.
\]

That is,

\[
\sum_g\Gamma_{e\rightarrow g}
=
\Gamma.
\]

### PASS criterion

Relative error smaller than a chosen strict threshold such as

\[
10^{-12}.
\]

---

## TEST 7 — Stretched-State Cycling

Disable \(F'=1\) and \(F'=2\).

Use only the resonant

\[
F=2\rightarrow F'=3
\]

manifold.

Initialize the atom in

\[
|F=2,m_F=+2\rangle.
\]

Illuminate it with pure \(\sigma^+\) light.

The only absorption channel should be

\[
|2,+2\rangle
\rightarrow
|3,+3\rangle.
\]

The excited stretched state should decay only back to

\[
|2,+2\rangle.
\]

### Expected behavior

The atom should cycle indefinitely:

\[
|2,+2\rangle
\leftrightarrow
|3,+3\rangle.
\]

It should never:

- enter another \(m_F\) state,
- enter \(F=1\),
- or leave the stretched cycle.

### PASS criterion

After a large number of simulated photon cycles, for example

\[
10^5
\]

events, all events remain in the stretched cycle.

---

## TEST 8 — Optical Pumping Toward the Stretched State

Again enable only

\[
F=2\rightarrow F'=3.
\]

Use pure \(\sigma^+\) light.

Initialize an ensemble uniformly across

\[
m_F=-2,-1,0,+1,+2.
\]

Allow repeated absorption and spontaneous emission.

### Expected behavior

The population should migrate toward

\[
m_F=+2.
\]

At sufficiently long time,

\[
P(F=2,m_F=+2)
\]

should approach 1 in the idealized single-beam pumping configuration.

### PASS criterion

The \(m_F=+2\) population increases monotonically in ensemble expectation and dominates the late-time distribution.

---

## TEST 9 — Reverse Optical Pumping

Repeat the previous test with pure \(\sigma^-\) light.

Expected late-time state:

\[
|F=2,m_F=-2\rangle.
\]

### PASS criterion

The population is pumped toward \(m_F=-2\).

This is an important sign-convention test.

---

## TEST 10 — Zeeman-Sign Test

Choose a simple uniform magnetic field

\[
\mathbf B=B\hat{\mathbf z}.
\]

Set

\[
\mathbf v=0.
\]

Calculate transition frequencies for known channels.

For example compare:

\[
|2,+2\rangle\rightarrow|3,+3\rangle
\]

and

\[
|2,-2\rangle\rightarrow|3,-3\rangle.
\]

Verify that reversing

\[
B\rightarrow -B
\]

reverses the sign of the Zeeman contribution.

Also verify that reversing both

\[
B\rightarrow -B
\]

and

\[
m_F\rightarrow -m_F
\]

restores the appropriate symmetry.

### PASS criterion

All Zeeman shifts agree with the analytic formula

\[
\Delta\omega_Z
=
\frac{\mu_BB}{\hbar}
(g_{F'}m'-g_Fm).
\]

---

## TEST 11 — Doppler-Sign Regression

Disable Zeeman shifts.

Use one pair of counterpropagating cooling beams.

Give the atom a known velocity along the beam axis.

Verify that the beam opposing the atomic velocity moves closer to resonance for red-detuned light.

This must reproduce the already validated two-level Doppler-sign behavior.

### PASS criterion

Correct beam has the larger scattering rate.

---

## TEST 12 — Zero-Field Stability

Propagate atoms across the MOT center where

\[
|\mathbf B|\approx0.
\]

Verify that:

- no NaNs appear,
- no divide-by-zero occurs,
- polarization fractions remain normalized,
- the chosen fallback quantization-axis strategy behaves continuously enough for numerical stability.

### PASS criterion

No numerical instability or discontinuous state explosion occurs.

---

## TEST 13 — Dark-State Leakage Disabled

Enable only

\[
F'=3.
\]

Initialize all atoms in \(F=2\).

Run many scattering events.

### Expected behavior

Because

\[
F'=3\rightarrow F=1
\]

is forbidden,

\[
P_{\rm dark}(t)=0
\]

for all times.

### PASS criterion

Zero atoms enter \(F=1\) to machine precision / exact discrete counting.

---

## TEST 14 — Dark-State Leakage Enabled

Enable

\[
F'=1,2,3.
\]

Use the real hyperfine detunings.

Initialize all atoms in \(F=2\).

Use the real cooling-laser detuning.

### Expected behavior

Rare off-resonant excitation through \(F'=2\) and \(F'=1\) should eventually produce nonzero population in \(F=1\).

Therefore

\[
P_{\rm dark}(t)
\]

should increase with time.

### PASS criterion

- dark population remains exactly zero if \(F'=1,2\) are disabled,
- dark population becomes nonzero when they are enabled,
- recorded dark-state entry events trace back only to allowed \(F'=1\) or \(F'=2\) decay channels.

---

## TEST 15 — Dark-State Parent-Manifold Test

Record the parent excited hyperfine state for every dark-state transition.

Expected:

\[
F'=3
\]

must contribute zero dark-state events.

Only

\[
F'=1
\]

and

\[
F'=2
\]

may contribute.

### PASS criterion

```text
dark_from_Fp3 = 0
```

for any simulation size.

---

## TEST 16 — Off-Resonant Suppression

For fixed position and velocity, compare excitation rates from \(F=2\) into:

\[
F'=3,
\]

\[
F'=2,
\]

\[
F'=1.
\]

At a cooling detuning of approximately \(-15\) MHz from \(F'=3\), the \(F'=3\) excitation should dominate strongly.

### PASS criterion

The rate ordering is physically sensible:

\[
R_{F'=3}\gg R_{F'=2}\gg R_{F'=1}
\]

for representative states with nonzero coupling.

The exact ratio depends on Clebsch-Gordan strengths and polarization.

---

## TEST 17 — Event-Time Statistics

Choose a state with a constant known total outgoing rate

\[
R_{\rm total}.
\]

Generate many event waiting times.

The mean should satisfy

\[
\langle \tau\rangle
=
\frac{1}{R_{\rm total}}.
\]

The distribution should be exponential.

### PASS criterion

Sample mean agrees with the analytic expectation within Monte Carlo uncertainty.

---

## TEST 18 — Event-Channel Statistics

Choose a state with two artificial or controlled outgoing channels:

\[
R_1,\qquad R_2.
\]

Expected probabilities:

\[
P_1
=
\frac{R_1}{R_1+R_2},
\]

\[
P_2
=
\frac{R_2}{R_1+R_2}.
\]

Generate a large number of events.

### PASS criterion

Observed fractions agree with expected probabilities within statistical uncertainty.

---

## TEST 19 — Excited-State Lifetime

Disable stimulated emission.

Initialize atoms in a selected excited state.

The sampled spontaneous lifetime distribution must have mean

\[
\langle\tau\rangle
=
\Gamma^{-1}
\approx 26.2~{\rm ns}.
\]

### PASS criterion

Monte Carlo mean agrees with the expected lifetime within statistical uncertainty.

---

## TEST 20 — Recoil Magnitude

For every absorption event verify:

\[
|\Delta\mathbf p_{\rm abs}|
=
\hbar k.
\]

For every spontaneous emission event verify:

\[
|\Delta\mathbf p_{\rm em}|
=
\hbar k.
\]

### PASS criterion

Momentum-kick magnitudes agree with the analytic value to numerical tolerance.

---

## TEST 21 — Two-Level Limit Regression

Artificially reduce the multilevel system to one ground state and one excited state with:

\[
C^2=1,
\]

one allowed polarization channel, and the same

\[
\Gamma,\quad
I_{\rm sat},\quad
\Delta,\quad
\mathbf k.
\]

Compare the resulting scattering behavior against the validated two-level implementation.

### PASS criterion

The multilevel engine reproduces the two-level model in the appropriate limiting case.

This is a crucial regression test.

---

# 34. Preliminary Test Output

Create a test runner that prints a concise summary such as:

```text
MULTILEVEL MOT VALIDATION
-------------------------

[PASS] State count
[PASS] Selection rules
[PASS] CG normalization
[PASS] Polarization decomposition
[PASS] Hyperfine branching ratios
[PASS] Spontaneous decay normalization
[PASS] Stretched-state cycling
[PASS] Sigma+ optical pumping
[PASS] Sigma- optical pumping
[PASS] Zeeman sign
[PASS] Doppler sign
[PASS] Zero-field stability
[PASS] Dark leakage disabled
[PASS] Dark leakage enabled
[PASS] Dark-state parent manifold
[PASS] Off-resonant suppression
[PASS] Gillespie waiting-time statistics
[PASS] Gillespie channel statistics
[PASS] Excited-state lifetime
[PASS] Recoil magnitude
[PASS] Two-level regression

21 / 21 tests passed
```

If a test fails, print useful numerical details.

Do not merely return a Boolean.

---

# 35. Recommended Development Order

Implement the multilevel model in the following order.

## Stage A — Atomic Structure Only

Implement:

- state generation,
- \(g_F\),
- hyperfine offsets,
- Clebsch-Gordan strengths,
- selection rules,
- spontaneous branching tables.

Run Tests 1-6.

Do not integrate trajectories yet.

---

## Stage B — Polarization and Zeeman Physics

Implement:

- local quantization axis,
- spherical basis,
- beam polarization decomposition,
- Zeeman shift,
- Doppler shift.

Run Tests 4, 10, 11, and 12.

---

## Stage C — Event Engine

Implement:

- laser excitation rates,
- Gillespie waiting-time sampling,
- event-channel selection,
- spontaneous decay,
- stimulated emission.

Run Tests 17-19.

---

## Stage D — Optical Pumping

Implement:

- state updates,
- repeated scattering,
- stretched-state cycling,
- optical pumping.

Run Tests 7-9.

---

## Stage E — Dark-State Leakage

Enable off-resonant

\[
F'=1,2
\]

channels.

Implement:

- \(F=1\) shelving,
- dark-state counters,
- dark-state history.

Run Tests 13-16.

---

## Stage F — Recoil and Trajectory Coupling

Reconnect the validated trajectory engine.

Run:

- recoil test,
- two-level-limit regression,
- simple MOT trajectory comparisons.

---

## Stage G — Full Capture Simulation

Only after all preliminary tests pass:

- resimulate capture trajectories,
- calculate capture cross sections,
- calculate dark-state survival probability,
- calculate multilevel capture velocity,
- recalculate loading rate.

---

# 36. Expected Physical Outcome Without Repumper

The no-repumper system is not expected to behave as a true steady-state experimental MOT.

Initially bright atoms in \(F=2\) can scatter many cooling photons.

However, weak off-resonant excitation through

\[
F'=2
\]

and

\[
F'=1
\]

will eventually transfer some atoms into

\[
F=1.
\]

Once there, they are dark to the cooling light.

The key physical question for this stage is therefore:

\[
\boxed{
\text{How much cooling and trapping occurs before hyperfine optical pumping shelves the atom in }F=1?
}
\]

The simulation should quantify:

- characteristic dark-state pumping time,
- photons scattered before shelving,
- distance traveled before shelving,
- probability of capture before shelving,
- effect on capture cross section,
- effect on loading rate.

---

# 37. Future Repumper Extension

The software architecture must make a repumper straightforward to add later.

A repumper laser will add laser-driven transitions out of

\[
F=1.
\]

For example, one may later drive a transition such as

\[
F=1\rightarrow F'=2.
\]

The same general transition graph should then allow:

\[
F=1
\rightarrow
F'
\rightarrow
F=1
\]

or

\[
F=1
\rightarrow
F'
\rightarrow
F=2.
\]

Do not write the no-repumper model in a way that makes \(F=1\) permanently special at the architecture level.

Instead, define whether lasers couple to \(F=1\) through configuration.

For the present stage:

```text
repumper_enabled = False
```

and therefore \(F=1\) behaves as dark.

---

# 38. Final Implementation Requirement

The completed multilevel model should represent:

\[
\boxed{
\begin{array}{l}
\text{Classical 3D trajectory}\\
+\\
\text{23-state Rb-87 D2 hyperfine/Zeeman system}\\
+\\
\text{local }\sigma^+/\pi/\sigma^-\text{ decomposition}\\
+\\
\text{state-resolved Clebsch-Gordan strengths}\\
+\\
\text{hyperfine + Doppler + Zeeman detuning}\\
+\\
\text{stochastic excitation and stimulated emission}\\
+\\
\text{state-resolved spontaneous decay}\\
+\\
\text{photon recoil}\\
+\\
F=1\text{ dark-state tracking}
\end{array}
}
\]

The implementation should use:

\[
\boxed{
\text{incoherent rate equations + stochastic quantum jumps}
}
\]

rather than full optical Bloch equations.

The validated two-level simulation must remain available as a regression benchmark.

No full capture/loading calculation should be trusted until the atomic-physics validation suite passes.

---

# 39. Codex Execution Guidance

When implementing this specification:

1. Inspect the existing repository before modifying code.
2. Reuse the existing validated beam, field, trajectory, and loading-rate infrastructure.
3. Create the multilevel atomic-physics system as a modular extension.
4. Preserve the existing two-level model.
5. Add configuration that selects between the two-level and multilevel engines.
6. Implement the preliminary atomic-physics tests before coupling the new model to full MOT loading calculations.
7. Run the tests.
8. Print a clear summary of passed and failed tests.
9. If any required preliminary validation test fails, do not proceed to full loading-rate calculations until the failure is understood and corrected.
10. Keep the implementation computationally efficient enough for large Monte Carlo trajectory ensembles.

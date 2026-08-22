# EFFICIENT_MOT.md
## Adiabatic-Elimination Rate-Equation Approach for Efficient Multilevel MOT Simulation (Rb-87, F=1,2 → F'=1,2,3)

### Purpose

This document describes how to convert a photon-by-photon (fully stochastic, event-driven)
Monte Carlo simulation of a Rb-87 magneto-optical trap into a much faster simulation that
retains the full multilevel atomic structure (F=1,2 ground manifold; F'=1,2,3 excited
manifold; Zeeman sublevels; repumper) but replaces explicit photon-event tracking with a
**quasi-steady-state internal population distribution**, evaluated at each trajectory
timestep using **generalized multilevel rate equations** (populations only — no coherences).

This mirrors the structure of a simpler deterministic 2-level MOT model (mean force +
diffusion, integrated over position/velocity with a fixed timestep), generalized to the
full multilevel system via a small linear system solve per timestep (steady-state
populations across 23 sublevels) instead of a scalar saturated-scattering-rate formula.

**Design decision (explicit):** This implementation uses the **rate-equation approximation**
(populations/diagonal density-matrix elements only), NOT the full optical Bloch equations.
Zeeman coherences and coherent population trapping are therefore not captured — this
implementation will reproduce Doppler-cooling-scale trapping dynamics, F=1/F=2 optical
pumping, and repumper action correctly, but will NOT reproduce sub-Doppler (Sisyphus-type)
cooling or dark-state coherence effects from polarization-gradient beams. This is an
intentional, accepted trade-off for computational speed; do not add coherence terms back in.

---

### 1. Physical Justification (Why This Works)

The internal atomic state (populations and coherences across the 8 ground + 15 excited
Zeeman sublevels) equilibrates via Rabi flopping, spontaneous decay, and optical pumping on
a timescale of order Γ⁻¹ to a few Γ⁻¹, where Γ = 2π × 6.065 MHz is the D2 natural linewidth
(~26 ns, so equilibration in ~tens to ~hundreds of ns). The external motion (position,
velocity) evolves on timescales set by the damping time, trap oscillation period, or
beam-crossing time — typically tens of microseconds to milliseconds.

Because these timescales are separated by 3–4 orders of magnitude, the internal state can be
treated as being in **quasi-steady-state** at every instant of the external trajectory. This
eliminates the need to stochastically sample individual absorption/emission events: instead,
solve directly for the steady-state sublevel populations at the atom's current (position,
velocity, local B field, beam detunings), extract the mean force and a diffusion coefficient
from them, and integrate the external motion with a Langevin equation — the same architecture
as the existing deterministic 2-level model, with a richer per-step internal solve (a 23×23
linear system for populations instead of a scalar rate).

The rate-equation approximation is valid whenever optical coherences decay fast compared to
the rate at which populations change — true whenever detunings and Rabi frequencies are not
so large/fast that coherent (Rabi-oscillation-scale) dynamics matter for the force, i.e. away
from extreme saturation and away from near-degenerate dark-state configurations. This is the
standard "generalized rate equation" treatment of multilevel laser cooling (see Ungar, Weiss,
Riis & Chu, J. Opt. Soc. Am. B 6, 2058 (1989), for the canonical multilevel-rate-equation MOT
treatment) and it is the method to be implemented here.

---

### 2. Basis and Precomputed Quantities (compute once, offline)

**Ground manifold (5S₁/2, J=1/2, I=3/2) — 8 states:**
- F=1: mF = −1, 0, 1
- F=2: mF = −2, −1, 0, 1, 2

**Excited manifold (5P₃/2, J=3/2, I=3/2) — 15 states:**
- F'=1: mF' = −1, 0, 1
- F'=2: mF' = −2, −1, 0, 1, 2
- F'=3: mF' = −3, −2, −1, 0, 1, 2, 3

Total: 23-level system (8 ground + 15 excited).

**Dipole matrix elements** between |F,mF⟩ and |F',mF'⟩ via spherical component q ∈ {−1,0,+1}:

```
<F'||d||F> = (-1)^(F+Jg+1+I) * sqrt((2F'+1)(2Jg+1)) * Wigner6j(Jg, Je, 1, F', F, I)

<F',mF'| d_q |F,mF> = <F'||d||F> * (-1)^(F'-mF') * Wigner3j(F', 1, F; -mF', q, mF)
```

with Jg=1/2, Je=3/2, I=3/2 for Rb-87 D2. Compute these once using `sympy.physics.wigner`
(`wigner_3j`, `wigner_6j`) and cache as a fixed array `D[q, e_idx, g_idx]`. This tensor
encodes both the laser-coupling strengths (used to build per-transition Rabi frequencies,
Section 3.1) and the spontaneous-emission branching ratios (used to build the decay rate
matrix, Section 3.3) — it should NOT be recomputed per timestep.

Reference for numerical values / sanity checks: Steck, *Rubidium 87 D Line Data*.

---

### 3. Explicit Rate-Matrix ("Rate-Equation Liouvillian") Construction

The rate-equation method replaces the full 23×23 density matrix (with coherences) by a
23-element **population vector** `p = [p_g1 ... p_g8, p_e1 ... p_e15]`, and replaces the full
Liouvillian superoperator by a 23×23 **real rate matrix** `R` acting only on populations:

```
dp/dt = R(x, v, t) · p
```

`R` contains three physically distinct kinds of terms, all built from the same precomputed
dipole tensor `D[q, e_idx, g_idx]` from Section 2 — nothing new needs to be computed offline
beyond what Section 2 already provides.

#### 3.1 Per-transition Rabi frequency and detuning ("Hamiltonian" inputs)

For every (ground sublevel g, excited sublevel e, beam b) triplet with nonzero dipole
coupling, compute:

```python
# Local quantization axis / field
Bvec = B_field_fn(x)
Bmag = np.linalg.norm(Bvec)

# Zeeman shift of each sublevel (rad/s)
zeeman_g = np.array([gF_ground[F] * mu_B * Bmag * mF / hbar for (F, mF) in GROUND])
zeeman_e = np.array([gF_excited[Fp] * mu_B * Bmag * mFp / hbar for (Fp, mFp) in EXCITED])

# For each beam b: local polarization decomposition into (sigma-, pi, sigma+) = q(-1,0,+1)
# in the LOCAL B-frame (rotate lab-frame polarization vector into frame defined by Bvec)
eps_q_b = rotate_polarization(beam_b.polarization_labframe, Bvec)   # complex 3-vector

# Doppler shift for this beam
doppler_b = np.dot(beam_b.k_vector, v)

# Rabi frequency for transition (g -> e) via beam b, polarization component q:
#   Omega_{g,e,b} = eps_q_b[q] * Omega0_b(x) * D[q, e_idx, g_idx]
# where Omega0_b(x) is the beam's peak Rabi-frequency scale at position x,
# Omega0_b(x) = Gamma * sqrt(I_b(x) / (2 * I_sat)) (standard two-level convention,
# applied per-transition here via the D tensor as the relative coupling strength)

# Detuning for this specific transition:
#   Delta_{g,e,b} = beam_b.laser_detuning - doppler_b - (zeeman_e[e_idx] - zeeman_g[g_idx])
```

This step is the direct analog of "building H" in the OBE approach — it is exactly the same
physical inputs (Zeeman shifts, Doppler shifts, local polarization decomposition, dipole
coupling strengths). The only difference is what happens next: instead of assembling these
into an off-diagonal Hamiltonian and solving a coherence-containing master equation, they are
plugged directly into a **saturated two-level scattering-rate formula**, transition by
transition.

#### 3.2 Stimulated excitation/de-excitation rates (off-diagonal entries of R)

For each (g, e, b) triplet, compute the single-transition Lorentzian scattering rate using
the standard two-level saturated-scattering formula, evaluated with that transition's own
Rabi frequency and detuning:

```python
def transition_rate(Omega_geb, Delta_geb, Gamma):
    # Excitation rate g -> e driven by beam b (population scattered per unit time)
    s_geb = 2 * Omega_geb**2 / Gamma**2          # transition-specific saturation parameter
    return (Gamma / 2) * s_geb / (1 + s_geb + (2*Delta_geb/Gamma)**2)

R_excite = np.zeros((N_E, N_G))   # rate FROM ground g TO excited e
for gi in range(N_G):
    for ei in range(N_E):
        rate_sum = 0.0
        for beam in beams:
            for qi in range(3):
                Omega = eps_q_b[qi] * Omega0(beam, x) * D[qi, ei, gi]
                if abs(Omega) < 1e-12: continue
                Delta = beam.laser_detuning - np.dot(beam.k_vector, v) \
                        - (zeeman_e[ei] - zeeman_g[gi])
                rate_sum += transition_rate(abs(Omega), Delta, Gamma)
        R_excite[ei, gi] = rate_sum
```

**Important:** when multiple beams (or multiple polarization components) drive the *same*
(g,e) transition simultaneously, sum their **Rabi frequencies coherently within each q
component** before computing the saturation parameter (i.e. sum `Omega` contributions from
different beams at the same q before squaring), not simply sum independently-computed rates,
if those beams are mutually coherent (e.g. counter-propagating beams from the same laser).
If beams are independently phase-randomized/incoherent in your model (a common and acceptable
MOT approximation), summing rates directly (as above) is the standard practice and is fine —
just be explicit in code comments about which assumption is being made, since it affects the
saturation parameter at high intensity.

#### 3.3 Spontaneous decay rates (the other off-diagonal entries of R)

```python
Gamma_total = 2*np.pi*6.065e6   # D2 natural linewidth (rad/s)

R_decay = np.zeros((N_G, N_E))  # rate FROM excited e TO ground g
branching_norm = np.zeros(N_E)
for ei in range(N_E):
    branching_norm[ei] = sum(D[qi, ei, gi]**2 for qi in range(3) for gi in range(N_G))
for ei in range(N_E):
    for gi in range(N_G):
        branch = sum(D[qi, ei, gi]**2 for qi in range(3)) / branching_norm[ei]
        R_decay[gi, ei] = Gamma_total * branch
```

This encodes the correct Clebsch-Gordan-weighted branching ratios automatically, including
decay from F'=1,2,3 back into F=1 (dark to cooling light, requires repumper) and F=2.

#### 3.4 Assembling the full rate matrix and solving for steady state

```python
R = np.zeros((N_TOT, N_TOT))
R[N_G:, :N_G] = R_excite          # ground -> excited (stimulated absorption)
R[:N_G, N_G:] = R_decay           # excited -> ground (spontaneous decay)
# stimulated emission (excited -> ground, induced by the same beams) uses the same
# transition_rate() value as R_excite[e,g] by detailed balance in the two-level
# saturated-rate formula already used above — no separate term needed beyond R_excite,
# since that formula already represents the net two-level cycling rate.

# Diagonal: total outflow rate from each level (negative, for population conservation)
for i in range(N_TOT):
    R[i, i] = -np.sum(R[:, i][np.arange(N_TOT) != i])

# Steady state: solve R @ p = 0 subject to sum(p) = 1
A = R.copy()
A[0, :] = 1.0                      # replace one equation with normalization
b = np.zeros(N_TOT); b[0] = 1.0
p_ss = np.linalg.solve(A, b)
```

This is a single **real, 23×23 linear solve** per atom per timestep — substantially cheaper
than the complex 529-dimensional (23²) linear solve required for the full density-matrix
Liouvillian in the OBE approach, and it involves no complex arithmetic at all.

---

### 4. Force and Diffusion from the Steady State

**Mean force** (deterministic drift term). With populations only (no coherences), the
per-beam scattering rate is recovered directly from the same per-transition rates computed
in Section 3.2, evaluated at `p_ss`:

```python
def compute_force(p_ss, beams, k_vectors_per_beam):
    F = np.zeros(3)
    for beam in beams:
        # scattering rate attributable to this beam = sum over (g,e) of the beam's
        # contribution to R_excite[e,g], weighted by ground-state population p_ss[g]
        R_b = beam_specific_rate_matrix(beam, x, v, zeeman_g, zeeman_e, D, Gamma)  # N_E x N_G
        scatter_rate_b = np.sum(R_b @ p_ss[:N_G])
        F += hbar * beam.k_vector * scatter_rate_b
    return F
```

i.e., the scattering rate contributed by each beam (computed from that beam's own share of
`R_excite`, dotted into the steady-state ground populations) times ħk for that beam, summed
over all beams. This is the direct rate-equation analog of reading excited-state populations
off the diagonal of ρ_ss in the OBE approach — here it is a population dot-product instead.

**Diffusion coefficient** (semiclassical estimate, for stochastic recoil heating):

```
D(x, v) ≈ (ħk)^2 * (total photon scattering rate) / 2
```

summed appropriately over contributing beams. This does not require simulating individual
recoil events — it is computed analytically from the steady-state scattering rate.

**Langevin integration of external motion** (direct replacement for the existing 2-level
deterministic integrator — same structure, richer force/diffusion functions):

```
dp = F(x, v) * dt + sqrt(2 * D * dt) * randn()
dx = (p / m) * dt
```

---

### 5. Implementation / Performance Notes

- **Precompute once and cache:** dipole matrix elements, Clebsch-Gordan/Wigner coefficients,
  Landé g-factors, the spontaneous-decay rate matrix `R_decay` (Section 3.3), and the sparse
  structure (nonzero pattern) of the rate matrix `R`. Only per-transition Rabi frequencies,
  detunings, and Zeeman shifts (functions of x, v, t) should be recomputed per timestep.
- **Vectorize across the atom ensemble:** since the simulation will run many atoms for
  statistics, batch the steady-state solves across atoms rather than looping atom-by-atom.
  numpy broadcasting is a reasonable starting point; JAX (`vmap`, JIT) or GPU batched linear
  algebra is recommended if further speedup is needed.
- **Timestep size:** can likely match the timestep used in the existing 2-level deterministic
  model (microsecond scale), since individual photon events are no longer being resolved —
  just ensure the timestep remains small relative to the damping/trap-oscillation timescale
  being resolved.
- **Adiabaticity check:** the quasi-steady-state assumption can break down for atoms crossing
  B=0 fast enough that the internal state cannot adiabatically track the changing
  quantization axis. This is a corner case (relevant mainly for high-velocity atoms passing
  very near the trap center) — worth a sanity check against a small number of full
  photon-by-photon trajectories, but should not affect the bulk of typical MOT capture
  velocities (~1–20 m/s) and gradients (~10 G/cm).
- **Validation strategy:** before replacing the existing photon-by-photon code, validate the
  steady-state-solve approach against a subset of trajectories run both ways (old
  photon-by-photon method vs. new adiabatic-elimination method) to confirm force,
  temperature, and capture-velocity agreement, particularly in regimes with strong
  polarization gradients or near B=0.

---

### 6. Minimal Function-Level Structure to Implement

```
precompute_dipole_tensor()          -> D[q, e_idx, g_idx]      (once, at startup)
precompute_branching_ratios(D)      -> R_decay[g_idx, e_idx]   (once, at startup)

per timestep, per atom:
    B = local_field(x)
    R_excite = build_stimulated_rate_matrix(x, v, t, B, beams, D, Gamma)  # Sec 3.1-3.2
    R = assemble_rate_matrix(R_excite, R_decay)                          # Sec 3.4
    p_ss = steady_state_populations(R)                                   # Sec 3.4
    F = compute_force(p_ss, beams)                                       # Sec 4
    Dcoef = compute_diffusion(p_ss, beams)                                # Sec 4
    p_momentum, x = langevin_step(p_momentum, x, F, Dcoef, dt)
```

Note on the adiabaticity check (Section 5): under the rate-equation approximation this check
is doubly important, since the method additionally assumes coherences are negligible at all
times, not just that populations track adiabatically. If validation against photon-by-photon
trajectories reveals systematic disagreement (particularly near B=0 or in strongly
polarization-gradient regions), that is the expected signature of missing coherence physics,
not a bug — it indicates the rate-equation approximation itself is breaking down in that
regime, and should be documented as a known limitation rather than debugged as an error.

This structure is a drop-in replacement for the internal-state-tracking portion of the
existing photon-by-photon code, while preserving the outer trajectory-integration loop
(timestep, duration, logging) from the original 2-level deterministic model.

# Population Rate Equation in the Multilevel MOT Code

**pMOT Monte Carlo Project**  
**September 18, 2026**

## 1. What a population rate equation is

A population rate equation is a reduced description of an atom's internal quantum state. Instead of tracking the full wavefunction amplitudes or the full density matrix, it tracks only the probability that the atom occupies each internal level. In this project those levels are the hyperfine-Zeeman sublevels of the $^{87}\mathrm{Rb}$ D2 line.

The state vector is therefore not a position or velocity vector; it is a vector of internal-state probabilities,

\[
\mathbf p =
\begin{pmatrix}
p_{g_1},\ldots,p_{g_{N_g}},
p_{e_1},\ldots,p_{e_{N_e}}
\end{pmatrix}^{T},
\]

where $g_i$ are ground hyperfine-Zeeman states and $e_j$ are excited hyperfine-Zeeman states.

The central approximation is that only populations are retained. In density-matrix language, the diagonal elements $\rho_{ii}$ are kept and interpreted as populations, while the off-diagonal elements $\rho_{ij}$, $i\neq j$, are discarded. Those off-diagonal elements are optical and ground-state coherences. They are required for phase-sensitive physics such as Rabi oscillations, coherent dark states, coherent population trapping, Raman coherences, standing-wave interference, and sub-Doppler polarization-gradient cooling. A population rate equation does not represent those effects.

The population dynamics are written as

\[
\frac{d\mathbf p}{dt}
=
R(\mathbf r,\mathbf v)\,\mathbf p,
\]

where $R$ is a real rate matrix. Its off-diagonal entries are transition rates moving population from one internal state to another. Its diagonal entries are negative outflow rates, chosen so that total population is conserved.

In the current multilevel MOT implementation, the internal state is not time-integrated during each external-motion step. Instead, the code assumes that optical pumping equilibrates much faster than mechanical motion and solves the steady-state problem

\[
R(\mathbf r,\mathbf v)\,\mathbf p_{\mathrm{ss}}=0,
\qquad
\sum_i p_{\mathrm{ss},i}=1.
\]

This is the adiabatic-elimination or quasi-steady-state approximation for the internal state.

## 2. Internal basis used by the current code

The authoritative implementation is in:

- `src/pmot/mot_multilevel/rate_equations.py`
- `src/pmot/mot_multilevel/atomic_structure.py`

It uses 24 internal states:

- **8 ground states**
  - $5S_{1/2}, F=1, m_F=-1,0,+1$
  - $5S_{1/2}, F=2, m_F=-2,-1,0,+1,+2$
- **16 excited states**
  - $5P_{3/2}, F'=0,1,2,3$
  - with all allowed $m_F'$ values in each manifold.

Older design text sometimes describes a 23-state model because the first cooling-only specification omitted $F'=0$. The current production graph keeps $F'=0$, because the repumper includes the dipole-allowed $F=1\rightarrow F'=0$ channel.

## 3. Precomputed atomic structure

The code constructs all allowed electric-dipole transitions once in `build_atomic_structure()`.

For each transition it stores:

- the ground-state index;
- the excited-state index;
- $F,m_F$ and $F',m_F'$;
- spherical polarization component
  \[
  q=m_F'-m_F\in\{-1,0,+1\};
  \]
- normalized line strength $C^2$;
- excited-manifold hyperfine offset.

The transition strengths are computed from Wigner $3j$ and $6j$ factors and normalized to the cycling transition

\[
|F=2,m_F=+2\rangle
\rightarrow
|F'=3,m_F'=+3\rangle.
\]

The code also precomputes spontaneous-decay channels. For every excited state, the allowed decay branches into $F=1$ and $F=2$ ground states are weighted by the same dipole strengths and normalized so that the total spontaneous decay rate out of each excited state is $\Gamma$.

## 4. Stimulated transition rates

At a given atom position $\mathbf r$ and velocity $\mathbf v$, the ordinary multilevel MOT path evaluates the anti-Helmholtz magnetic field and uses its direction as the local quantization axis. Each laser beam polarization is then decomposed into local spherical components $\sigma^-$, $\pi$, $\sigma^+$.

For each beam $b$ and allowed transition $g\rightarrow e$, the code computes a transition-specific saturation parameter

\[
s_{eg}^{(b)}
=
\frac{I_b(\mathbf r)}{I_{\mathrm{sat}}}
C_{eg}^{2}\,w_{bq},
\]

where $I_b(\mathbf r)$ is the local Gaussian beam intensity, $C_{eg}^{2}$ is the normalized dipole strength, and $w_{bq}$ is the local spherical-polarization weight for the transition's required $q$.

The effective detuning used by the code is, in angular-frequency units,

\[
\Delta_{eg}^{(b)}
=
\Delta_b
-
\delta_{\mathrm{HFS},eg}
-
\mathbf k_b\cdot\mathbf v
-
\frac{\mu_{eg}^{(Z)}|\mathbf B|}{\hbar}
-
\Delta_{eg}^{(\mathrm{extra})}.
\]

For the conventional multilevel MOT,

\[
\Delta_{eg}^{(\mathrm{extra})}=0.
\]

The pMOT branch reuses the same local environment entry point and may supply an additional transition-resonance shift.

The stimulated rate used in the current implementation is a saturated Lorentzian,

\[
W_{eg}^{(b)}
=
\frac{\Gamma}{2}
\frac{s_{eg}^{(b)}}
{1+s_{eg}^{(b)}+4(\Delta_{eg}^{(b)})^2/\Gamma^2}.
\]

The implementation treats different beams as incoherent rate contributors. It does not coherently sum optical fields before squaring. This is a standard fast MOT approximation, but it means that coherent standing-wave and phase-sensitive saturation effects are absent.

## 5. Rate-matrix construction in the current code

The function `build_beam_stimulated_rate_matrices()` returns an array

```text
W[b, e, g]
```

containing the stimulated rate contributed by each beam.

Cooling beams address ground $F=2$ states. Repump beams, when enabled, address ground $F=1$ states. The current repumper graph permits

\[
F=1\rightarrow F'=0,1,2.
\]

The beam matrices are summed:

\[
W_{\mathrm{tot}}[e,g]
=
\sum_b W[b,e,g].
\]

The full real population Liouvillian $R$ is then assembled in `assemble_rate_matrix()`:

\[
R=
\begin{pmatrix}
R_{gg} & R_{ge}\\
R_{eg} & R_{ee}
\end{pmatrix}.
\]

The implemented block structure is

\[
R_{eg}[e,g]
=
W_{\mathrm{tot}}[e,g],
\]

\[
R_{ge}[g,e]
=
W_{\mathrm{tot}}[e,g]
+
\Gamma_{\mathrm{decay}}[g,e].
\]

Thus the same laser-driven rate is used in reverse as a stimulated-emission population link, and spontaneous decay is added independently.

Finally, the diagonal entries are set column-by-column to the negative total outflow rate:

\[
R_{ii}
=
-\sum_{j\neq i}R_{ji}.
\]

This enforces population conservation.

## 6. Steady-state solve

The function `steady_state_populations()` solves

\[
R\mathbf p_{\mathrm{ss}}=0,
\qquad
\sum_i p_{\mathrm{ss},i}=1.
\]

Numerically, the code scales the matrix by $\Gamma$, replaces one row with the normalization condition, and calls `numpy.linalg.solve`.

If the matrix is singular, it falls back to a least-squares solve. Roundoff-scale negative populations are clipped to zero, and the population vector is renormalized.

The resulting $\mathbf p_{\mathrm{ss}}$ is interpreted as the internal-state distribution that would be reached by fast optical pumping at the current $(\mathbf r,\mathbf v)$.

## 7. Force and diffusion

The code computes each beam's scattering contribution by weighting that beam's stimulated rates by the steady-state ground populations:

\[
\Gamma_b^{(\mathrm{sc})}
=
\sum_{e,g}
W[b,e,g]\,p_{\mathrm{ss}}(g).
\]

The mean optical force is then

\[
\mathbf F
=
\sum_b
\hbar\mathbf k_b\,\Gamma_b^{(\mathrm{sc})}.
\]

This is a radiation-pressure force from absorption momentum. Gravity is added separately during trajectory integration when enabled. The stored force is the optical force, not including gravity.

The code also computes a scalar recoil-diffusion coefficient,

\[
D
=
\frac{1}{2}
\sum_b
|\hbar\mathbf k_b|^2
\Gamma_b^{(\mathrm{sc})}.
\]

When trajectory diffusion is enabled, this coefficient drives a Langevin random momentum kick.

This is an approximate recoil-heating model, not an explicit photon-by-photon emission-direction simulation and not a full diffusion tensor.

## 8. External trajectory execution

The function `simulate_rate_equation_trajectory()` advances external motion with a fixed timestep.

At each step it:

1. evaluates the local magnetic field;
2. chooses the local quantization axis;
3. builds the beam-resolved stimulated rates;
4. assembles and solves the population rate equation;
5. computes the mean optical force and diffusion coefficient;
6. updates mechanical momentum using optical force and gravity;
7. optionally adds a Langevin recoil-diffusion kick;
8. updates velocity and position;
9. terminates if the atom escapes the configured radius.

The trajectory update is effectively

\[
\mathbf p_{\mathrm{mech}}(t+\Delta t)
=
\mathbf p_{\mathrm{mech}}(t)
+
[\mathbf F_{\mathrm{opt}}(\mathbf r,\mathbf v)+m\mathbf g]\Delta t
+
\sqrt{2D\Delta t}\,\boldsymbol\xi,
\]

when diffusion is enabled, where $\boldsymbol\xi$ is a vector of independent standard normal random variables.

Then

\[
\mathbf v=\frac{\mathbf p_{\mathrm{mech}}}{m},
\qquad
\mathbf r(t+\Delta t)=\mathbf r(t)+\mathbf v\Delta t.
\]

Capture and loading studies commonly disable diffusion so classifications are deterministic and reproducible. Temperature studies enable diffusion and interpret the result as a finite-time Langevin temperature diagnostic.

## 9. Physical fidelity and limitations

The current population-rate model captures important MOT-scale physics:

- Rb-87 D2 hyperfine and Zeeman sublevel structure;
- Clebsch-Gordan-dependent coupling strengths;
- optical pumping among Zeeman sublevels;
- leakage into $F=1$;
- repumper return from $F=1$;
- spontaneous-decay branching;
- Doppler shifts;
- Zeeman shifts from the anti-Helmholtz field;
- local polarization decomposition relative to the local quantization axis;
- mean radiation-pressure force;
- approximate recoil diffusion for Langevin trajectories.

It is therefore much more realistic than a two-level MOT model for capture, loading, force-curve, optical-pumping, and repumper-dependence studies.

However, it is not a full optical Bloch equation model. It omits:

- optical coherences;
- ground-state coherences;
- coherent dark states;
- coherent population trapping;
- sub-Doppler and Sisyphus cooling;
- Rabi oscillations;
- coherent standing-wave interference;
- phase-dependent beam correlations;
- a full tensor recoil-diffusion model.

The model is best understood as a fast, state-resolved, Doppler-scale, incoherent optical-pumping MOT model. It is physically defensible when the internal populations relax much faster than the external motion and when coherence effects are not the dominant source of force or temperature. It is not sufficient for final quantitative claims about sub-Doppler temperatures, coherent dark-state physics, or pMOT dynamics requiring a full state-resolved AC-Stark Hamiltonian.

## 10. Current status for this project

For the conventional multilevel MOT, this rate-equation engine is the authoritative production model, but quantitative claims remain provisional until the relevant timestep, duration, and event-engine comparison checks are documented for the regime being reported.

For the pMOT, the same rate-equation engine is reused as the 780-nm cooling and repump radiation-pressure kernel. The pMOT-specific 1529-nm trapping-light physics is not fully represented by this population-rate equation.

The current pMOT branch still requires a validated state-resolved AC-Stark Hamiltonian, conservative Stark force, trapping-light scattering/heating/loss, coherent interference treatment, and physical handling of the fictitious-field zero before quantitative trapping claims are justified.

## 11. Understanding Verification

We have a system of 24 populations: 8 ground-state levels ($5S_{1/2}$) and 16 excited-state levels ($5P_{3/2}$).

The cooling and repumper lasers can induce transitions between these levels. The population rate equation tracks the amount of atoms in each level at a given time or equivalently the probability of a single atom to be in any level at a given time.

If we assume internal dynamics equilibrate faster than the macroscopic mechanical motion, we can solve the steady-state problem

\[
\frac{d\mathbf p}{dt}
=
0
=
M(\mathbf r,\mathbf v)\mathbf p_{\mathrm{ss}}
\tag{1}
\]

where $\mathbf p$ is the internal-state probabilities vector which satisfies

\[
\sum_i p_{\mathrm{ss},i}=1.
\]

\[
\mathbf p=
\begin{pmatrix}
\mathbf p_g\\
\mathbf p_e
\end{pmatrix},
\qquad
[\mathbf p]=24\times1,
\]

\[
[\mathbf p_g]=8\times1,
\qquad
[\mathbf p_e]=16\times1.
\tag{2}
\]

$R$ is the rate matrix, whose diagonal entries are negative population outflow rates leaving the state and whose off-diagonal elements represent transition rates between states. $R$ is $24\times24$ dimensional and is symmetric:

\[
M_{\mu\nu}=M_{\nu\mu}.
\tag{3}
\]

It is symmetric because the stimulated absorption and emission is induced by the same laser, which provides the same atom-light interaction Hamiltonian.

For each beam $b$ and allowed transition $(g\leftrightarrow e)$, compute the pairwise stimulated transition coefficient:

\[
W_{ge}^{(b)}
=
\frac{
\Gamma_{ge}
|\Omega_{ge}^{(b)}|^2
}{
\Gamma_{ge}^2
+
4(\Delta_{ge}^{(b)})^2
}.
\tag{4}
\]

\[
\Delta_{ge}^{(b)}
\equiv
\Delta_{\mathrm{eff}}
=
\Delta_{\mathrm{laser}}
-
\mathbf k\cdot\mathbf v
-
\Delta_{Z,ge}.
\tag{5}
\]

\[
\Omega_{ge}^{(b)}
=
-\frac{1}{\hbar}
\langle e|\mathbf d\cdot\mathbf E_b|g\rangle
=
-\frac{E_b}{\hbar}
\sum_{q=-1}^{+1}
\epsilon_{b,q}
\langle F'm_F'|d_q|Fm_F\rangle.
\tag{6}
\]

Per beam,

\[
\text{Absorption Rate:}\qquad
W_{ge}^{(b)}p_g,
\]

\[
\text{Stimulated Emission Rate:}\qquad
W_{ge}^{(b)}p_e,
\]

\[
\text{Spontaneous Emission Rate:}\qquad
\Gamma_e p_e.
\tag{7}
\]

$W_{ge}$ is derived from the optical Bloch equations by setting the coherences equal to $0$.

The rate matrix $M$ is populated by $W_{ge}$, meaning it is a $24\times24$ dimensional matrix:

\[
M=
\begin{pmatrix}
-D_g & W^T+\Gamma\\
W & -D_e
\end{pmatrix},
\qquad
[M]
=
\begin{pmatrix}
8\times8 & 8\times16\\
16\times8 & 16\times16
\end{pmatrix}
=
24\times24.
\tag{8}
\]

Dimensionality and definitions of the various blocks are as follows:

1. $D_g$: diagonal block, where the diagonal entries are the negative population flow rates out of the ground states. Dim: $8\times8$.
2. $D_e$: diagonal block, where the diagonal entries are the negative population flow rates out of the excited states. Dim: $16\times16$.
3. $W$: the transition coefficient, connecting ground to excited states. This represents stimulated absorption, bringing a ground state into an excited state. Dim: $16\times8$.
4. $W^T$: the transpose of the transition coefficient, connecting excited to ground states. This represents stimulated emission, bringing an excited state into a ground state. Dim: $8\times16$.
5. $\Gamma$: spontaneous decay/emission rates. Dim: $8\times16$.

The $D_g$ and $D_e$ are chosen such that the sum of each column of $M$ equates to $0$, conserving total population number.

### 11.1 Steady State

What does "steady state" mean in this language? We already defined the situation to be

\[
\frac{d\mathbf p}{dt}
=
M\mathbf p_{\mathrm{ss}}
=
0.
\]

So physically it means the population vector is unchanging. But the system is still dynamic. The only manner in which it can occur is if the flow of population into a state is equal to the flow out of a state:

\[
\left.\frac{d\mathbf p}{dt}\right|_{\mathrm{into}}
=
\left.\frac{d\mathbf p}{dt}\right|_{\mathrm{out}}.
\]

Rates are obtained by multiplying $W$ with $\mathbf p$ as shown in Equation (7). But this is exactly $M\mathbf p$ since $M$ is composed of the $W$ configurations.

And since

\[
M\mathbf p
=
\frac{d\mathbf p}{dt}
\]

during steady state, this is the statement that the flow of populations into and out of states equilibrates into such a state where the populations are unchanging.

So, solving for what the system looks like when

\[
\frac{d\mathbf p}{dt}=0
\]

is identical to solving the linear algebra problem

\[
M\mathbf p_{\mathrm{ss}}=0,
\]

specifically the nonzero eigenvector which satisfies this equation.

As an example, let ground state $|g;1\rangle$ connect to excited states $|e;5\rangle$, $|e;6\rangle$, $|e;7\rangle$ via stimulated absorption, with the reverse true through stimulated emission and spontaneous emission.

Then, in steady state,

\[
0
=
\sum_b
W_{ge}^{(b)}p_e
-
W_{ge}^{(b)}p_g
-
\Gamma_e^{(b)}p_e
=
\text{Total abs. rate + stim. em. rate + spont. em. rate}.
\tag{9}
\]

We can also write this as

\[
\dot p_e
=
W_{ge}(p_g-p_e)-\Gamma p_e,
\]

\[
\dot p_g
=
-W_{ge}(p_g-p_e)+\Gamma p_e
=
-\dot p_e.
\tag{10}
\]

### 11.2 Some Derivations

#### 11.2.1 $W$ Abbreviated

Consider a Hamiltonian that bridges ground and excited states via laser coupling:

\[
H_{\mathrm{int}}
=
-\frac{\hbar\Omega}{2}
\left(
|e\rangle\langle g|
+
|g\rangle\langle e|
\right),
\tag{11}
\]

where $\Omega$ is the Rabi frequency.

Define the dimensionless saturation parameter

\[
s
=
\frac{2|\Omega|^2}{\Gamma^2}.
\tag{12}
\]

Now construct the stimulated-state transition coefficient:

\[
W_{ge}
=
\frac{\Gamma|\Omega_{ge}|^2}
{\Gamma^2+4\Delta_{ge}^2}
=
\frac{1}{\Gamma}
\frac{|\Omega_{ge}|^2}
{1+4\Delta_{ge}^2/\Gamma^2}
=
\frac{\Gamma}{2}
\frac{s}
{1+4\Delta_{ge}^2/\Gamma^2}.
\tag{13}
\]

This strongly resembles the scattering rate for a two-level atom:

\[
R_{\mathrm{sc}}^{(2\text{-level})}
=
\frac{\Gamma}{2}
\frac{s}
{1+s_{\mathrm{tot}}+4\Delta^2/\Gamma^2}.
\tag{14}
\]

The transition coefficient for the multilevel atom is merely missing that $s_{\mathrm{tot}}$ term in the denominator. So the two expressions are coincident in the low-intensity regime.

#### 11.2.2 Two-Level Atom Scattering Rate

Consider the limit of a two-level atom, where there exists a single ground state and single excited state.

Steady state:

\[
\dot p_e
=
0
=
W_{ge}(p_g-p_e)-\Gamma p_e.
\tag{15}
\]

In this case,

\[
p_g=1-p_e
\]

since they are the only two possible states. Then,

\[
W_{ge}(1-2p_e)-\Gamma p_e=0
\]

so

\[
p_e
=
\frac{W_{ge}}{2W_{ge}+\Gamma}.
\tag{16}
\]

The spontaneous photon scattering rate is given by the decay rate multiplied by the population in the excited state:

\[
R_{\mathrm{sc}}
=
\Gamma p_e.
\]

Let $W_{ge}:=W$. Then

\[
R_{\mathrm{sc}}
=
\frac{\Gamma W}{2W+\Gamma}
=
\frac{\Gamma}{2}
\frac{s}
{1+s+4\Delta^2/\Gamma^2}.
\tag{17}
\]

#### 11.2.3 $\Omega_{ge}^{(b)}$ Derivation

The interaction Hamiltonian is

\[
\hat H_{\mathrm{int}}
=
-\mathbf d\cdot\mathbf E_b(\mathbf r,t)
=
-\frac{1}{2}
\boldsymbol\epsilon_b E_b^+
e^{+i(\mathbf k\cdot\mathbf r-\omega_b t)}
-
\frac{1}{2}
\boldsymbol\epsilon_b E_b^-
e^{-i(\mathbf k\cdot\mathbf r-\omega_b t)}.
\tag{18}
\]

Now equate this Hamiltonian to that describing the interaction between two states coupled by a driving laser. The Rabi frequency is $\Omega$ and defines the rate at which the atomic population flips between the ground and excited states:

\[
H_{\mathrm{int}}
=
-\frac{\hbar\Omega_{ge}^{(b)}}{2}
|e\rangle\langle g|e^{-i\omega_b t}
-
\frac{\hbar\Omega_{ge}^{(b)*}}{2}
|g\rangle\langle e|e^{i\omega_b t}.
\tag{19}
\]

Then, upon $\hat H=\hat H_{\mathrm{int}}$,

\[
\Omega_{ge}^{(b)}
=
-\frac{E_b\epsilon_b}{\hbar}
\langle g|\mathbf d|e\rangle
e^{i\mathbf k\cdot\mathbf r}.
\tag{20}
\]

Therefore,

\[
|\Omega_{ge}^{(b)}|^2
=
\Omega\Omega^*
=
\frac{E_b^2\epsilon_b^2}{\hbar^2}
|\langle g|\mathbf d|e\rangle|^2.
\tag{21}
\]

such that

\[
|\Omega|^2
=
\sum_b |\Omega^{(b)}|^2.
\]

Here, $E$ is the electric-field magnitude and $\epsilon$ is the polarization.

As usual,

\[
E_b
=
\sqrt{\frac{2I_b}{c\epsilon_0}}.
\]

#### 11.2.4 $W_{ge}$ Full Derivation

The full density-matrix equations, including populations and coherences, are

\[
\dot\rho_{ee}
=
-\Gamma\rho_{ee}
+
\frac{i}{2}
\left(
\Omega\rho_{ge}
-
\Omega^*\rho_{eg}
\right),
\]

\[
\dot\rho_{ge}
=
-\left(
\frac{\Gamma}{2}
+
i\Delta
\right)\rho_{ge}
+
\frac{i\Omega^*}{2}
(\rho_{ee}-\rho_{gg}).
\tag{22}
\]

They come from the quantum master equation for the density operator:

\[
\dot\rho
=
-\frac{i}{\hbar}[H,\rho]
+
\mathcal L[\rho]
=
(\text{Quantum Evolution})
+
(\text{Irreversible Spontaneous Decay}).
\tag{23}
\]

We use the rotating-frame Hamiltonian in the basis $(|g\rangle,|e\rangle)$ and the standard density matrix:

\[
H
=
\hbar
\begin{pmatrix}
0 & -\Omega^*/2\\
-\Omega/2 & -\Delta
\end{pmatrix},
\qquad
\rho
=
\begin{pmatrix}
\rho_{gg} & \rho_{ge}\\
\rho_{eg} & \rho_{ee}
\end{pmatrix}.
\tag{24}
\]

The spontaneous decay term comes from defining the Lindblad term. For a decay $|e\rangle\rightarrow|g\rangle$, occurring with rate $\Gamma$, define the Lindblad jump operator:

\[
\mathcal L_\Gamma[\rho]
\equiv
\mathcal L[\rho]
=
L\rho L^\dagger
-
\frac{1}{2}
\left(
L^\dagger L\rho
+
\rho L^\dagger L
\right).
\tag{25}
\]

Evaluating its contributing matrix elements yields

\[
\dot\rho_{ee,\mathrm{spont}}
=
-\Gamma\rho_{ee},
\]

\[
\dot\rho_{gg,\mathrm{spont}}
=
+\Gamma\rho_{ee}.
\tag{26}
\]

In the population-rate model, we make the approximation that the optical coherences relax much faster than the populations themselves, so we eliminate it:

\[
\dot\rho_{ge}=0.
\tag{27}
\]

Now set the equation for $\dot\rho_{ge}$ equal to zero and rearrange for $\rho_{ge}$:

\[
\left(
\frac{\Gamma}{2}
+
i\Delta
\right)\rho_{ge}
=
\frac{i\Omega^*}{2}
(\rho_{ee}-\rho_{gg}).
\tag{28}
\]

\[
\rho_{ge}
=
\frac{i\Omega^*}{2}
\frac{\rho_{ee}-\rho_{gg}}
{\Gamma/2+i\Delta}
=
\frac{i\Omega^*}{2}
(\rho_{ee}-\rho_{gg})
\frac{\Gamma/2-i\Delta}
{(\Gamma/2)^2+\Delta^2}.
\tag{29}
\]

Using

\[
\rho_{eg}
=
\rho_{ge}^*,
\]

insert this into the equation for $\dot\rho_{ee}$:

\[
\dot\rho_{ee}
=
\frac{\Gamma|\Omega|^2}
{\Gamma^2+4\Delta^2}
(\rho_{gg}-\rho_{ee})
-
\Gamma\rho_{ee}.
\tag{30}
\]

Equivalently,

\[
\dot p_e
=
W_{ge}(p_g-p_e)
-
\Gamma p_e,
\tag{31}
\]

where

\[
W_{ge}
=
\frac{\Gamma|\Omega_{ge}|^2}
{\Gamma^2+4\Delta_{ge}^2}
\]

is identified as the pairwise stimulated-transition coefficient, with units

\[
[W]=\mathrm{s}^{-1}.
\]

Per beam,

\[
W_{ge}^{(b)}
=
\frac{
\Gamma_{ge}
|\Omega_{ge}^{(b)}|^2
}{
\Gamma_{ge}^2
+
4(\Delta_{ge}^{(b)})^2
}.
\tag{32}
\]

where

\[
\Gamma_{ge}
=
\frac{\omega^3}
{3\pi\epsilon_0\hbar c^3}
|\langle g|\mathbf d|e\rangle|^2
\tag{33}
\]

and $\omega$ is taken to be the frequency gap between the considered ground $|g\rangle$ and excited $|e\rangle$ states.

Lastly,

\[
W_{ge}
=
\sum_b W_{ge}^{(b)}.
\tag{34}
\]

### 11.3 Force for the Numerical Solver

In this Monte Carlo simulation, we require a force to be applied to the atom at the subsequent timestep. These rates are only part of the picture. The fourth-order Runge-Kutta numerical solver requires force.

A force is a change in momentum over time. By solving the steady-state equations, we have unearthed the rates of stimulated absorption and emission and spontaneous emission.

Rates have units of inverse time. So, multiplying these rates by momentum will yield a force. The momentum imparted to the atom comes from the photons it scatters.

So, the force per beam is represented as

\[
\mathbf F_b
=
\hbar\mathbf k_b
\sum_{ge}
W_{ge}(p_g-p_e).
\tag{35}
\]

Total net vectorial force is then

\[
\mathbf F
=
\sum_b\mathbf F_b.
\tag{36}
\]

Interestingly, since

\[
W_{ge}(p_g-p_e)
=
\Gamma p_e,
\]

the force can also be written as

\[
\mathbf F_b
=
\hbar\mathbf k_b\Gamma p_e
\tag{37}
\]

despite the spontaneous emission itself contributing a zero net force. Although this might only be true in the one-beam case, do not use this formula to generally calculate the force.

Thus, a MOT can have a steady internal-state distribution while simultaneously exerting a nonzero damping force on the atom.

### 11.4 Pseudocode Flow

1. **Define the 24 internal states.**

   Use the 8 ground states of $5S_{1/2}$,

   \[
   F=1,2,
   \]

   and the 16 excited states of $5P_{3/2}$,

   \[
   F'=0,1,2,3.
   \]

   Fix a consistent ordering so every matrix index maps to one $|F,m_F\rangle$ state.

2. **At each atom position, define the local quantization axis.**

   Compute the MOT magnetic field $\mathbf B(\mathbf r)$ and take

   \[
   \hat z_{\mathrm{quant}}\parallel\mathbf B.
   \]

   Then express every cooling and repumper beam in that common local basis as $\sigma^+$, $\pi$, and $\sigma^-$ components.

3. **Precompute all allowed dipole matrix elements.**

   For every allowed

   \[
   |g\rangle\leftrightarrow|e\rangle
   \]

   transition, calculate

   \[
   d_{eg}^{(q)}
   =
   \langle e|d_q|g\rangle.
   \]

   ARC can provide these. These are atomic constants, so this only needs to be done once.

4. **For every beam $b$, calculate the local Rabi frequency.**

   Use

   \[
   \Omega_{eg}^{(b)}
   =
   \frac{E_b}{\hbar}
   \epsilon_{b,q}
   d_{eg}^{(q)},
   \]

   where

   \[
   E_b
   =
   \sqrt{
   \frac{2I_b(\mathbf r)}
   {c\epsilon_0}
   }.
   \]

   This incorporates beam intensity, polarization, and transition strength.

5. **Calculate the effective detuning for every beam-transition pair.**

   Include laser detuning, Doppler shift, and Zeeman shift:

   \[
   \Delta_{eg}^{(b)}
   =
   \Delta_{\mathrm{laser}}
   -
   \mathbf k_b\cdot\mathbf v
   -
   \Delta_{Z,eg}
   =
   \Delta_{\mathrm{laser}}
   -
   \mathbf k_b\cdot\mathbf v
   -
   \frac{\mu_B|\mathbf B|}{\hbar}
   \left(
   g_{F'}m_F'-g_Fm_F
   \right).
   \]

6. **Calculate the pairwise stimulated-transition coefficient.**

   For each beam,

   \[
   W_{eg}^{(b)}
   =
   \frac{
   \Gamma
   |\Omega_{eg}^{(b)}|^2
   }{
   \Gamma^2
   +
   4(\Delta_{eg}^{(b)})^2
   }.
   \]

   Do this for both cooling and repumper beams. Do not use the two-level saturated scattering formula containing $1+s_{\mathrm{tot}}$.

7. **Sum all laser contributions for each state pair.**

   Define

   \[
   W_{eg}
   =
   \sum_b W_{eg}^{(b)}.
   \]

   This gives the $16\times8$ stimulated-transition matrix $W$.

8. **Build the $24\times24$ population-rate matrix $M$.**

   For every allowed ground-excited pair,

   \[
   M_{e,g}\mathrel{+}=W_{eg}
   \]

   for stimulated absorption,

   \[
   M_{g,e}\mathrel{+}=W_{eg}
   \]

   for stimulated emission,

   and add spontaneous branching,

   \[
   M_{g,e}\mathrel{+}=\Gamma_{e\rightarrow g}.
   \]

   The diagonal entries are the negative total rates leaving each state.

9. **Check probability conservation.**

   Every column must satisfy

   \[
   \sum_i M_{ij}=0.
   \]

   This is an essential debugging check.

10. **Solve for the steady-state populations.**

    At each phase-space point $(\mathbf r,\mathbf v)$, solve

    \[
    M\mathbf p_{\mathrm{ss}}=0,
    \qquad
    \sum_i p_i=1.
    \]

    Since $M$ is singular, replace one row of $M$ with the normalization condition and solve the resulting linear system.

11. **Calculate the force from each beam.**

    Use the beam-resolved coefficients, not the summed matrix $W$:

    \[
    \mathbf F_b
    =
    \hbar\mathbf k_b
    \sum_{g,e}
    W_{eg}^{(b)}
    (p_g-p_e).
    \]

    The $W_{eg}^{(b)}p_g$ term represents stimulated absorption, while $W_{eg}^{(b)}p_e$ represents stimulated emission.

12. **Sum all beam forces.**

    \[
    \mathbf F
    =
    \sum_b\mathbf F_b.
    \]

    Spontaneous emission gives zero average force, so it does not contribute directly to the mean force.

13. **Update the atom's classical motion.**

    Use

    \[
    \dot{\mathbf v}
    =
    \frac{\mathbf F}{m},
    \qquad
    \dot{\mathbf r}
    =
    \mathbf v.
    \]

    Then recompute $M$, $\mathbf p_{\mathrm{ss}}$, and $\mathbf F$ at the new phase-space point.

The core simulation loop is therefore

\[
(\mathbf r,\mathbf v)
\rightarrow
\Omega,\Delta
\rightarrow
W^{(b)}
\rightarrow
M
\rightarrow
\mathbf p_{\mathrm{ss}}
\rightarrow
\mathbf F
\rightarrow
(\mathbf r',\mathbf v').
\]

## 12. MOT

At each $5\,\mu\mathrm{s}$ time step, evaluate $\mathbf r$, $\mathbf v$, $\hat{\mathbf B}$, which is the direction of the quantization axis, and $|\mathbf B|$.

\[
\Delta_{ge}^{(b)}
\equiv
\Delta_{\mathrm{eff}}
=
\Delta_{\mathrm{laser}}
-
\mathbf k\cdot\mathbf v
-
\Delta_{Z,ge}
=
\Delta_{\mathrm{laser}}
-
\mathbf k\cdot\mathbf v
-
\frac{\mu_B|\mathbf B|}{\hbar}
\left(
g_{F'}m_F'-g_Fm_F
\right).
\tag{38}
\]

\[
\Omega_{ge}^{(b)}
=
-\frac{1}{\hbar}
\langle e|\mathbf d\cdot\mathbf E_b|g\rangle
=
-\frac{E_b}{\hbar}
\sum_{q=-1}^{+1}
\epsilon_{b,q}
\langle F'm_F'|d_q|Fm_F\rangle.
\tag{39}
\]

\[
\Gamma_{ge}
=
\frac{\omega^3}
{3\pi\epsilon_0\hbar c^3}
|\langle g|\mathbf d|e\rangle|^2,
\qquad
\Gamma_e
=
\sum_g\Gamma_{ge}
\equiv
\sum_g\Gamma_{e\rightarrow g}.
\tag{40}
\]

where $\omega$ is taken to be the frequency gap between the considered ground $|g\rangle$ and excited $|e\rangle$ states.

\[
W_{ge}^{(b)}
=
\frac{
\Gamma_e
|\Omega_{ge}^{(b)}|^2
}{
\Gamma_e^2
+
4(\Delta_{ge}^{(b)})^2
}.
\tag{41}
\]

\[
W_{ge}
=
\sum_b W_{ge}^{(b)}.
\tag{42}
\]

\[
M_{\mu\nu}
=
\begin{pmatrix}
-D_g & W^T+\Gamma\\
W & -D_e
\end{pmatrix},
\qquad
[M_{\mu\nu}]
=
\begin{pmatrix}
8\times8 & 8\times16\\
16\times8 & 16\times16
\end{pmatrix}
=
24\times24.
\tag{43}
\]

\[
\sum_\mu M_{\mu\nu}=0
\quad
\forall\nu,
\qquad
M_{\mu\nu}\mathbf p_{\mathrm{ss}}=0,
\qquad
\sum_i p_{\mathrm{ss},i}=1.
\tag{44}
\]

\[
\mathbf F_b
=
\hbar\mathbf k_b
\sum_{ge}
W_{ge}^{(b)}
(p_g-p_e).
\tag{45}
\]

\[
\mathbf F
=
\sum_b\mathbf F_b.
\tag{46}
\]

\[
\dot{\mathbf v}
=
\frac{\mathbf F}{m_{\mathrm{atom}}}.
\tag{47}
\]

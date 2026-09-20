# Provisional differential AC-Stark pMOT model

## Status

This document records the first executable AC-Stark detuning layer for the
no-coil pMOT. The cooling and repump dynamics are the authoritative 24-state,
repumper-enabled, adiabatic population-rate equations. The pMOT supplies an
explicitly zero external magnetic field, a local optical-spin basis, and a
transition-resonance shift. The population solve, saturation law, optical
pumping, scattering rates, recoil diffusion coefficient, and 780-nm
radiation-pressure force are otherwise unchanged.

The present Stark layer is a **provisional diagnostic**, not a production
24-state Hamiltonian. The supplied CSV contains only one differential scalar,
vector, and tensor value at each wavelength. It does not contain separate
polarizabilities for the 8 ground and 16 excited hyperfine-Zeeman states. That
information is sufficient for the stretched cycling-transition reference, but
not for a unique shift of every other channel.

## Three-dimensional trapping-light Doppler shift

For trapping component (j), the atom-frame frequency and wavelength use the
same nonrelativistic convention as the multilevel MOT:

\[
\nu'_j=\nu_j\left(1-\frac{\hat{\mathbf k}_j\cdot\mathbf v}{c}\right),
\qquad
\lambda'_j=
\frac{\lambda_j}{1-\hat{\mathbf k}_j\cdot\mathbf v/c}.
\]

The full three-dimensional dot product is evaluated independently for all six
trapping components. A component parallel to the atomic velocity is red
shifted in the atomic frame, its counterpropagating partner is blue shifted,
and a perpendicular component is unchanged to first order. The resulting six
wavelengths are interpolated only in the narrow
`Arora_CCSD_Differential_Polarizabilities.csv` table. Extrapolation and silent
clipping are forbidden.

## General shift that a production model must calculate

For a level \(|F,m\rangle\), component intensity (I_j), normalized complex
polarization \(\boldsymbol\epsilon_j\), and quantization axis
\(\hat{\mathbf n}\), the diagonal expression is

\[
U_{Fm}^{(j)}=-\frac{2I_j}{c\epsilon_0}
\left[
\alpha_F^{(0)}
+\alpha_F^{(1)}
\left(i\boldsymbol\epsilon_j^*\!\times\boldsymbol\epsilon_j\right)
\cdot\hat{\mathbf n}\frac{m}{F}
+\alpha_F^{(2)}
\left(3|\boldsymbol\epsilon_j\cdot\hat{\mathbf n}|^2-1\right)
\frac{3m^2-F(F+1)}{2F(2F-1)}
\right].
\]

Vector and tensor terms vanish for (F=0). A production implementation must
have separate upper- and lower-level polarizabilities, sum the non-collinear
light-shift operators, diagonalize the local Hamiltonian, and transform the
780-nm dipole couplings into that eigenbasis. The current CSV cannot supply
those inputs.

## Implemented stretched-reference proxy

The CSV is converted to coefficients
\(\beta_0,\beta_1,\beta_2\) in MHz per
`mW/(100 um)^2`, where one intensity unit is
\(I_u=10^5\ \mathrm{W/m^2}\). At 1529.268881 nm these are approximately

\[
(\beta_0,\beta_1,\beta_2)
=(-53.054435,+136.334045,+53.053782).
\]

The stored tensor coefficient already contains the transverse-circular
geometry and stretched-state factor (1/2). For each component define
\(D_j=I_j/I_u\) and

\[
\mathbf Q=\sum_j \beta_1(\lambda'_j)D_j s_j\hat{\mathbf k}_j,
\]

where (s=-1) for propagation-frame `sigma+` and (s=+1) for `sigma-`.
Away from \(\mathbf Q=0\), the provisional quantization axis is
\(\hat{\mathbf n}=\mathbf Q/|\mathbf Q|\). At the zero, the previous
well-defined axis is retained as a numerical regularization.

For rate-equation transition (t:g\rightarrow e\), the implemented terms are

\[
\delta\nu_t^{(0)}=\sum_j\beta_0(\lambda'_j)D_j,
\]

\[
\delta\nu_t^{(1)}=
\frac{g_{F'_e}m_e-g_{F_g}m_g}
{3g_{F'=3}-2g_{F=2}}
\mathbf Q\cdot\hat{\mathbf n},
\]

and

\[
\delta\nu_t^{(2)}=
\sum_j\beta_2(\lambda'_j)D_j
\left[-2
\left(3|\boldsymbol\epsilon_j\cdot\hat{\mathbf n}|^2-1\right)
\frac{3m_e^2-F'_e(F'_e+1)}
{2F'_e(2F'_e-1)}
\right].
\]

The tensor term is set to zero for (F'_e=0). The vector rescaling imposes
the intended Zeeman-like transition pattern, and the tensor expression uses an
excited-state angular proxy while neglecting the unavailable separate ground
tensor response. Both are explicit approximations outside the stretched
cycling channel.

Internally the calculation is performed as energy in joules. The total
transition frequency shifts are

\[
\delta\nu_t^{\rm AC}=
\frac{\Delta E_t^{(0)}+\Delta E_t^{(1)}+\Delta E_t^{(2)}}{h},
\qquad
\delta\omega_t^{\rm AC}=2\pi\delta\nu_t^{\rm AC}.
\]

## Effective detuning and force

For cooling or repump beam (b), the implemented angular-frequency detuning is

\[
\boxed{
\Delta_{b,t}^{\rm eff}
=\Delta_{L,b}-\delta_{{\rm HFS},t}
-\mathbf k_b\cdot\mathbf v
-\frac{\Delta E_t^{\rm AC}}{\hbar}
}
\]

with \(\mathbf B_{\rm external}=\mathbf0\). A positive shifted transition
resonance is therefore subtracted from the laser detuning. The unchanged
multilevel kernel's stimulated rate and absorption-momentum force are

\[
W_{b,t}=\frac{\Gamma}{2}
\frac{s_{b,t}}{1+s_{b,t}+4(\Delta_{b,t}^{\rm eff}/\Gamma)^2},
\qquad
\mathbf F_{780}=\sum_b\hbar\mathbf k_bR_b.
\]

Here the existing kernel defines
\(R_b=\sum_{e,g}W_{b,e,g}p_g\), a ground-population-weighted available
absorption rate. This is not the spontaneous scattering rate
\(\Gamma\sum_ep_e\). The inherited solver simultaneously uses saturated
per-transition rates and explicit reverse stimulated links in its population
matrix. Those conventions have not yet been jointly validated against a
consistent two-level limit or the event engine. Consequently, the plotted
force magnitude and even its sign in strongly saturated regions remain a
solver-level uncertainty shared with the multilevel MOT; they cannot establish
quantitative pMOT trapping. Correcting that closure requires a separate
multilevel-solver re-derivation and rerun, not a pMOT-only subtraction.

Gravity is added in the external trajectory integration, not in the plotted
optical force.

## First diagnostic configuration and result

The first run used 27 mW per cooling beam, 0.1 mW per repump beam, no external
field, deterministic external motion, and a 5-us timestep. The trapping path
helicities were `sigma+`, `sigma+`, and `sigma-` on x, y, and z, respectively,
for both incident and retro components. This orientation matches the restoring
signs of the retained multilevel cooling beams.

Because no physical trapping power was supplied, the run used a provisional
stretched-reference calibration to a 20 G/cm vector-gradient magnitude. The
ideal geometry gives about 522.27 G/cm per watt on a path, corresponding to
38.294486 mW launched on each path (114.883458 mW total incident power). This
is a scale choice, not a power recommendation.

At the origin, the six trapping components produce approximately
30,797.97 W/m2 total intensity. Their vector sum is zero and their tensor proxy
cancels by three-axis symmetry, while the scalar cycling-reference shift is
-16.339691 MHz. The nominal -15 MHz cooling detuning therefore becomes
approximately +1.339691 MHz for the stretched cycling reference at the exact
center.

The resulting static radiation force is anti-restoring infinitesimally around
the origin: the fitted central same-axis slope is about
+5.387e-19 N/m on x, y, and z. The force changes sign again at approximately
0.6496 mm on each Cartesian axis. Full three-dimensional force Jacobians show
that these axial zeros are saddles, not stable points. Along each of the eight
body diagonals, the proxy has an inner anti-restoring zero near 0.3230 mm per
coordinate, a position-restoring zero near 0.5985 mm per coordinate, and an
outer saddle near 1.08 mm per coordinate. This static classification does not
establish dynamical stability. The four deterministic 5-ms tests all
remained within 1.393 mm, but they began inside the cooling region and reached
speeds as high as about 4.0 m/s. That is evidence of bounded short-run motion
in the proxy only; it is not evidence of loading, capture, or a physical pMOT.

Data are in `outputs/trajectories/pmot/provisional_stark_20Gpcm` and plots are
in `outputs/figures/pmot/provisional_stark_20Gpcm`.

## Damping-preserving vector-only helicity audit

The pMOT design assigns separate roles to the two wavelengths: the already
validated 780-nm cooling/repump beams must remain velocity damping, while the
1529-nm trapping light supplies a position-dependent **vector** transition
shift. At the specified 1529.268881-nm design point, the intended physical
condition is cancellation of the scalar and tensor differential shifts. The
appropriate design audit therefore suppresses those two provisional terms,
keeps the nominal -15 MHz cooling detuning red, and passes only
\(\Delta E_{\rm vector}/\hbar\) to the unchanged 24-state rate kernel.

The retained 780-nm propagation-frame tuple is `++-` on x/y/z for both the
incident and retro components. With no trapping-light shift, its origin
velocity Jacobian is

\[
\frac{\partial\mathbf F}{\partial\mathbf v}
=\operatorname{diag}(-1.051985,-1.051985,-0.711500)
\times10^{-21}\ \mathrm{N\,s/m},
\]

so the 780-nm light is damping. The exact magnitudes depend slightly on the
arbitrary quantization-axis fallback at the vector-field zero, but all tested
eigenvalues remain negative.

For a centered 1529-nm configuration, the incident and retro
propagation-frame labels must match on each path. The vector-only
position-Jacobian diagonal signs for the eight centered tuples are:

| Matched x/y/z path code | x slope | y slope | z slope | Origin position class |
| --- | ---: | ---: | ---: | --- |
| `+++` | - | - | + | saddle |
| `++-` | - | - | - | position-restoring |
| `+-+` | - | + | + | saddle |
| `+--` | - | + | - | saddle |
| `-++` | + | - | + | saddle |
| `-+-` | + | - | - | saddle |
| `--+` | + | + | + | anti-restoring |
| `---` | + | + | - | saddle |

Here `+` means `sigma+`, `-` means `sigma-`, and a negative
force-versus-position slope is restoring. Thus the unique centered choice that
is restoring on all three axes for the present waist ordering is `++-` for
both the incident and retro 1529-nm components—the same propagation-frame
tuple as the 780-nm light. At the provisional 20 G/cm vector-gradient power
scale its position Jacobian is

\[
\frac{\partial\mathbf F}{\partial\mathbf r}
=-1.311295\times10^{-18}\,\mathbf I\ \mathrm{N/m},
\]

and its combined velocity Jacobian is

\[
\frac{\partial\mathbf F}{\partial\mathbf v}
=-7.144316\times10^{-22}\,\mathbf I\ \mathrm{N\,s/m}.
\]

It is therefore locally position restoring and velocity damping in this
vector-only diagnostic. The 1529-nm light does not contribute a direct
mechanical force in this calculation; its vector shift changes the local
780-nm resonance, and the imbalanced 780-nm photon absorption supplies the
restoring radiation-pressure force. The other 56 independent incident/retro
assignments produce a nonzero optical-spin bias at the origin and are not
centered pMOT candidates.

## Full provisional total-shift audit at nominal -15 MHz

This separate audit must not be used to choose the design helicity. It retains
the provisional scalar, vector, and tensor transition-shift ansatz at a fixed
laboratory cooling detuning of -15 MHz. Its helicity-independent central
-16.339691 MHz shift makes the stretched-reference effective cooling detuning
+1.339691 MHz (blue), so every centered tuple is anti-damping. The apparent
restoring sign for the globally reversed tuple in this deliberately
uncompensated diagnostic is consequently a blue-detuning artifact.

At the same fixed powers and detunings, all 64 independent sigma-only choices
for incident x/y/z followed by retro x/y/z were evaluated at the origin. A
centered optical-spin zero requires the incident and retro propagation-frame
helicity labels to match separately on each Cartesian path. Exactly 8 of the
64 configurations satisfy that condition; the other 56 produce a nonzero
central effective-field proxy and force, so the origin is not an equilibrium
and its local Jacobian cannot be interpreted as an origin-stability test.

For a centered path code, one x/y/z tuple is used for both the incident and
retro components. The position-Jacobian diagonal signs are:

| Matched x/y/z path code | x slope | y slope | z slope | Origin position class |
| --- | ---: | ---: | ---: | --- |
| `+++` | + | + | - | saddle |
| `++-` | + | + | + | anti-restoring |
| `+-+` | + | - | - | saddle |
| `+--` | + | - | + | saddle |
| `-++` | - | + | - | saddle |
| `-+-` | - | + | + | saddle |
| `--+` | - | - | - | position-restoring |
| `---` | - | - | + | saddle |

Here `+` means `sigma+`, `-` means `sigma-`, a positive force-versus-position
slope is anti-restoring, and a negative slope is restoring. The current
matched `++-` configuration has
\(\partial F_a/\partial a=+5.4183\times10^{-19}\ \mathrm{N/m}\) on every
axis. Reversing every trapping-path helicity to matched `--+` changes those
three slopes to
\(-5.4183\times10^{-19}\ \mathrm{N/m}\), making it the unique centered
position-restoring choice in this blue-centered provisional force proxy. It is
not the helicity choice for the intended vector-only pMOT design.

That reversal does **not** produce a dynamically stable pMOT at the fixed
parameters. The origin velocity slopes remain positive: approximately
\(+4.352\times10^{-22}\ \mathrm{N\,s/m}\) for the current configuration and
\(+4.327\times10^{-22}\ \mathrm{N\,s/m}\) for the reversed one. Both are
therefore anti-damping. This is consistent with the central cycling-reference
effective detuning being blue,
\(-15-(-16.339691)=+1.339691\ \mathrm{MHz}\). Helicity reversal changes the
position-force sign in this proxy but cannot repair a common detuning error.
Suppressing the scalar/tensor terms at their intended cancellation point—or,
in a future complete Hamiltonian, choosing a carrier that leaves the relevant
cooling transitions red—restores the physically consistent result: matched
`++-` is position restoring and damping, while matched `--+` is position
anti-restoring and damping. Missing production physics must still be resolved
before claiming a trap.

The sweep data and full position/velocity Jacobians are in
`outputs/statistics/pmot/helicity_sweep_20Gpcm`; the bias, classification,
force, shift-decomposition, optical-spin, and velocity-response figures are in
`outputs/figures/pmot/helicity_sweep_20Gpcm`. Experimental retroreflection
handedness must ultimately be derived through the actual window, waveplate,
and mirror Jones transformations rather than copied directly from these six
independent software labels.

## Physics still missing

The following omissions prevent a quantitative trapping claim:

- separate absolute 5S and 5P rank polarizabilities with hyperfine recoupling;
- local Stark-Hamiltonian diagonalization and transformed dipole couplings;
- measured trapping power, path split, transmission, and retro efficiency;
- coherent standing-wave structure, if it is not experimentally averaged;
- window and mirror polarization transformations;
- the conservative force \(-\nabla U\) from absolute state potentials;
- 1529-nm photon scattering, heating, and loss near the polarizability pole;
- nonadiabatic dynamics at the fictitious-field zero; and
- timestep/duration convergence and event-driven cross-checks beyond this
  smoke test.

In addition, the inherited multilevel stimulated-rate/absorption-force closure
must be re-derived and validated before its trajectory force is quantitative.

Reproduce the diagnostic with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.pmot.stark_trajectory_study
```

# pMOT Geometry and Trapping-Beam Design Record

## Status and authority

This document defines the optical geometry starting point for the
pseudo-magneto-optical trap (pMOT). A provisional differential-transition
Stark detuning layer is now implemented, but the production pMOT force model is
not. Its atomic basis and dissipative-light starting point are the
24-state, repumper-enabled population-rate equations in
`src/pmot/mot_multilevel`; the exploratory two-level helper in
`src/pmot/pmot/preliminary_scattering.py` is not a production pMOT engine.
The provisional mapping, its equations, first short trajectories, and the data
missing for a state-resolved Hamiltonian are recorded in
`docs/pmot/PROVISIONAL_AC_STARK_MODEL.md`.

The current pMOT design has one configurable trapping-light frequency. Its
default wavelength is

\[
\lambda_{\mathrm{trap}}=1529.268881\ \mathrm{nm}.
\]

This wavelength replaces the former two-tone proposal. It is selected as the
single-frequency point at which the scalar and tensor shifts cancel while a
vector shift remains. The wavelength must be a named configuration parameter,
not a constant embedded in field or plotting code.

Interpolating the repository's Arora CCSD differential-polarizability table at
this wavelength gives scalar, vector, and tensor coefficients of approximately
-53.054435, +136.334045, and +53.053782 MHz per
`mW/(100 um)^2`, respectively. The scalar-plus-tensor residual is about
-0.000653 in the same units, or (4.8\times10^{-6}) of the vector magnitude.
This checks the stored differential-coefficient cancellation; the production
physics stage must still apply the correct hyperfine-state angular factors.

The pMOT has no anti-Helmholtz coils and no applied external magnetic field:

\[
\mathbf B_{\mathrm{external}}(\mathbf r)=\mathbf 0.
\]

Coil objects, coil exclusion volumes, quadrupole-field evaluations, and
Zeeman shifts from an external field do not belong in the pMOT configuration.
The state-dependent vector AC Stark shift from the trapping light is the
candidate fictitious field. Geometry verification does not yet establish that
this field produces a stable trap.

## Authoritative light-field inventory

### Cooling and repump light

Retain the three mutually orthogonal Cartesian paths used by the multilevel
MOT:

| Path | In-trap propagation axis | Cell-incidence metadata |
|---|---:|---:|
| `horizontal_x` | x | 45 degrees |
| `horizontal_y` | y | 45 degrees |
| `vertical_z` | z | 0 degrees; enters from above |

Each path has an incident and a counterpropagating component. Cooling and
repump light share these directions, giving six cooling and six repump
components. The initial pMOT geometry does not change their beam diameter,
detuning, power, or polarization; every calculation must record the chosen
multilevel-MOT baseline explicitly rather than silently selecting among old
study defaults.

The in-trap x, y, and z axes are mutually orthogonal. A 45-degree angle of
incidence describes the orientation of a glass surface relative to a
horizontal beam; it does not tilt the in-trap Cartesian axis into z.

### Single-frequency trapping light

"One trapping beam" means one laser frequency/configuration, routed into three
Cartesian round-trip paths. It does not mean deleting the retroreflected
component or illuminating only one spatial axis. The inventory is therefore:

- one configurable trapping wavelength, defaulting to 1529.268881 nm;
- three Cartesian paths, x, y, and z; and
- an incident and retroreflected traveling component on every path, for six
  trapping-light components in total.

For a path coordinate \(u\), positive from the incident side toward the
retroreflection mirror, the incident waist is centered at
\(u=-10\ \mathrm{mm}\) and the retroreflected waist is centered at
\(u=+10\ \mathrm{mm}\). The focal positions, not the focal lengths of the two
lenses, are offset by 20 mm. The origin lies midway between the two waist
centers.

Trapping-light helicity is independent of the cooling/repump choices. Specify
each traveling component as propagation-frame `sigma+`, `sigma-`, or `pi`.
Incident and retro helicities must be independently configurable because the
mirror, wave plates, and windows can transform polarization. Do not infer the
return helicity from the incident label, and do not use ambiguous RCP/LCP
labels.

## External optical layouts and the in-trap model

The stored optical-layout studies describe two different laboratory paths.
They are mechanical evidence, not a requirement that the simulation reproduce
every external optic.

### Horizontal paths

`pMOT-Geometry-Codex/FinalHorizontalSolution_with_780.png` is the current
horizontal reference. It uses a 45-degree cell angle of incidence and an
`AC508-080-C` achromat on each side. The nominal lens focal/effective focal
length is 80.3 mm, the input trapping-beam diameter is 35 mm, the glass-cell
outer diameter is 30 mm, and the wall thickness is 5 mm. The round-trip layout
targets the trapping waists at -10 mm and +10 mm. Its mirror is 40 mm beyond
the second-lens principal plane in the simplified model.

The paraxial layout applies fused-silica refraction through an effective ABCD
propagation length. It does not establish measured in-cell astigmatism,
aberration, clipping, polarization transformations, or lateral walkoff.

### Vertical path

The working laboratory sketch for the normal-incidence vertical path uses two
`AC254-045-C` achromats with 45 mm focal lengths and the same target waist
centers at -10 mm and +10 mm. Its former coil-clearance restriction is
obsolete because the pMOT has no anti-Helmholtz coils. Removing that mechanical
restriction may permit a later re-optimization of the vertical optics.

`pMOT-Geometry-Codex/pmot_exp_visualization.png` is a qualitative 45-mm-lens
dichroic/cat-eye sketch. It illustrates distinct incident and return focal
positions; it must not be read as showing two trapping frequencies or unequal
lens focal lengths.

### Symmetric in-trap assumption

The first geometry-verification implementation deliberately uses the same
ideal Gaussian field geometry on x, y, and z: Cartesian axes, waist centers at
-10 mm and +10 mm, and equal ideal waist parameters. This is a symmetric
in-trap approximation. It does not claim that the horizontal 80.3-mm optics
and vertical 45-mm optics are mechanically identical.

Until measured beam parameters are available, use the diffraction-limited
horizontal reference to set the nominal symmetric waist. For
\(\lambda=1529.268881\ \mathrm{nm}\), input radius 17.5 mm, and focal length
80.3 mm,

\[
w_0=\frac{\lambda f}{\pi w_{\mathrm{in}}}\simeq2.234\ \mu\mathrm{m},
\qquad
z_R=\frac{\pi w_0^2}{\lambda}\simeq10.25\ \mu\mathrm{m}.
\]

At the origin, 10 mm from either waist, this ideal model has a beam radius of
approximately 2.179 mm. These are geometry-QA defaults, not measured beam
parameters. Wavelength, waist radius, Rayleigh range or input-optics
parameters, focal offsets, and helicities must remain configurable.

Here (w) is the (1/e^2)-intensity radius. The corresponding nominal
diameters are therefore 4.467 micrometres at either waist and 4.359 mm for
either traveling component at the trap origin. At the plane containing the
opposite component's waist, the defocused component has an approximately
8.717-mm diameter. The configured 35-mm value is the collimated beam diameter
before the focusing lens, not the trapping-beam diameter inside the cell.

The ideal paraxial Gaussian half-divergence parameter is

\[
\theta=\frac{\lambda}{\pi w_0}=0.21793\ \mathrm{rad}
\simeq12.49\ \mathrm{degrees},
\]

or approximately 24.97 degrees full angle. The geometric angle obtained from
the plotted far-field envelope slope is
\(\arctan(0.21793)\simeq12.29\) degrees. This fairly large angle makes the
paraxial diffraction-limited construction a geometry-QA approximation; the
physical waist, divergence, aberration, and astigmatism still require
measurement.

## Geometry-QA intensity model

The physical trapping power has not been specified. Geometry plots must therefore be
normalized per launched watt on each Cartesian path. Let \(P_a\) be the
incident power launched on path \(a\). Report \(I/P_a\) in units of
\(\mathrm{m}^{-2}\), equivalently the intensity in \(\mathrm{W/m^2}\) for
\(P_a=1\ \mathrm{W}\). For geometry QA, use unit retroreflection efficiency;
retain the efficiency as a configurable parameter for later physical work.

For a component with waist center \(u_0\),

\[
w(u)=w_0\sqrt{1+\left(\frac{u-u_0}{z_R}\right)^2},
\]

and its ideal Gaussian intensity per component power is

\[
\frac{I(u,\rho)}{P}
=\frac{2}{\pi w(u)^2}
\exp\!\left[-\frac{2\rho^2}{w(u)^2}\right],
\]

where \(\rho\) is distance perpendicular to the selected Cartesian path. Apply
this expression with \(u_0=-10\ \mathrm{mm}\) to the incident component and
\(u_0=+10\ \mathrm{mm}\) to the retro component.

The initial geometry plots use a standing-wave-averaged, incoherent-envelope
model:

\[
I_{\mathrm{total}}=I_{\mathrm{incident}}+I_{\mathrm{retro}}.
\]

No optical cross term is included. This is a geometry-visualization convention
only, not a conclusion that coherence or standing-wave structure is
negligible for the eventual atomic dynamics.

### Unsigned and vector-weighted quantities

The unsigned total intensity and the vector-shift-driving quantity must not be
confused. For equal offset-waist components, the unsigned sum is even about
the origin and its first axial derivative vanishes there.

For geometry QA, also report the optical-spin intensity factor

\[
\mathbf I_{\mathrm{spin}}
=\sum_j s_j I_j\hat{\mathbf k}_j,
\]

which is the intensity-normalized form of
\(i\mathbf E^*\!\times\mathbf E\). In the project's propagation-frame Jones
convention, \(s=-1\) for `sigma+`, \(s=+1\) for `sigma-`, and \(s=0\) for
`pi`. With equal `sigma+` labels on the incident and retro components, their
opposite wavevectors produce the signed retro-minus-incident profile: it is
odd, crosses zero at the origin, and is locally linear there. Reversing both
helicities reverses this vector. The optical-spin factor verifies geometry and
sign conventions; the production solver must still apply the leading minus
sign, vector polarizability, and state-dependent \(F g_F\mu_B\) factors when
constructing \(\mathbf B_{\mathrm{eff}}\).

In each axial verification figure, the upper panel contains only the selected
axis's incident/retro pair, not all six components. Its black curve is the
unsigned sum for that pair. The lower panel is a zoom of the signed axial
component near the trap center; the word "central" describes the plotted
region. Its value is exactly zero at the origin while its slope is nonzero.
The two-dimensional plane plots, in contrast, sum all six components.

Every geometry verification should show, at minimum:

1. incident, retro, unsigned-total, and signed-vector-proxy intensity along
   each of x, y, and z;
2. normalized two-dimensional intensity planes through the origin, with both
   offset foci explicitly marked; and
3. a three-dimensional or orthogonal-plane view demonstrating that all three
   Cartesian path pairs are present.

## Boundary between geometry, provisional dynamics, and production physics

The production pMOT solver must distinguish at least three trapping-light effects:

1. state-dependent transition shifts that modify cooling and repump
   scattering rates;
2. conservative forces from gradients of dressed-state energies; and
3. stochastic heating or loss from trapping-light scattering.

Treating the vector shift as a fictitious magnetic field is a useful
representation only after its equivalence to the state-resolved AC Stark
Hamiltonian has been established. Production work must map the shifts onto all
8 ground and 16 excited hyperfine-Zeeman states, define the quantization basis
near the fictitious-field zero, recompute local fields and populations at every
required trajectory evaluation, and document interpolation around sharp
polarizability features.

Before quantitative dynamics, also establish the physical trapping power,
transmission and retroreflection efficiency, measured or intended beam
quality, waist and astigmatism, window/mirror polarization transformations,
and whether standing-wave interference must be retained.

Validation must include even unsigned-intensity symmetry, odd vector-proxy
symmetry, the intended three-axis restoring signs, and reversal under
helicity or focus-order reversal. Numerical validation must cover field-grid,
trajectory-timestep, duration, and state-basis convergence and include
representative comparisons with the event-driven multilevel engine.

The current provisional diagnostic implements only item 1 by feeding a
transition-resonance shift into the unchanged 24-state cooling/repump
rate-equation kernel. It deliberately excludes the conservative force and
trapping-light scattering/heating because a differential-polarizability table
cannot determine either one. Its optional 20-G/cm calibration is a
stretched-reference power scale, not a specified apparatus power or production
default.

The retained light is built by the authoritative multilevel beam constructor,
including the 780.232684-nm repump wavelength. The inherited kernel's plotted
rate is its ground-population-weighted available absorption rate, and its force
uses the corresponding absorption momentum even though the population matrix
also contains reverse stimulated links. That solver-wide closure has not yet
been validated against a consistent two-level limit or the event engine, so
the pMOT output labels it as an absorption-force proxy rather than a validated
net scattering force.

## Appendix: superseded historical two-tone proposal

The initial repository commit, `914a7fc`, contained a two-frequency concept
that was removed in commit `60c88b3`. It is retained here only as provenance
and is not a current requirement or default.

| Quantity | Superseded historical value |
|---|---:|
| Trap tone 1 wavelength | 1529.376949 nm |
| Trap tone 2 wavelength | 1529.358429 nm |
| Tone 1 relative intensity | 0.492762 |
| Tone 2 relative intensity | 0.507238 |
| Input beam diameter | 35 mm |
| Incident focus position | -10 mm along its path |
| Retro focus position | +10 mm along its path |
| Focus separation | 20 mm |
| Mirror-gap metadata | 40 mm |
| Historical total power per beam pair | 0.5 W |
| Horizontal lens identifier | AC508-080-C |
| Lens focal/effective focal length | 80.3 mm |
| Lens back focal length | 66.9 mm |
| Glass-cell outer diameter | 30 mm |
| Glass-cell wall thickness | 5 mm |

The two historical tones were separated by approximately 2.37377 GHz. Using
the stored Arora CCSD conversion convention, their scalar, vector, and tensor
coefficients were approximately

| Tone | Scalar | Vector | Tensor |
|---|---:|---:|---:|
| 1529.376949 nm | -738.470536 | -662.817454 | -74.038857 |
| 1529.358429 nm | +717.471334 | +648.808021 | +72.006236 |

Weighting them by 0.492762 and 0.507238 gave approximately 0.038506,
2.488829, and 0.040764 in the same units. That appears to have been an attempt
to cancel scalar and tensor shifts while retaining a vector residual. The
single-frequency 1529.268881-nm design supersedes this mechanism and power
split.

The historical code also stored the vectors

\[
\hat e_1=(1,0,1)/\sqrt2,\qquad
\hat e_2=(0,1,1)/\sqrt2,\qquad
\hat e_3=(0,0,1).
\]

They are not mutually orthogonal and must not be used for the current
Cartesian in-trap model. No historical code constructed a complete 1529-nm
Stark Hamiltonian or pMOT force model.

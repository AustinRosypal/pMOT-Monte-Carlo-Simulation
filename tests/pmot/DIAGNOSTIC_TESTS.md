\# pMOT Simulation Diagnostic Test Plan



\## Purpose



This document defines an ordered diagnostic procedure for a pseudo-magneto-optical-trap (pMOT) simulation in which 1529-nm vector AC Stark shifts replace the spatially varying Zeeman shift of a conventional MOT.



The objective is to determine why the simulation does not trap atoms before changing multiple physical effects simultaneously. The tests isolate:



1\. Doppler cooling and velocity damping.

2\. The signed 1529-nm vector light shift.

3\. The spatial restoring force.

4\. Transition-level scattering contributions.

5\. Detuning and light-shift operating ranges.

6\. Optical pumping and internal-state dynamics.

7\. Polarization handedness and basis conventions.

8\. Deterministic trajectories and capture.

9\. Three-dimensional stability.



Run the tests in this order. Stop at the first failed test, identify the cause, and repeat that test before continuing.



\## Central physical requirement



Near the intended trap center, the force should have the form



$$

F\_z(z,v\_z)\\simeq-\\kappa\_z z-\\beta\_z v\_z,

$$



where



$$

\\kappa\_z=-\\left.\\frac{\\partial F\_z}{\\partial z}\\right|\_{z=v\_z=0}>0

$$



is the spatial spring coefficient and



$$

\\beta\_z=-\\left.\\frac{\\partial F\_z}{\\partial v\_z}\\right|\_{z=v\_z=0}>0

$$



is the velocity-damping coefficient.



Equivalently, sufficiently close to the origin,



$$

zF\_z(z,0)<0

$$



and



$$

v\_zF\_z(0,v\_z)<0.

$$



The 1529-nm field does not need to form an ordinary intensity maximum at the origin. Its principal pMOT role is to produce a position-dependent, state-dependent transition shift. The 780-nm cooling beams then convert that shift into an imbalance in radiation pressure.



\## General execution rules



\- Begin with a deterministic, one-dimensional model.

\- Disable gravity, random recoil, collisions, and unrelated axes until the basic force tests pass.

\- Save the numerical data behind every plot, not only the rendered image.

\- Record all signs, units, beam wavevectors, polarization vectors, and detuning conventions in the run manifest.

\- Do not use the words “left circular” and “right circular” as the simulation's primary polarization representation.

\- At every diagnostic point, retain beam-resolved and transition-resolved quantities until the final force sum is formed.

\- Use symmetric finite differences around zero when estimating local derivatives.

\- Repeat derivative and trajectory tests with at least two smaller position, velocity, and time-step increments.

\- Do not tune parameters against stochastic trajectories until the corresponding deterministic force curves have the correct signs.



The required result layout is specified in \[results/README.md](results/README.md).



\---



\## Test 0: Configuration and unit audit



Before calculating a force, write a machine-readable configuration snapshot and verify the following.



\### Detuning convention



Choose and document one convention. A convenient angular-frequency convention is



$$

\\Delta=\\omega\_L-\\omega\_0.

$$



Then a beam with wavevector $\\mathbf{k}\_j$ has Doppler-shifted detuning



$$

\\Delta\_{D,j}=\\Delta\_j-\\mathbf{k}\_j\\cdot\\mathbf{v}.

$$



If the transition angular frequency is shifted by



$$

\\delta\\omega\_{eg}=\\frac{U\_e-U\_g}{\\hbar},

$$



the effective detuning is



$$

\\Delta\_{\\mathrm{eff},j}^{g\\rightarrow e}

=\\Delta\_j-\\mathbf{k}\_j\\cdot\\mathbf{v}-\\delta\\omega\_{eg}.

$$



For ordinary frequency in hertz, use



$$

\\delta\\nu\_{eg}=\\frac{U\_e-U\_g}{h},

\\qquad

\\delta\\nu\_{D,j}=\\frac{\\mathbf{k}\_j\\cdot\\mathbf{v}}{2\\pi}

=\\frac{\\hat{\\mathbf{k}}\_j\\cdot\\mathbf{v}}{\\lambda\_j}.

$$



Do not mix $h$ with angular frequencies or $\\hbar$ with frequencies in hertz. Confirm whether the linewidth is stored as $\\Gamma$ in rad/s or $\\Gamma/2\\pi$ in Hz.



\### Additional unit checks



Verify and record:



\- Position units.

\- Velocity units.

\- Beam-waist convention, normally the $1/e^2$ intensity radius.

\- Intensity units, preferably $\\mathrm{W/m^2}$ internally.

\- Polarizability units and all conversions to SI.

\- Whether intensity denotes one beam, one propagation direction, or the total field.

\- Whether saturation parameters use per-beam or total intensity.

\- Whether the 1529-nm shift is a level shift or a differential cooling-transition shift.



\### Pass criteria



\- Every frequency-like quantity has an explicit unit type: Hz or rad/s.

\- The code contains one documented definition of effective detuning.

\- Replacing $U\_e-U\_g$ by zero exactly removes the AC Stark contribution.

\- Reversing $\\mathbf{v}$ reverses only the Doppler term.



\---



\## Test 1: Cooling-only Doppler force



\### Setup



1\. Disable the 1529-nm field.

2\. Disable gravity and stochastic recoil.

3\. Retain only the two counterpropagating 780-nm beams along one axis.

4\. Place the atom at $z=0$.

5\. Evaluate $F\_z(0,v\_z)$ for symmetric positive and negative velocities.



\### Required outputs



\- Total force versus velocity.

\- Force from each propagation direction versus velocity.

\- Scattering rate from each beam versus velocity.

\- A fitted value of $\\beta\_z$ near zero velocity.



\### Pass criteria



For small $|v\_z|$,



$$

v\_zF\_z(0,v\_z)<0,

$$



and



$$

\\beta\_z>0.

$$



At $v\_z=0$, balanced beams should satisfy



$$

F\_z(0,0)\\simeq0.

$$



\### Likely causes of failure



\- Incorrect sign of $\\mathbf{k}\\cdot\\mathbf{v}$.

\- Blue rather than red cooling detuning.

\- Inconsistent Hz and rad/s units.

\- Incorrect sign for the absorbed photon momentum $\\hbar\\mathbf{k}$.

\- Unequal counterpropagating beam powers.

\- Polarization weights that do not sum to unity.



Do not continue until this test demonstrates velocity damping.



\---



\## Test 2: Signed 1529-nm vector-shift profile



\### Setup



Disable the 780-nm force and calculate the 1529-nm shifts across the trapping axis.



Save the individual level shifts



$$

\\frac{U\_g(z,m\_F)}{h},

\\qquad

\\frac{U\_e(z,m\_F')}{h},

$$



and the differential transition shifts



$$

\\Delta\\nu\_{\\mathrm{AC}}^{g\\rightarrow e}(z)

=\\frac{U\_e(z)-U\_g(z)}{h}.

$$



For the displaced-focus design, separately calculate



$$

I\_f(z),\\qquad I\_r(z),

$$



and the signed vector profile



$$

S\_v(z)=\\xi\_f I\_f(z)+\\xi\_r I\_r(z),

$$



where $\\xi\_j$ includes the vector-polarizability sign and the local polarization factor. Do not replace $S\_v$ by the unsigned sum $I\_f+I\_r$.



\### Expected symmetry



For equal powers, symmetric foci, and opposing vector contributions,



$$

S\_v(0)\\simeq0

$$



and



$$

S\_v(+z)\\simeq-S\_v(-z).

$$



\### Required outputs



\- Individual forward and return intensities.

\- Signed vector contribution from each beam.

\- Total signed vector profile.

\- Ground- and excited-state shifts for every included $m\_F$ state.

\- Differential shift of every driven 780-nm transition.

\- Odd and even decomposition:



$$

S\_{\\mathrm{odd}}(z)=\\frac{S\_v(z)-S\_v(-z)}{2},

$$



$$

S\_{\\mathrm{even}}(z)=\\frac{S\_v(z)+S\_v(-z)}{2}.

$$



\### Pass criteria



\- The central signed vector shift is zero within the chosen balance tolerance.

\- The odd component dominates the even component throughout the intended central capture region.

\- Opposite $m\_F$ states receive opposite vector shifts.

\- The differential transition shift, rather than only the excited-state shift, is passed to the cooling model.



\### Controlled sign tests



Repeat the profile after individually reversing:



1\. One 1529-nm beam polarization.

2\. The sign of $\\alpha^{(1)}$.

3\. The two focus locations.



Each single reversal should reverse the signed gradient. Reversing any two should recover the original gradient.



\---



\## Test 3: Static restoring force



\### Setup



1\. Enable both the 780-nm cooling beams and the 1529-nm shifts.

2\. Keep $v\_z=0$.

3\. Evaluate $F\_z(z,0)$ at symmetric points about the origin.

4\. Save beam-resolved scattering rates before summing forces.



\### Convention-independent pass criteria



At $z=+\\delta$, the cooling beam propagating toward $-z$ must dominate:



$$

R\_{-z}(+\\delta)>R\_{+z}(+\\delta),

$$



so that



$$

F\_z(+\\delta,0)<0.

$$



At $z=-\\delta$, the opposite beam must dominate:



$$

R\_{+z}(-\\delta)>R\_{-z}(-\\delta),

$$



so that



$$

F\_z(-\\delta,0)>0.

$$



The fitted spring coefficient must satisfy



$$

\\kappa\_z=-\\left.\\frac{\\partial F\_z}{\\partial z}\\right|\_0>0.

$$



\### Failure interpretation



If the force points outward on both sides, the simulation is producing an anti-trap. Reverse exactly one of the following and rerun the test:



\- The relevant 780-nm spherical-polarization assignment.

\- The signed 1529-nm vector contribution.

\- The ordering of the displaced foci.

\- The sign with which $U\_e-U\_g$ enters the effective detuning.



Do not decide which sign is correct from a “left/right circular” label. Decide it from the inward-scattering assertions above.



\---



\## Test 4: Transition-resolved scattering audit



At each of the points



$$

(z,v\_z)=(-\\delta,0),\\quad(0,0),\\quad(+\\delta,0),

$$



save one row for every beam and allowed transition. Include:



\- Beam identifier.

\- $\\hat{\\mathbf{k}}$ and $\\mathbf{k}$.

\- Complex Cartesian polarization vector.

\- Spherical-polarization index $q$.

\- Spherical-polarization weight $P\_q$.

\- Initial and final $F,m\_F$ quantum numbers.

\- Clebsch–Gordan coefficient and its square.

\- Initial-state population.

\- $U\_g/h$ and $U\_e/h$.

\- Differential AC Stark shift $(U\_e-U\_g)/h$.

\- Doppler shift.

\- Effective detuning.

\- Saturation parameter.

\- Scattering rate.

\- Momentum contribution to the force.



\### Pass criteria



\- The sum of transition-resolved momentum contributions reproduces the reported total force.

\- Forbidden transitions have zero coupling.

\- The expected inward beam has the larger total scattering rate away from the origin.

\- Each individual contribution uses that beam's own wavevector and polarization.

\- Spherical-polarization weights obey



$$

P\_{-1}+P\_0+P\_{+1}=1

$$



within numerical tolerance.



\---



\## Test 5: Central detuning and power scan



A correctly signed gradient can still fail to capture atoms if the transition is shifted out of resonance everywhere or only crosses resonance in a very narrow spatial shell.



At the origin calculate



$$

\\Delta\_{\\mathrm{eff}}(0,0)

=\\Delta\_{780}-\\Delta\\nu\_{\\mathrm{AC}}(0).

$$



Scan:



\- 1529-nm power.

\- 780-nm detuning.

\- 780-nm intensity.

\- Focus separation or intensity-gradient scale.

\- Forward/return 1529-nm power ratio.



For every point, record:



\- Central scattering rate.

\- $\\kappa\_z$.

\- $\\beta\_z$.

\- Spatial range over which $zF\_z(z,0)<0$.

\- Velocity range over which $v\_zF\_z(0,v\_z)<0$.



\### Pass criteria



\- The origin retains a useful cooling-photon scattering rate.

\- Both $\\kappa\_z$ and $\\beta\_z$ are positive over a nonzero parameter region.

\- The restoring region is large compared with the intended initial cloud size.

\- The damping region includes a useful fraction of the initial velocity distribution.

\- The optimum is not located at the numerical boundary of the parameter scan.



More 1529-nm power is not automatically beneficial. Excessive shifts can reduce the capture volume even while increasing the central gradient.



\---



\## Test 6: Internal-state population dynamics



Add internal-state physics in stages:



1\. Fixed $m\_F$ populations or a simple toy transition.

2\. Incoherent optical pumping.

3\. Repumping from the other ground hyperfine manifold.

4\. Spontaneous-emission recoil.

5\. Coherences and dark-state physics, if required.

6\. Gravity and other external effects.



At every stage, rerun Tests 1 and 3.



\### Required outputs



\- Population versus time for every included state.

\- Total population versus time.

\- Scattering rate versus time.

\- Force versus time.

\- Population leakage into uncoupled or weakly coupled states.



\### Pass criteria



\- Total population is conserved, apart from explicitly modeled loss channels.

\- The repumper prevents permanent accumulation in the wrong ground hyperfine manifold.

\- The restoring force survives the steady-state optical-pumping distribution.

\- Any loss of force can be associated with identified states or coherences.



If a rate-equation model predicts unexplained force suppression near the field zero, compare selected operating points with an optical-Bloch-equation calculation.



\---



\## Test 7: Handedness and polarization conventions



\### Global representation



Represent every beam in one fixed laboratory Cartesian basis using



$$

\\hat{\\mathbf{k}}\_j

$$



and a normalized complex polarization vector



$$

\\boldsymbol{\\epsilon}\_j,

\\qquad

\\boldsymbol{\\epsilon}\_j^\*\\cdot\\boldsymbol{\\epsilon}\_j=1,

\\qquad

\\hat{\\mathbf{k}}\_j\\cdot\\boldsymbol{\\epsilon}\_j=0.

$$



The simulation should not determine physics from observer-dependent strings such as “left-handed” or “right-handed.” Those labels may be retained only as display metadata.



\### Vector-light-shift sign



Calculate the optical-spin vector directly:



$$

\\mathbf{s}\_j

=i\\boldsymbol{\\epsilon}\_j^\*\\times\\boldsymbol{\\epsilon}\_j.

$$



For a transverse circular plane wave, $\\mathbf{s}\_j$ lies parallel or antiparallel to $\\hat{\\mathbf{k}}\_j$. Use



$$

\\mathbf{B}\_{\\mathrm{eff},j}

=C\_j I\_j\\mathbf{s}\_j,

$$



where $C\_j$ contains the appropriate vector polarizability, Landé factor, and sign conventions.



For equal forward and return intensities at the trap center, the desired cancellation is



$$

\\mathbf{B}\_{\\mathrm{eff},f}(0)

+\\mathbf{B}\_{\\mathrm{eff},r}(0)

\\simeq0.

$$



If the two beams have the same atomic coefficient, this generally requires



$$

\\mathbf{s}\_r\\simeq-\\mathbf{s}\_f

$$



along the relevant laboratory axis.



\### Spherical polarization components



Choose and document a spherical-basis convention. For example,



$$

\\mathbf{e}\_{+1}

=-\\frac{\\mathbf{e}\_x+i\\mathbf{e}\_y}{\\sqrt{2}},

\\qquad

\\mathbf{e}\_{0}=\\mathbf{e}\_z,

\\qquad

\\mathbf{e}\_{-1}

=\\frac{\\mathbf{e}\_x-i\\mathbf{e}\_y}{\\sqrt{2}}.

$$



Then calculate



$$

\\epsilon\_q=\\mathbf{e}\_q^\*\\cdot\\boldsymbol{\\epsilon},

\\qquad

P\_q=|\\epsilon\_q|^2.

$$



Use the same convention in the dipole matrix elements and Clebsch–Gordan coefficients.



\### Retroreflection



Do not reverse a beam by changing only the sign of $\\mathbf{k}$. Propagate its Jones vector through the actual optical train, including:



\- Wave plates.

\- Cell windows.

\- Any $s/p$ phase differences at oblique incidence.

\- Mirror reflection.

\- The return passage through the same components.



After constructing the returned Cartesian field, recompute



$$

\\mathbf{s}\_r,

\\qquad

P\_{-1},P\_0,P\_{+1}.

$$



The definitive handedness test is not a label. It is



$$

R\_{\\mathrm{inward}}(+\\delta)

>

R\_{\\mathrm{outward}}(+\\delta)

$$



and its counterpart at $-\\delta$.



\### Field-zero basis continuity



Avoid defining the internal-state basis independently along the local direction of $\\mathbf{B}\_{\\mathrm{eff}}$ at every position. At the trap center, that direction is undefined; across the origin it reverses. A naive local-axis implementation can therefore relabel $m\_F$, $\\sigma^+$, and $\\sigma^-$ discontinuously.



For diagnostics, retain a fixed global basis and construct the vector Hamiltonian as



$$

H\_{\\mathrm{vec}}(\\mathbf r)

\\propto

\\mathbf{F}\\cdot\\mathbf{B}\_{\\mathrm{eff}}(\\mathbf r).

$$



If a local eigenbasis is required, transform states and operators continuously and verify basis-invariant observables across the zero crossing.



\---



\## Test 8: Deterministic trajectories and capture map



Do not begin with a large thermal ensemble. Run the following single-atom cases first:



1\. $z=+\\delta$, $v\_z=0$: the atom should initially accelerate inward.

2\. $z=-\\delta$, $v\_z=0$: the atom should initially accelerate inward.

3\. $z=0$, $v\_z>0$: the atom should slow.

4\. $z=0$, $v\_z<0$: the atom should slow.

5\. Small displacement and small velocity: the atom should return through damped oscillation or monotonic relaxation.



Repeat every trajectory with time steps



$$

\\Delta t,\\qquad\\frac{\\Delta t}{2},\\qquad\\frac{\\Delta t}{4}.

$$



If photon scattering is sampled probabilistically, require



$$

R\_{\\mathrm{sc}}\\Delta t\\ll1.

$$



Next, scan a grid of initial $(z,v\_z)$ and classify trajectories as captured or escaped using explicit spatial and time thresholds.



\### Required outputs



\- Position versus time.

\- Velocity versus time.

\- Force versus time.

\- Scattering rate versus time.

\- Phase-space plot of $v\_z$ versus $z$.

\- Capture/escape map in initial $(z,v\_z)$ space.

\- Time-step convergence comparison.



\### Pass criteria



\- Small perturbations decay toward the origin.

\- The qualitative trajectory does not change under time-step refinement.

\- The capture map contains a connected region around $(0,0)$.



\---



\## Test 9: Three-dimensional stability



Add the three spatial axes one at a time. Near the origin calculate the position and velocity Jacobians



$$

A\_{ij}

=\\left.\\frac{\\partial F\_i}{\\partial r\_j}\\right|\_0,

\\qquad

D\_{ij}

=\\left.\\frac{\\partial F\_i}{\\partial v\_j}\\right|\_0.

$$



For an approximately uncoupled system, the diagonal elements should satisfy



$$

A\_{ii}<0,

\\qquad

D\_{ii}<0.

$$



For the general coupled system, form the linearized phase-space matrix



$$

M=

\\begin{pmatrix}

0\&I\\\\

m^{-1}A\&m^{-1}D

\\end{pmatrix}.

$$



Local linear stability requires every eigenvalue of $M$ to have a negative real part.



\### Required outputs



\- Position-force Jacobian $A$.

\- Velocity-force Jacobian $D$.

\- Eigenvalues of $M$.

\- Force-vector slices in the $xy$, $xz$, and $yz$ planes.

\- Per-axis and cross-axis beam contributions.



\### Pass criteria



\- All three axes are locally restoring and damping.

\- Cross-axis terms do not create an unstable eigenmode.

\- Adding an axis does not reverse a previously correct force because of a polarization or basis error.

\- The equilibrium point remains near the intended origin.



Because the fictitious field is not a real magnetostatic field, it need not obey the anti-Helmholtz $1:1:-2$ gradient ratio. The decisive condition is stability of the complete optical force, not literal reproduction of Maxwell's equations for a real quadrupole field.



\---



\## Minimum acceptance checklist



The pMOT model should not be called trapping until all of the following are demonstrated:



\- \[ ] Cooling-only force opposes velocity.

\- \[ ] Balanced cooling beams give approximately zero force at the origin.

\- \[ ] The signed 1529-nm vector shift is approximately odd about the origin.

\- \[ ] The differential transition shift $(U\_e-U\_g)/h$ is used.

\- \[ ] At positive displacement, the inward beam scatters more strongly.

\- \[ ] At negative displacement, the opposite inward beam scatters more strongly.

\- \[ ] Both fitted coefficients satisfy $\\kappa\_i>0$ and $\\beta\_i>0$.

\- \[ ] Polarization vectors are normalized and transverse.

\- \[ ] Spherical-polarization weights sum to unity.

\- \[ ] Retroreflected polarizations are derived from Cartesian/Jones vectors rather than verbal handedness labels.

\- \[ ] Internal-state populations are conserved except for explicit losses.

\- \[ ] The repumper prevents permanent hyperfine-state leakage.

\- \[ ] Deterministic trajectories converge under time-step refinement.

\- \[ ] A connected capture region exists around zero position and velocity.

\- \[ ] The complete three-dimensional linearized system is stable.



\## References



\- F. Le Kien, P. Schneeweiss, and A. Rauschenbeutel, “Dynamical polarizability of atoms in arbitrary light fields: general theory and application to cesium,” \*European Physical Journal D\* 67, 92 (2013). \[Author preprint](https://arxiv.org/abs/1211.2673)

\- E. L. Raab, M. Prentiss, A. Cable, S. Chu, and D. E. Pritchard, “Trapping of Neutral Sodium Atoms with Radiation Pressure,” \*Physical Review Letters\* 59, 2631 (1987). \[Article](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.59.2631)

\- X. Xu, T. H. Loftus, M. J. Smith, J. L. Hall, A. Gallagher, and J. Ye, “Dynamics in a two-level atom magneto-optical trap,” \*Physical Review A\* 66, 011401(R) (2002). \[Article](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.66.011401)

\- M. R. Tarbutt, “Magneto-optical trapping forces for atoms and molecules with complex level structures,” \*New Journal of Physics\* 17, 015007 (2015). \[Author preprint](https://arxiv.org/abs/1409.0244)

\- J. A. Devlin and M. R. Tarbutt, “Laser cooling and magneto-optical trapping of molecules analyzed using optical Bloch equations and the Fokker–Planck–Kramers equation,” \*Physical Review A\* 98, 063415 (2018). \[Article](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.063415)


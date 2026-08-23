# Physical model: temperature versus cooling detuning

## What was modeled

At each cooling detuning, ten independently seeded realizations of a preloaded 25-atom Rb-87 cloud are evolved in the fixed default six-beam MOT using the repumper-enabled 24-state adiabatic population-rate equations and Langevin recoil diffusion.  Each realization's reported final temperature is the time average over the final 5 ms of the unbiased, instantaneous-center-of-mass-subtracted three-dimensional velocity variance; the plotted point is the arithmetic mean of the ten realization temperatures. Only the cooling detuning Delta=n Gamma changes between points.

This is a trapped-cloud equilibration/temperature study, not a capture/loading
study. No incident launch disk is used: a disk would require choosing an
incident launch speed and would therefore add another physical variable.
Instead, each of the 10 independent realizations is a randomly sampled 25-atom
preloaded cloud. A plotted estimate is interpreted as an equilibrium
temperature only when it passes the survivor and stationarity checks below.

## Fixed physical and numerical parameters

- Atomic/optical model: repumper-enabled 24-state Rb-87 D2 adiabatic population-rate equations (8 ground states and 16 excited states).
- Cooling beams: six default Gaussian beams, 20 mW per beam and 12.7 mm diameter.
- Repumper: enabled at 0.1 mW per beam with the fixed baseline diameter and detuning; all dipole-allowed F=1 channels including F'=0 are retained.
- Magnetic field, gravity, beam geometry, polarizations, linewidth, and every configuration value other than cooling detuning: fixed at the production baseline recorded in the metadata. The repumper is explicitly enabled for this baseline even though the generic configuration factory defaults to disabled.
- Initial cloud: independent normal position draws with sigma=0.25 mm per coordinate and Maxwell-Gaussian velocity draws at 2 mK.
- Evolution: 25 ms with 5 microsecond steps and Langevin recoil diffusion enabled.
- Final plateau: last 5 ms; a temperature survivor reaches the requested duration and stays within the 2 mm core throughout this final interval.
- Detuning grid: 25 values from n=-10 to n=-0.1, with Delta=n Gamma and Gamma/(2 pi)=6.07 MHz.
- Doppler reference: T_D=hbar Gamma/(2 k_B)=145.657 microkelvin.

The ten random initial phase-space clouds are drawn once and reused at every
detuning (common random initial conditions). Langevin recoil streams are
independent between atoms, realizations, and detunings. A configuration audit
verifies that only the solver, apparatus, and cooling-beam detuning fields vary.

## Temperature and uncertainty estimator

For each realization and recorded time, the instantaneous center-of-mass
velocity is removed and the unbiased sample variance is calculated separately
for x, y, and z: T_i=m Var(v_i)/k_B. The scalar temperature is
(T_x+T_y+T_z)/3 and is averaged over the final plateau. The plotted point is
the arithmetic mean of the ten realization temperatures. Error bars are one
standard error of that mean; the saved CSV also includes a two-sided 95%
Student-t interval across the ten independent realizations.

For a cloud estimate to pass, at least
5 atoms must survive in
the final core and both its four-subwindow spread and fitted relative drift over
the plateau must be below 0.15. A plot
point is marked valid only if all ten cloud estimates pass. Hollow red circles
are nonstationary diagnostics and the hollow orange triangle marks an
insufficient-survivor diagnostic; neither is claimed as an equilibrium
temperature.

The survivor fraction is reported independently from temperature. Its primary
uncertainty interval is a 95% Student-t interval across the ten realization
fractions, which preserves the ensemble clustering. A pooled 95% Wilson
interval across all 250 trajectories is also saved as a secondary diagnostic.

## Why an earlier trapped fraction could equal one

These atoms start as a compact preloaded cloud (position sigma 0.25 mm, well
inside the 2 mm core), rather than arriving from a capture disk. With only ten
previous trajectories, all ten could readily survive for the short simulated
duration, producing a displayed fraction of 1. That value described survival
of that small preloaded sample; it did not mean every incident atom would be
captured. This run uses 250 trajectories per detuning and reports an interval.

## Limitations

- A survivor-only temperature is conditional on remaining in the final core and can exhibit survivor bias; survivor fraction is therefore shown separately.
- The adiabatic rate equations omit optical coherences and sub-Doppler polarization-gradient cooling.
- Atoms are noninteracting; density-dependent reabsorption and collisions are absent.
- The Doppler line is a reference, not a claim that this full magnetic MOT must attain the ideal low-saturation optical-molasses limit.
- The 25 ms evolution is finite; quality-flagged values diagnose where this duration or the final-core sample does not establish equilibrium.
- Quantitative claims remain provisional until timestep and duration convergence are checked independently.

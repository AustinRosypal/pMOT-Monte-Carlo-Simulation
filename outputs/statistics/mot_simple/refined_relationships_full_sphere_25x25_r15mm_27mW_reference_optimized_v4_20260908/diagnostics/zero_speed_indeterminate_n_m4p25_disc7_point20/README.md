# Zero-flux indeterminate diagnostic: detuning -4.25, ray (7, 20)

The schema-5 production campaign stopped correctly at this ray because the
direct velocity-grid audit could not assign a trapped or escaped state at
exactly zero launch speed.  The complete configured adaptive ladder, ending
at 1.0 s with 1.25 and 0.625 microsecond timesteps, remained non-definitive or
timestep-dependent.  Production preserved 615 of 625 rays for this detuning
and did not reinterpret the censored trajectory as an escape.

Two independent 2.0 s integrations were then run from the exact saved launch
state with the same MOT configuration.  Both terminated by timeout, remained
finite, made no entry into the 2 mm core, and ended at a radius of about
19.8493 mm.  Their terminal positions agree to 4.4270e-14 m in Euclidean norm;
their terminal velocities agree to 1.8275e-12 m/s.  This is converged evidence
for persistence through 2.0 s, but it is not evidence that the atom will
eventually be captured or escape.  The physical capture classification is
therefore retained as indeterminate.

Exactly zero incident speed has zero effusive loading weight because the
loading integrand contains `v^3`.  A schema-6 continuation may consequently
retain this node as a first-class `indeterminate_zero_flux` value while still
calculating an exact loading rate: every positive-speed node must remain
definitive and timestep-agreeing, the undefined cross section at v=0 must not
be plotted as zero, and the quadrature must retain the kinematic integrand
anchor g(0)=0.  This exception does not relax the two-core-entry or 5 ms core
residence trapped criterion and does not classify a timeout as an escape.

See `diagnostic.json` for the exact geometry, configuration, terminal states,
wall times, convergence differences, and required downstream interpretation.

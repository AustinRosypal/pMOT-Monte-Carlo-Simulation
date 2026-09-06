# Test 1 — cooling-only Doppler force

**Result: PASS WITH QUALIFICATIONS.** With gravity, recoil, the 1529-nm shift, repumping, and
unrelated axes disabled, the calculation prescribes equal populations of 0.2
in the five F=2 Zeeman substates. This is the fixed-population stage required
before adding optical pumping in Test 6. It uses the production saturated
transition-rate matrix and reconstructs the absorption-momentum proxy from the
two z cooling components.

The near-zero fit gives
`beta_z = 9.499376087e-22 N s/m`; all three central-difference refinements are
positive, and the balanced zero-velocity force is
`0.000e+00 N`.

The companion steady-state control documents why the unrestricted 24-state
solve is not meaningful with only these two beams: population occupies the
disconnected F=1 dark subspace and the force collapses to zero. Numerical plot
inputs are in `cooling_force_and_rates_vs_velocity.csv`; the complete
beam-by-beam and transition-by-transition calculation is retained in
`transition_resolved_cooling_force.csv`, with grouped sums in
`transition_sum_reconstruction.csv`. Accordingly this is a pass of the
fixed-population Doppler stage, not a claim that the unrestricted two-beam
24-state steady state provides damping. The retained mechanical quantity is
the inherited ground-population-weighted available-absorption momentum proxy;
it is not yet a validated net scattering force because reverse stimulated
emission momentum is omitted. The rate model stores normalized full hyperfine
`C^2`; `sqrt(C^2)` in the ledger is a magnitude, not a signed Clebsch--Gordan
coefficient.

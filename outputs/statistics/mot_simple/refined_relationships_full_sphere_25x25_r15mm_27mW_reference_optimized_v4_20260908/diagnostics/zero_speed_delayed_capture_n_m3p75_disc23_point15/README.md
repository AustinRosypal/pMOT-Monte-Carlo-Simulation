# Delayed-capture diagnostic: detuning -3.75, ray (23, 15)

The production campaign stopped correctly because its fail-closed zero-speed
audit never obtained a terminal classification for this ray. The 200, 250,
and 400 ms levels all ended in `timeout` at both configured timesteps, with no
entry into the 2 mm core. A new 600 ms pair at 1.25 and 0.625 microseconds also
ended in `timeout`, again with zero core entries. The two 600 ms terminal states
agree at approximately the 1e-13 m position scale, so the persistence is
physical rather than a meaningful timestep discrepancy.

Continuing each independently converged 600 ms state showed delayed capture.
At both timesteps the atom first entered the core at 796.3675 ms, exited after
about 1.792 ms (short of the 5 ms residence route), and entered again at
806.452 ms. The second entry satisfies the project's alternative trapped
criterion. The two terminal times differ by only 0.625 microseconds, one fine
integration step, and both runs reached the same 1.058272 mm minimum radius.

Therefore an 800 ms audit would still be censored. The first demonstrated
terminal event is 806.4525 ms; a bounded 850 ms node-only adaptive level using
1.25 and 0.625 microsecond timesteps is recommended. It supplies about 43.5 ms
of duration margin and should classify this exact zero-speed node as trapped
without weakening the trapped criterion or accepting a timeout as escape.

The production schema-5 policy adopts a more conservative 1.0 s cap at those
same two timesteps. This does not increase the cost for a trajectory that
terminates at 806.452 ms, but it leaves additional horizon margin for other
rare direct-grid nodes later in the detuning sweep.

See `diagnostic.json` for the exact initial state, physical configuration,
results at every tested duration/timestep, terminal states, wall times, and
core-crossing timestamps.

# Initial pMOT construction diagnostic campaign

This campaign executes the ordered procedure in
[`docs/pmot/DIAGNOSTIC_TESTS.md`](../../../../docs/pmot/DIAGNOSTIC_TESTS.md).
It uses the provisional ideal-magic vector-only transition proxy and does not
claim a production pMOT.

## Ordered result

- Test 0: **PASS_WITH_QUALIFICATIONS** — Configuration and unit audit (raw polarizability unit provenance remains external to the CSV)
- Test 1: **PASS_WITH_QUALIFICATIONS** — Cooling-only Doppler force (fixed uniform F=2 stage using the inherited absorption-momentum proxy)
- Test 2: **FAIL** — Signed 1529-nm vector-shift profile (The differential-only Arora table cannot determine individual ground- and excited-level shifts U_g/h and U_e/h for all 24 states.)
- Test 3: **NOT_RUN** — Static restoring force (ordered plan stopped after Test 2 failed)
- Test 4: **NOT_RUN** — Transition-resolved scattering audit (ordered plan stopped after Test 2 failed)
- Test 5: **NOT_RUN** — Central detuning and power scan (ordered plan stopped after Test 2 failed)
- Test 6: **NOT_RUN** — Internal-state population dynamics (ordered plan stopped after Test 2 failed)
- Test 7: **NOT_RUN** — Handedness and polarization conventions (ordered plan stopped after Test 2 failed)
- Test 8: **NOT_RUN** — Deterministic trajectories and capture map (ordered plan stopped after Test 2 failed)
- Test 9: **NOT_RUN** — Three-dimensional stability (ordered plan stopped after Test 2 failed)

Execution stopped after Test 2.
Test 2 was repeated with a denser spatial grid and three smaller symmetric
finite-difference steps. The same failure remained: the optical vector profile
is odd, but the applied fixed-name transition shift is even because the local
basis flips, and separate level shifts are unavailable from the differential-
only input table.

Each executed-test directory contains its own explanation, result JSON, and
source CSV data; Tests 1 and 2 also contain rendered figures. `result_index.csv`
is the concise machine-readable campaign status and `run_manifest.json`
records provenance and assumptions.

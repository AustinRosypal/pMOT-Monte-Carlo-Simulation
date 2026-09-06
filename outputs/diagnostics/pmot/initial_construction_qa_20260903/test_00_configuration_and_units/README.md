# Test 0 — configuration and unit audit

**Result: PASS WITH QUALIFICATIONS.** The production rate path uses angular
frequency and implements

`Delta_eff[b,g->e] = Delta_L[b] - delta_HFS[g->e] - k_b dot v - (DeltaE_AC[e]-DeltaE_AC[g])/hbar; B_external = 0`

An all-zero transition-shift array is bit-for-bit identical to omitting the
Stark input. Direct calls to the production rate kernel verify both its
`-k dot v` and `-DeltaE_AC/hbar` signs, including compensating Doppler/Stark
identities and the rate ordering on the red-detuned side. Resolved beam
detunings, wavelengths, and powers match the production configuration.

With the Stark shift frozen, reversing velocity changes only the Doppler term.
In the complete pMOT call, velocity also Doppler-shifts the 1529-nm wavelengths
used to interpolate polarizability; the measured maximum shift change for the
recorded probe is
2366.68 Hz.

The remaining provenance gap is that the raw Arora CSV headers do not state a
physical polarizability unit. The code assumes SI and the values have the
expected SI scale, but that source metadata has not been independently
verified. See `unit_audit.csv`, `configuration_snapshot.json`, and the complete
18-component `beam_manifest.csv`. The qualification is explicitly non-gating
because every code-side convention and conversion is recorded and internally
consistent; only independent provenance for the raw table's unit label is
missing.

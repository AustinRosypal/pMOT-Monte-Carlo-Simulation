# Test 2 — signed 1529-nm vector-shift profile

**Result: FAIL (confirmed by refined repeat).** The optical vector-energy
profile itself is centered and odd: its maximum even/odd ratio is
`1.073e-15`. However, the current model cannot produce the required
individual `U_g/h` and `U_e/h` values because its Arora input contains only
differential polarizabilities.

A second independent failure appears at the field zero. The quantization axis
follows the optical-spin vector and reverses across the origin, while the
reference energy is projected onto that axis and becomes a magnitude. For the
named F=2,mF=+2 -> F'=3,mF'=+3 transition at z=+/-0.1 mm, the actually applied
shifts are `0.279976529` and `0.279976529` MHz: the same
sign. In a fixed global z basis the signed reference values are
`0.279976529` and `-0.279976529` MHz: opposite signs.

The requested single-component polarization reversal was also checked. It
creates an even center-biased profile rather than a reversed odd gradient;
`controlled_sign_reversals.csv` shows that reversing both path helicities,
alpha1, or the focus order gives the physically meaningful global sign tests.

The nonzero differential transition proxy is wired into the production rate
kernel correctly: the pMOT wrapper exactly matches an independent direct
shifted-kernel call, and the shift changes the beam-resolved rates. Those
checks are preserved in `production_kernel_shift_probe.csv` and
`wrapper_shift_handoff_beam_rates.csv`; the failure is therefore in the
available Stark-state model and field-zero basis treatment, not in the final
argument handoff.

Per the canonical stop rule, Tests 3--9 were not executed. The refined profile,
three finite-difference gradient steps, all 54 transition-level proxies, and
the explicit missing-level table preserve the complete diagnosis.

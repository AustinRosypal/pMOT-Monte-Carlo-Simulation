# Temporary fixed-stretched-transition pMOT diagnostic

Status: **PASS_WITH_QUALIFICATIONS**, with the qualifications below.

This isolated test evaluates only the differential AC-Stark shift of
`5S1/2 F=2, mF=+2 -> 5P3/2 F'=3, mF'=+3`. It is not imported by the production pMOT
workflow. The six 1529-nm traveling components use matched propagation-frame
helicities `++-` on x/y/z for incident and retro paths. That is the previously
identified restoring orientation in the *ideal vector-only, red-detuned*
diagnostic.

## What was calculated

For every trapping component, the code evaluates the full three-dimensional
Doppler projection, its own atom-frame wavelength, its local Gaussian
intensity, and the narrow-table scalar/vector/tensor differential
polarizabilities. It then forms energy in joules, sums the six components, and
converts with `delta_nu_AC = DeltaE_AC/h`. The shifted cooling resonance is
`nu_res = nu_bare + delta_nu_AC`.

Ordinary-frequency detuning used for the named reference transition:

`delta_nu_eff[b] = delta_nu_L[b] - delta_nu_HFS - dot(k_hat_b, v)/lambda_b - DeltaE_AC/h`

The F'=3 stretched reference has `delta_nu_HFS = 0` by definition. The
equivalent angular-frequency equation is:

`Delta_eff[b] = Delta_L[b] - delta_HFS - k_b dot v - DeltaE_AC/hbar`

The 1529-nm Doppler shift is used only to choose the polarizability at the
atom-frame wavelength; it is not added again as a 780-nm detuning.

## Principal result

The demonstration uses 38.294486 mW incident on each Cartesian path,
chosen only as the historical 20 G/cm vector-gradient proxy. The physical
trapping power remains unspecified, and every shift scales linearly with path
power in this incoherent-envelope model.

At rest at the symmetric origin (fixed lab-z basis), the six-beam shift is:

- scalar: -16.339691 MHz
- vector: +0.000000 MHz
- tensor: +0.000000 MHz
- total: -16.339691 MHz

Thus a cooling carrier held 15 MHz below the *bare* stretched line is
+1.339691 MHz from the shifted central line. A positive value is
blue detuning. The scalar/tensor cancellation is excellent for one aligned
circular component, but it is not a six-beam identity: for one fixed axis the
orthogonal beams have different tensor geometry. At the symmetric origin the
three tensor-pair contributions cancel one another while all scalar
contributions add.

## Interpretation and boundary

This run corrects the sign problem in the older local-adiabatic transition
proxy by holding the named mF basis fixed and using the signed vector
projection `Q dot n_fixed`. The x, y, and z lineouts are three separate basis
diagnostics. A single atom cannot be simultaneously stretched along all three
axes. With no external magnetic field, the state labels become degenerate and
ambiguous at the optical-spin zero; a physical 3D prediction requires the
complete local Stark Hamiltonian and its eigenvectors.

The supplied CSV contains differential transition coefficients only. It
cannot yield separate ground/excited level shifts for all 24 states,
conservative Stark forces, or transformed 780-nm dipole couplings. Its column
headers also do not state the raw polarizability unit; this test follows the
repository's SI assumption, whose dimensional chain is internally consistent
but whose source provenance remains unverified.

No pMOT trajectory, loading, capture, temperature, or quantitative restoring
force is claimed. Coherent standing waves, 1529-nm scattering/heating/loss,
window and mirror polarization transformations, and nonadiabatic passage
through the fictitious-field zero remain outside this test.

## Files

- `figures/`: seven rendered diagnostic figures.
- `data/`: the underlying beamwise ledgers and lineout tables.
- `run_manifest.json`: formulas, configuration, hashes, and assumptions.
- `qa_result.json`: numerical identity, parity, reversal, and unit-chain checks.
- `summary.json`: concise machine-readable findings.

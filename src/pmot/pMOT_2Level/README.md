# pMOT_2Level

This package models a no-coil, effective two-level pseudo-MOT on the Rb-87
cooling manifold

\[
5S_{1/2},F=2 \longrightarrow 5P_{3/2},F'=3.
\]

It retains the deterministic saturated mean-force architecture of
`pmot.mot_simple`, but replaces the conventional magnetic Zeeman shift with a
differential AC-Stark shift from the six 1529-nm trapping-light components.
It does not resolve the individual `m_F` populations.

The pMOT branch uses 27 mW in each 780-nm cooling traveling component. This
does not alter the independently validated 20 mW default in `mot_simple`.

## Current boundary

Implemented here:

- the existing single-frequency 1529.268881-nm trapping-light geometry;
- three Cartesian round-trip paths and six traveling components;
- ideal Gaussian intensity at any finite 3D position;
- explicit x/y/z launched powers, in watts;
- every component propagation direction, `k_hat_i`;
- the helicity-weighted optical-spin vector
  `sum_i s_i I_i(r) k_hat_i`; and
- a local quantization axis with explicit handling of vector-shift zeros;
- one Doppler-shifted trapping wavelength and interpolated scalar, vector, and
  tensor differential polarizability per trapping component;
- the effective differential scalar/vector/tensor energy shifts;
- decomposition of every 780-nm cooling beam into local
  `sigma-`/`pi`/`sigma+` weights;
- channel-resolved effective detunings and scattering rates;
- the summed six-beam, three-dimensional mean radiation-pressure force; and
- an RK4 trajectory integrator that recomputes the full local problem at every
  stage.

Not implemented here, and therefore not implied by this model:

- an `m_F`-resolved lower- and upper-manifold Stark Hamiltonian;
- optical pumping among the Zeeman sublevels;
- conservative trapping-light gradient force;
- trapping-light photon scattering, heating, or loss;
- recoil diffusion; or
- validated capture, loading, or temperature predictions.

The default path helicities are `(sigma+, sigma+, sigma-)` for both incident
and retro components in x/y/z order. These are propagation-frame labels at the
atoms. They retain the prior provisional centered geometry but are not yet a
validated two-level pMOT design conclusion.

Physical trapping power is still unspecified, so the point evaluator requires
the caller to supply `path_powers_w=(P_x, P_y, P_z)`. Use `(1, 1, 1)` only when
an intensity-per-launched-watt geometry normalization is intended.

## Equations and conventions

For trapping component `j`, the atom-frame wavelength is evaluated separately:

\[
\lambda'_j=\frac{\lambda_j}{1-\hat{\mathbf k}_j\cdot\mathbf v/c}.
\]

The code preloads and linearly interpolates only
`Arora_CCSD_Differential_Polarizabilities.csv`. It neither clips nor
extrapolates. The dataset convention is excited minus ground,
`5P3/2 - 5S1/2`, so the calculated differential energy is
`Delta U = U_e - U_g`. With `E_j^2=2I_j/(c epsilon_0)`, the implemented
differential energy terms are

\[
\Delta U_j^{(0)}=-\Delta\alpha_j^{(0)}E_j^2,
\]

\[
\Delta U_j^{(1)}=-\Delta\alpha_j^{(1)}E_j^2s_j
(\hat{\mathbf k}_j\cdot\hat{\mathbf n}),
\]

\[
\Delta U_j^{(2)}=-\frac{1}{2}\Delta\alpha_j^{(2)}E_j^2
\left(3|\boldsymbol\epsilon_j^*\cdot\hat{\mathbf n}|^2-1\right).
\]

Thus the scalar and vector shifts acquire the conversion factor
`-2/(c h epsilon_0)` from polarizability times intensity, while the tensor term
has the extra one-half angular prefactor. The vector-energy geometry vector is

\[
\mathbf Q=-\sum_j\Delta\alpha_j^{(1)}E_j^2s_j\hat{\mathbf k}_j,
\qquad
\hat{\mathbf n}=\frac{\mathbf Q}{|\mathbf Q|}.
\]

At `Q=0`, the physical axis is undefined. The code retains the normalized
previous axis solely as a continuous numerical basis and reports
`quantization_axis_from_vector_shift=False`.

Because the input CSV supplies one differential triplet rather than separate
level- and `m_F`-resolved polarizabilities, the effective closure uses the
tabulated vector shift as the `sigma+` reference, reverses it for `sigma-`, and
sets it to zero for `pi`. The scalar and tensor reference contribution is
common to all three channels. For `q=-1,0,+1`,

\[
\Delta\nu_{\rm AC}^{(q)}=
\frac{\Delta U^{(0)}+\Delta U^{(2)}+q\Delta U^{(1)}}{h}.
\]

The ordinary-frequency detuning supplied to the scattering formula is

\[
\boxed{\delta_{\rm eff}^{(b,q)}=
\delta_0^{(b)}-\frac{\mathbf k_b\cdot\mathbf v}{2\pi}
-\Delta\nu_{\rm AC}^{(q)}}.
\]

The minus sign is required: a positive `Delta U = U_e-U_g` raises the atomic
transition frequency, reducing `laser frequency - atomic frequency`. Each
beam's propagation-frame polarization is projected onto the spherical basis
about `n_hat`; all `q`-channel rates from all six cooling beams are summed.
No single beam or photon is randomly selected.

## Public API map

- `build_pmot_2level_trapping_beams`: construct the six 1529-nm traveling
  components.
- `build_pmot_2level_cooling_beams`: construct the six 780-nm cooling
  components with matched propagation-frame path helicities, defaulting to
  `(sigma+, sigma+, sigma-)` on x/y/z for both directions and 27 mW per
  traveling component.
- `component_intensities_w_per_m2`: return all six local component intensities.
- `total_intensity_w_per_m2`: return the unsigned local intensity sum.
- `optical_spin_vector_w_per_m2`: return the geometry-only helicity/direction
  intensity vector.
- `quantization_axis_at_point`: return the geometry-only candidate axis and
  whether the optical-spin vector defines it.
- `evaluate_trapping_geometry`: collect the preceding geometry quantities in
  one record.
- `load_pmot_2level_polarizability_table`: preload the required narrow CSV.
- `atom_frame_trapping_wavelengths_nm`: return the six Doppler-shifted lookup
  wavelengths.
- `evaluate_differential_ac_stark_shift`: return interpolated component data,
  scalar/vector/tensor shifts, the physical vector-shift axis, and the three
  effective `q`-channel transition shifts.
- `evaluate_pmot_2level_local_observable`: decompose all six cooling beams and
  return their rates plus the total three-dimensional force and acceleration.
- `rk4_step_pmot_2level` and `simulate_pmot_2level_trajectory`: advance or
  integrate the deterministic mean-force dynamics with full local
  recomputation.

```python
from pmot.pMOT_2Level import (
    build_pmot_2level_trapping_beams,
    evaluate_differential_ac_stark_shift,
    evaluate_trapping_geometry,
    load_pmot_2level_polarizability_table,
)

beams = build_pmot_2level_trapping_beams()
sample = evaluate_trapping_geometry(
    (0.5e-3, 0.0, 0.0),
    beams,
    path_powers_w=(1.0, 1.0, 1.0),
)

sample.component_intensities_w_per_m2
sample.component_k_hat
sample.total_intensity_w_per_m2
sample.optical_spin_vector_w_per_m2
sample.quantization_axis
sample.quantization_axis_from_optical_spin

table = load_pmot_2level_polarizability_table()
stark = evaluate_differential_ac_stark_shift(
    beams,
    position_m=(0.5e-3, 0.0, 0.0),
    velocity_m_per_s=(0.0, 0.0, 0.0),
    path_powers_w=(0.01, 0.01, 0.01),
    polarizability_table=table,
)
```

At an exact optical-spin zero, such as the balanced origin, the physical axis
is undefined. The API returns the normalized caller-selected fallback axis and
sets `quantization_axis_from_optical_spin=False`; callers must not interpret that
fallback as a nonzero effective field.

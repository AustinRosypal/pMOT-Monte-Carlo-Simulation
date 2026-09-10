# Temporary pMOT effective-field/axis diagnostic

This calculation is evaluated at `r = [0.1, 0.1, 0.1] mm` for a stationary
atom using 1529.268881 nm trapping light and the matched propagation-frame path
helicities `(sigma+, sigma+, sigma-)`.  The displayed
`38.294486173 mW/path` is the historical 20 G/cm
stretched-transition proxy scale, not an apparatus power recommendation.

For each component,

`U_vec,j = -alpha_vector(lambda_seen,j) [2 I_j/(c epsilon_0)] s_j k_hat_j`

and the available differential cycling-transition data are represented as

`B_eq,j = U_vec,j / [mu_B (3 g_F'=3 - 2 g_F=2)]`.

The result is `B_eq = (+0.196690742, +0.196690742, -0.196690742) G`,
with magnitude `0.340678358 G`
and normalized axis
`n = (+0.577350269, +0.577350269, -0.577350269)`.

The CSV gives every beam contribution and its sigma+/pi/sigma- fractions in
that local spherical basis.  Cooling and repump components with the same path
direction and propagation-frame helicity have the same geometric fractions.

## Scientific boundary

This is a transition-equivalent diagnostic.  The Arora CSV contains only a
differential polarizability triplet for the reference transition; it cannot
produce separate ground- and excited-manifold fictitious fields or the full
24-state Stark Hamiltonian.  A production pMOT must obtain level-resolved
polarizabilities, sum the complete Stark operators, diagonalize them locally,
and transform the 780-nm dipole couplings into the resulting eigenbasis.

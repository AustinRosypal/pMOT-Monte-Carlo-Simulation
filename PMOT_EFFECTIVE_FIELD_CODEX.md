# pMOT Surrogate Effective-Magnetic-Field Specification

## Purpose

Implement a proof-of-principle pseudo-MOT (pMOT) test inside the existing conventional MOT simulation.

The 1530 nm pMOT beams produce vector AC Stark shifts. For this test, **do not simulate those Stark shifts directly in the atomic Hamiltonian**. Instead:

1. Evaluate the local intensity of each of the six 1530 nm beams.
2. Convert each beam intensity into that beam's vector effective-magnetic-field contribution.
3. **Vector-sum all six contributions** to obtain the total effective magnetic field
   \[
   \mathbf B_{\rm eff}(\mathbf r).
   \]
4. Feed \(\mathbf B_{\rm eff}(\mathbf r)\) into the existing MOT machinery everywhere the ordinary coil field would normally be used.
5. Set the actual coil magnetic field to zero.
6. Plot the resulting effective-field geometry and simulate atom trajectories/capture.

This is a surrogate test of whether the proposed optical geometry can reproduce a MOT-like restoring field.

---

# 1. Units and constants

Use SI units everywhere.

\[
\lambda = 1.530\times 10^{-6}\ {\rm m}
\]

\[
h = 6.62607015\times10^{-34}\ {\rm J\,s}
\]

\[
\mu_B = 9.2740100783\times10^{-24}\ {\rm J/T}
\]

The differential vector polarizability used for this surrogate mapping is

\[
\alpha_{\rm eff}^{(1)}
=
136.334\,
\frac{\rm MHz}
{\rm mW/(100\,\mu m)^2}
=
1363.34\,
\frac{\rm Hz}
{\rm W/m^2}.
\]

For the stretched MOT transition

\[
|5s_{1/2},F=2,m_F=2\rangle
\rightarrow
|5p_{3/2},F'=3,m_F'=3\rangle,
\]

use

\[
G \equiv g_{F'}m_{F'}-g_Fm_F
=3g_{F'}-2g_F
\simeq 1.
\]

Therefore define the constant

\[
\boxed{
\kappa
=
-\frac{2h\alpha_{\rm eff}^{(1)}}{\mu_B G}
}
\]

which numerically is

\[
\boxed{
\kappa
=
-1.948\times10^{-7}
\frac{\rm T}{\rm W/m^2}.
}
\]

Thus a single circularly polarized beam contributes

\[
\boxed{
\mathbf B_{{\rm eff},i}(\mathbf r)
=
\kappa\,
\xi_i\,
I_i(\mathbf r)\,
\hat{\mathbf k}_i
}
\]

where

- \(I_i\) is in \({\rm W/m^2}\),
- \(\hat{\mathbf k}_i\) is the beam propagation unit vector,
- \(\xi_i=\pm1\) is the helicity sign defined by
  \[
  i(\boldsymbol\epsilon_i^-\times\boldsymbol\epsilon_i^+)
  =
  \xi_i\hat{\mathbf k}_i.
  \]

**Do not scalar-sum the six intensities and then convert to \(B\).  
Convert each beam to a vector contribution and then vector-sum the six magnetic-field contributions.**

---

# 2. Vacuum-cell simulation domain

The Gaussian formulas below are the free-propagation solution inside the atom-containing vacuum region of the glass cell.

For the solved 45-degree optical path, the inner glass surfaces correspond to a longitudinal vacuum half-length

\[
L_{\rm vac}/2
=
14.1421356\ {\rm mm}
=
1.41421356\times10^{-2}\ {\rm m}.
\]

For the present 3D surrogate simulation, rotate the same solved optical geometry onto the \(x\), \(y\), and \(z\) axes and use the computational vapor domain

\[
\boxed{
\mathcal V_{\rm cell}
=
\left\{
(x,y,z):
|x|\le L,\;
|y|\le L,\;
|z|\le L
\right\},
\qquad
L=1.41421356\times10^{-2}\ {\rm m}.
}
\]

This is the current symmetric 3-axis approximation to the atom-containing glass-cell volume.

Define

\[
\chi_{\rm cell}(x,y,z)
=
\begin{cases}
1,&(x,y,z)\in\mathcal V_{\rm cell},\\
0,&\text{otherwise}.
\end{cases}
\]

All pMOT effective-field contributions should be set to zero outside this domain:

\[
\boxed{
\mathbf B_{\rm eff}(\mathbf r)=0
\qquad
\text{for }
\mathbf r\notin\mathcal V_{\rm cell}.
}
\]

Atoms should likewise not be evolved as vapor atoms beyond the glass walls; use the existing wall-loss/trajectory termination logic if available.

> If a more exact 3D CAD/analytic cell-interior boundary already exists in the project, use that boundary instead of the cubic approximation above. The beam formulas themselves remain unchanged.

---

# 3. Gaussian beam solution inside the cell

Inside the vacuum region there are no lenses or glass interfaces, so each 1530 nm beam is an ordinary freely propagating Gaussian beam.

The forward and return powers are equal.  As an October 2, 2026 centered-field
idealization, retain the solved waist locations but replace the former strongly
unequal Rayleigh ranges by the center-balanced pair below.  The pair makes the
two on-axis intensities equal at the origin and preserves the geometric mean of
the former Rayleigh ranges.  The former values remain available in code as
`LEGACY_ASYMMETRIC_SURROGATE_EFFECTIVE_FIELD_CONFIG` for provenance.

## Forward beam

Waist position:

\[
\boxed{
s_f=-1.00326971\times10^{-2}\ {\rm m}
}
\]

Rayleigh range:

\[
\boxed{
z_{R,f}=9.103072091279295\times10^{-6}\ {\rm m}
}
\]

Define

\[
D_f(s)
=
(s-s_f)^2+z_{R,f}^2
=
(s+1.00326971\times10^{-2})^2
+
(9.103072091279295\times10^{-6})^2.
\]

For power \(P\), longitudinal coordinate \(s\), and perpendicular distance \(\rho\),

\[
\boxed{
\mathcal I_f(P;s,\rho)
=
\chi_{\rm cell}
\frac{2Pz_{R,f}}
{\lambda D_f(s)}
\exp\left[
-\frac{2\pi z_{R,f}\rho^2}
{\lambda D_f(s)}
\right].
}
\]

## Return beam

Waist position:

\[
\boxed{
s_r=+9.97601576\times10^{-3}\ {\rm m}
}
\]

Rayleigh range:

\[
\boxed{
z_{R,r}=9.000504018534303\times10^{-6}\ {\rm m}
}
\]

Define

\[
D_r(s)
=
(s-s_r)^2+z_{R,r}^2
=
(s-9.97601576\times10^{-3})^2
+
(9.000504018534303\times10^{-6})^2.
\]

Then

\[
\boxed{
\mathcal I_r(P;s,\rho)
=
\chi_{\rm cell}
\frac{2Pz_{R,r}}
{\lambda D_r(s)}
\exp\left[
-\frac{2\pi z_{R,r}\rho^2}
{\lambda D_r(s)}
\right].
}
\]

---

# 4. Six beam intensities

Use

\[
P_x=P_y=10.8\ {\rm mW}=0.0108\ {\rm W},
\]

\[
P_z=21.6\ {\rm mW}=0.0216\ {\rm W}.
\]

For each axis, the forward beam propagates in the positive Cartesian direction and the return beam propagates in the negative Cartesian direction.

## Beam 1: \(+x\)

\[
\hat{\mathbf k}_{+x}=+\hat{\mathbf x},
\qquad
s=x,
\qquad
\rho_x=\sqrt{y^2+z^2}.
\]

\[
\boxed{
I_{+x}(x,y,z)
=
\mathcal I_f
\left(
0.0108;\,
x,\,
\sqrt{y^2+z^2}
\right).
}
\]

Equivalently,

\[
\boxed{
I_{+x}
=
\chi_{\rm cell}
\frac{
2(0.0108)z_{R,f}
}{
\lambda\left[(x-s_f)^2+z_{R,f}^2\right]
}
\exp\left[
-\frac{
2\pi z_{R,f}(y^2+z^2)
}{
\lambda\left[(x-s_f)^2+z_{R,f}^2\right]
}
\right].
}
\]

## Beam 2: \(-x\)

\[
\hat{\mathbf k}_{-x}=-\hat{\mathbf x},
\qquad
s=x,
\qquad
\rho_x=\sqrt{y^2+z^2}.
\]

\[
\boxed{
I_{-x}(x,y,z)
=
\mathcal I_r
\left(
0.0108;\,
x,\,
\sqrt{y^2+z^2}
\right).
}
\]

Equivalently,

\[
\boxed{
I_{-x}
=
\chi_{\rm cell}
\frac{
2(0.0108)z_{R,r}
}{
\lambda\left[(x-s_r)^2+z_{R,r}^2\right]
}
\exp\left[
-\frac{
2\pi z_{R,r}(y^2+z^2)
}{
\lambda\left[(x-s_r)^2+z_{R,r}^2\right]
}
\right].
}
\]

## Beam 3: \(+y\)

\[
\hat{\mathbf k}_{+y}=+\hat{\mathbf y},
\qquad
s=y,
\qquad
\rho_y=\sqrt{x^2+z^2}.
\]

\[
\boxed{
I_{+y}(x,y,z)
=
\mathcal I_f
\left(
0.0108;\,
y,\,
\sqrt{x^2+z^2}
\right).
}
\]

Equivalently,

\[
\boxed{
I_{+y}
=
\chi_{\rm cell}
\frac{
2(0.0108)z_{R,f}
}{
\lambda\left[(y-s_f)^2+z_{R,f}^2\right]
}
\exp\left[
-\frac{
2\pi z_{R,f}(x^2+z^2)
}{
\lambda\left[(y-s_f)^2+z_{R,f}^2\right]
}
\right].
}
\]

## Beam 4: \(-y\)

\[
\hat{\mathbf k}_{-y}=-\hat{\mathbf y},
\qquad
s=y,
\qquad
\rho_y=\sqrt{x^2+z^2}.
\]

\[
\boxed{
I_{-y}(x,y,z)
=
\mathcal I_r
\left(
0.0108;\,
y,\,
\sqrt{x^2+z^2}
\right).
}
\]

Equivalently,

\[
\boxed{
I_{-y}
=
\chi_{\rm cell}
\frac{
2(0.0108)z_{R,r}
}{
\lambda\left[(y-s_r)^2+z_{R,r}^2\right]
}
\exp\left[
-\frac{
2\pi z_{R,r}(x^2+z^2)
}{
\lambda\left[(y-s_r)^2+z_{R,r}^2\right]
}
\right].
}
\]

## Beam 5: \(+z\)

\[
\hat{\mathbf k}_{+z}=+\hat{\mathbf z},
\qquad
s=z,
\qquad
\rho_z=\sqrt{x^2+y^2}.
\]

\[
\boxed{
I_{+z}(x,y,z)
=
\mathcal I_f
\left(
0.0216;\,
z,\,
\sqrt{x^2+y^2}
\right).
}
\]

Equivalently,

\[
\boxed{
I_{+z}
=
\chi_{\rm cell}
\frac{
2(0.0216)z_{R,f}
}{
\lambda\left[(z-s_f)^2+z_{R,f}^2\right]
}
\exp\left[
-\frac{
2\pi z_{R,f}(x^2+y^2)
}{
\lambda\left[(z-s_f)^2+z_{R,f}^2\right]
}
\right].
}
\]

## Beam 6: \(-z\)

\[
\hat{\mathbf k}_{-z}=-\hat{\mathbf z},
\qquad
s=z,
\qquad
\rho_z=\sqrt{x^2+y^2}.
\]

\[
\boxed{
I_{-z}(x,y,z)
=
\mathcal I_r
\left(
0.0216;\,
z,\,
\sqrt{x^2+y^2}
\right).
}
\]

Equivalently,

\[
\boxed{
I_{-z}
=
\chi_{\rm cell}
\frac{
2(0.0216)z_{R,r}
}{
\lambda\left[(z-s_r)^2+z_{R,r}^2\right]
}
\exp\left[
-\frac{
2\pi z_{R,r}(x^2+y^2)
}{
\lambda\left[(z-s_r)^2+z_{R,r}^2\right]
}
\right].
}
\]

---

# 5. Helicity assignment

Use

\[
\boxed{
\xi_{+x}=\xi_{-x}=+1,
\qquad
\xi_{+y}=\xi_{-y}=+1,
\qquad
\xi_{+z}=\xi_{-z}=-1.
}
\]

This sign choice is intended to produce the local quadrupole-like structure

\[
\mathbf B_{\rm eff}
\sim
+b_\perp x\,\hat{\mathbf x}
+
b_\perp y\,\hat{\mathbf y}
-
b_z z\,\hat{\mathbf z}
\]

near the origin, with approximately

\[
b_\perp\sim 10\ {\rm G/cm},
\qquad
b_z\sim20\ {\rm G/cm}.
\]

Because \(\hat{\mathbf k}\) already changes sign for the counterpropagating member of each pair, the two beams in one axis pair use the same \(\xi\) in their own propagation frames.

---

# 6. Individual effective magnetic-field contributions

Compute all six separately:

\[
\boxed{
\mathbf B_{+x}
=
\kappa I_{+x}(+\hat{\mathbf x})
}
\]

\[
\boxed{
\mathbf B_{-x}
=
\kappa I_{-x}(-\hat{\mathbf x})
}
\]

\[
\boxed{
\mathbf B_{+y}
=
\kappa I_{+y}(+\hat{\mathbf y})
}
\]

\[
\boxed{
\mathbf B_{-y}
=
\kappa I_{-y}(-\hat{\mathbf y})
}
\]

\[
\boxed{
\mathbf B_{+z}
=
\kappa(-1) I_{+z}(+\hat{\mathbf z})
}
\]

\[
\boxed{
\mathbf B_{-z}
=
\kappa(-1) I_{-z}(-\hat{\mathbf z}).
}
\]

---

# 7. MASTER EQUATION: total 3D effective field

The total field is the **vector sum of all six beam contributions**:

\[
\boxed{
\mathbf B_{\rm eff}(x,y,z)
=
\mathbf B_{+x}
+
\mathbf B_{-x}
+
\mathbf B_{+y}
+
\mathbf B_{-y}
+
\mathbf B_{+z}
+
\mathbf B_{-z}.
}
\]

Equivalently,

\[
\boxed{
\mathbf B_{\rm eff}(x,y,z)
=
\kappa
\left[
(I_{+x}-I_{-x})\hat{\mathbf x}
+
(I_{+y}-I_{-y})\hat{\mathbf y}
-
(I_{+z}-I_{-z})\hat{\mathbf z}
\right].
}
\]

In general sum notation,

\[
\boxed{
\mathbf B_{\rm eff}(\mathbf r)
=
\kappa
\sum_{i=1}^{6}
\xi_i I_i(\mathbf r)\hat{\mathbf k}_i.
}
\]

This vector sum is mandatory. Do **not** compute

\[
B_{\rm eff}\propto \sum_i I_i
\]

as a scalar.

Return the field in Tesla:

```python
B_eff = np.array([Bx, By, Bz])  # tesla
```

---

# 8. Suggested implementation API

Implement a single field callback with the same interface as the existing coil-field function:

```python
def pmot_effective_B(position_m: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    position_m : ndarray, shape (3,)
        [x, y, z] in meters.

    Returns
    -------
    B_eff_T : ndarray, shape (3,)
        Total vector effective magnetic field [Bx, By, Bz] in tesla.
        Returns zero outside the vapor-cell domain.
    """
```

Suggested internal structure:

```python
LAMBDA_PMOT = 1.530e-6

SF = -1.00326971e-2
ZR_F = 9.103072091279295e-6

SR = +9.97601576e-3
ZR_R = 9.000504018534303e-6

PX = 0.0108
PY = 0.0108
PZ = 0.0216

KAPPA = -1.948e-7  # T / (W/m^2)

CELL_HALF_LENGTH = 1.41421356e-2


def inside_cell(x, y, z):
    return (
        abs(x) <= CELL_HALF_LENGTH
        and abs(y) <= CELL_HALF_LENGTH
        and abs(z) <= CELL_HALF_LENGTH
    )


def I_forward(P, s, rho2):
    D = (s - SF)**2 + ZR_F**2
    return (
        2.0 * P * ZR_F / (LAMBDA_PMOT * D)
        * np.exp(-2.0 * np.pi * ZR_F * rho2 / (LAMBDA_PMOT * D))
    )


def I_return(P, s, rho2):
    D = (s - SR)**2 + ZR_R**2
    return (
        2.0 * P * ZR_R / (LAMBDA_PMOT * D)
        * np.exp(-2.0 * np.pi * ZR_R * rho2 / (LAMBDA_PMOT * D))
    )


def pmot_effective_B(position_m):
    x, y, z = position_m

    if not inside_cell(x, y, z):
        return np.zeros(3)

    I_px = I_forward(PX, x, y*y + z*z)
    I_mx = I_return(PX, x, y*y + z*z)

    I_py = I_forward(PY, y, x*x + z*z)
    I_my = I_return(PY, y, x*x + z*z)

    I_pz = I_forward(PZ, z, x*x + y*y)
    I_mz = I_return(PZ, z, x*x + y*y)

    Bx = KAPPA * (I_px - I_mx)
    By = KAPPA * (I_py - I_my)
    Bz = -KAPPA * (I_pz - I_mz)

    return np.array([Bx, By, Bz])
```

Use the analytic equations above as the source of truth.

---

# 9. Integration into the existing MOT simulation

For this surrogate test:

- disable the physical anti-Helmholtz coil field;
- wherever the MOT code currently calls something like
  ```python
  B = coil_field(position)
  ```
  replace or switch it to
  ```python
  B = pmot_effective_B(position)
  ```
- continue using the existing MOT Zeeman-shift, local quantization-axis, polarization decomposition, scattering-rate, force, and trajectory machinery unchanged.

At nonzero field,

\[
\hat{\mathbf q}
=
\frac{\mathbf B_{\rm eff}}
{|\mathbf B_{\rm eff}|}
\]

is the local quantization axis.

At or extremely near a zero of the effective field, avoid numerically normalizing a zero vector. Reuse the simulation's existing zero-field handling.

This is intentionally a surrogate model: it asks whether the spatially structured vector AC Stark shift can emulate a MOT-like magnetic field closely enough to give trapping in the existing MOT dynamics.

---

# 10. Required diagnostic plots

Before running capture simulations, generate diagnostics of the effective field.

At minimum output:

1. \(B_x(x,0,0)\), \(B_y(0,y,0)\), \(B_z(0,0,z)\) line cuts.
2. \(|\mathbf B|\) along all three principal axes.
3. \(B_x\), \(B_y\), \(B_z\), and \(|\mathbf B|\) in the \(xy\) plane at \(z=0\).
4. The same quantities in the \(xz\) plane at \(y=0\).
5. The same quantities in the \(yz\) plane at \(x=0\).
6. A 2D/3D vector-field visualization near the trap center.
7. Numerical derivatives at the origin:
   \[
   \left.\frac{\partial B_x}{\partial x}\right|_0,\quad
   \left.\frac{\partial B_y}{\partial y}\right|_0,\quad
   \left.\frac{\partial B_z}{\partial z}\right|_0.
   \]
8. Report the field offset
   \[
   \mathbf B_{\rm eff}(0,0,0)
   \]
   explicitly.

The intended local target is approximately

\[
\frac{\partial B_x}{\partial x}\sim+10\ {\rm G/cm},
\]

\[
\frac{\partial B_y}{\partial y}\sim+10\ {\rm G/cm},
\]

\[
\frac{\partial B_z}{\partial z}\sim-20\ {\rm G/cm}.
\]

Use

\[
1\ {\rm T}=10^4\ {\rm G},
\qquad
1\ {\rm T/m}=100\ {\rm G/cm}.
\]

Do not assume these targets are exactly achieved; evaluate and report the actual field generated by the analytic beam model.

---

# 11. Trajectory and capture tests

After validating the field map, run the existing MOT trajectory simulation with

\[
\mathbf B_{\rm coil}=0
\]

and

\[
\mathbf B(\mathbf r)
=
\mathbf B_{\rm eff}(\mathbf r).
\]

Use the same 780 nm cooling/repumper model and all existing multilevel MOT physics unless a separate experiment is explicitly requested.

Compare against the ordinary anti-Helmholtz MOT using the same:

- initial atom positions,
- initial velocities,
- cooling detuning,
- cooling beam powers/saturations,
- repumper settings,
- capture criterion,
- integration time,
- wall-loss condition.

Output:

- representative 3D trajectories,
- position vs. time,
- velocity vs. time,
- whether each atom is captured/lost,
- capture probability or capture velocity using the project's existing metric,
- direct comparison with the conventional coil-field MOT.

---

# 12. Important interpretation

This calculation maps the stored **differential vector polarizability** of the stretched

\[
F=2,m_F=2
\rightarrow
F'=3,m_F'=3
\]

transition onto an equivalent magnetic field.

It is therefore a proof-of-principle geometry test. Treating this equivalent field as an ordinary magnetic field inside the full multilevel MOT machinery gives every transition ordinary Zeeman scaling. That is not yet identical to a full state-by-state AC-Stark calculation for arbitrary \(m_F\rightarrow m_F'\) transitions.

For this task, however, **do exactly this surrogate mapping first**. The immediate question is whether the six-beam optical intensity geometry produces a sufficiently MOT-like vector field to support capture/trapping in the existing simulation.

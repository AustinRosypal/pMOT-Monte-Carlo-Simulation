# Pseudo-MOT (pMOT): AC Stark-Shift Theory

## Motivation

The inspiration behind the pseudo-magneto-optical trap (pMOT) is to cool and trap atoms using only the electric fields present in light beams, without any external magnetic fields. Magnetic fields have historically been necessary to provide the restoring Zeeman force, accumulating atoms at a desired location: the origin of the MOT, where the magnetic-field magnitude is zero.

However, the wire coils needed to generate these magnetic fields are large, bulky, difficult to manage, and draw substantial current. Eliminating these coils would free considerable space and make the apparatus smaller and cleaner. In addition, the pMOT may lead to new ways of trapping and manipulating atoms.

## Theory

### Electric-dipole interaction

Light is an oscillating electromagnetic field. Atoms couple to light through the electric-dipole interaction,

$$
\hat{V}=-\vec{d}\cdot\vec{E}=-e\vec{r}\cdot\vec{E}.
$$

First-order perturbation theory gives the following correction to an atomic energy level:

$$
\begin{aligned}
\Delta E_{\psi}^{(1)}
&=\langle\psi|\hat{V}|\psi\rangle \\
&=-e\langle\psi|\vec{r}\cdot\vec{E}|\psi\rangle \\
&=-e\vec{E}\cdot\langle\psi|\vec{r}|\psi\rangle \\
&=-e\vec{E}\cdot\int\psi^*(\vec{r})\,\vec{r}\,\psi(\vec{r})\,d^3r.
\end{aligned}
$$

The position operator $\vec{r}$ is a rank-1 spherical tensor, with components $r_q$ for $q=0,\pm1$. It is odd under parity, so it cannot connect states of identical parity, such as $|\psi\rangle$ with itself. Equivalently, the integral of an even function times an odd function over all space vanishes. Thus, the first-order correction to the atomic energy is zero.

### Second-order shift and polarizability

The first nonzero correction occurs at second order and is called the **AC Stark shift**. It is computed by summing the dipole-interaction matrix elements over the other states in the unperturbed eigenbasis:

$$
\begin{aligned}
\Delta E_a^{(2)}
&=\sum_{b\ne a}
\frac{|\langle b|\hat{V}|a\rangle|^2}
{E_a^{(0)}-E_b^{(0)}} \\
&=\sum_{b\ne a}
\frac{
\left\langle b\left|\sum_i\hat{d}_i\widetilde{E}_i\right|a\right\rangle
\left\langle a\left|\sum_j\hat{d}_j\widetilde{E}_j^*\right|b\right\rangle}
{E_a^{(0)}-E_b^{(0)}} \\
&=\sum_{i,j}\widetilde{E}_i\widetilde{E}_j^*
\sum_{b\ne a}
\frac{\langle b|\hat{d}_i|a\rangle\langle a|\hat{d}_j|b\rangle}
{E_a^{(0)}-E_b^{(0)}} \\
&\equiv\sum_{i,j}\widetilde{E}_i\widetilde{E}_j^*\alpha_{ij}^{(a)}(0).
\end{aligned}
$$

Here, $\sum_i\equiv\sum_{i=x,y,z}$, $E_a^{(0)}$ and $E_b^{(0)}$ are unperturbed energies, and $\widetilde{E}_i$ denotes a component of the electric field. The rank-2 object $\alpha_{ij}$, which contains the matrix elements of states coupled by the electric-dipole operator, is the **polarizability tensor** associated with the AC Stark shift.

Polarizability measures how strongly an atom's electron cloud is distorted by an external electric field. First-order perturbation theory gives the perturbed state as

$$
|a^{(1)}\rangle
=|a^{(0)}\rangle
+\sum_{b\ne a}
\frac{\langle b|\hat{V}|a\rangle}
{E_a^{(0)}-E_b^{(0)}}|b\rangle.
$$

Because $\hat{V}$ is odd under parity, the field couples states $|a\rangle$ and $|b\rangle$ of opposite parity. The resulting mixed-parity perturbed state can have a nonzero dipole moment proportional to the polarizability and the applied field.

The AC Stark shift is proportional to the square of the electric field,

$$
\Delta E^{(2)}\propto\widetilde{E}^{,2},
$$

because the electric field must first perturb a parity-definite eigenstate into a mixed-parity state with a nonzero dipole moment and then act on that new state a second time. This quadratic field dependence is a characteristic feature of the AC Stark shift.

### Scalar, vector, and tensor polarizabilities

The polarizability is a rank-2 object with nine components: it maps a three-dimensional electric-field vector $\vec{E}$ to a three-dimensional induced-dipole vector $\vec{d}$. Because it is formed from two rank-1 objects, it can be decomposed into three irreducible components:

$$
\begin{aligned}
1\otimes1
&=0\oplus1\oplus2 \\
&\equiv\text{scalar}\oplus\text{vector}\oplus\text{tensor}.
\end{aligned}
$$

- The **scalar part**, $\alpha^{(0)}$, is obtained from the trace of the polarizability tensor and has one independent component.
- The **vector part**, $\alpha^{(1)}$, is the antisymmetric part of $\alpha_{ij}$ and has three independent components.
- The **tensor part**, $\alpha^{(2)}$, is the symmetric, traceless remainder and has five independent components.

The irreducible polarizabilities are

$$
\begin{aligned}
\alpha^{(0)}(F;\omega)
&=\sum_{F'}
\frac{2\omega_{F'F}|\langle F\|\vec{d}\|F'\rangle|^2}
{3\hbar\left(\omega_{F'F}^2-\omega^2\right)}, \\
\alpha^{(1)}(F;\omega)
&=\sum_{F'}(-1)^{F+F'+1}
\sqrt{\frac{6F(2F+1)}{F+1}}
\begin{Bmatrix}
1&1&1\\
F&F&F'
\end{Bmatrix}
\frac{\omega_{F'F}|\langle F\|\vec{d}\|F'\rangle|^2}
{\hbar\left(\omega_{F'F}^2-\omega^2\right)}, \\
\alpha^{(2)}(F;\omega)
&=\sum_{F'}(-1)^{F+F'}
\sqrt{\frac{40F(2F+1)(2F-1)}{3(F+1)(2F+3)}}
\begin{Bmatrix}
1&1&2\\
F&F&F'
\end{Bmatrix}
\frac{\omega_{F'F}|\langle F\|\vec{d}\|F'\rangle|^2}
{\hbar\left(\omega_{F'F}^2-\omega^2\right)}.
\end{aligned}
$$

The denominator represents the detuning of the incident light at angular frequency $\omega$ relative to each atomic transition. The transition angular frequency is

$$
\omega_{F'F}
\equiv\omega_{F'}-\omega_F
=\frac{E_{F'}-E_F}{\hbar}.
$$

The unperturbed energy levels are tabulated in the NIST Atomic Spectra Database. The reduced dipole matrix elements, $|\langle F\|\vec{d}\|F'\rangle|^2$, encode transition amplitudes induced by the field; their values have historically been measured experimentally.

### Energy shift in irreducible form

In terms of the scalar, vector, and tensor polarizabilities, the energy shift is

$$
\begin{aligned}
\Delta E(F,m_F;\omega)
={}&-\alpha^{(0)}(F;\omega)|E_0^{(+)}|^2 \\
&-\alpha^{(1)}(F;\omega)
\left(i\vec{E}_0^{(-)}\times\vec{E}_0^{(+)}\right)_z
\frac{m_F}{F} \\
&-\alpha^{(2)}(F;\omega)
\left(3|E_{0z}^{(+)}|^2-|E_0^{(+)}|^2\right)
\frac{3m_F^2-F(F+1)}{2F(2F-1)}.
\end{aligned}
$$

#### Scalar term

The scalar term represents an overall shift that depends on the light intensity and detuning, since

$$
|E_0^{(+)}|^2
=\vec{E}_0^{(+)}\cdot\vec{E}_0^{(-)}
=\frac{2I}{c\epsilon_0}.
$$

It is insensitive to the polarization or orientation of the light and shifts the entire hyperfine manifold uniformly according to the atom's induced electric dipole.

#### Vector term

The vector term is sensitive to the light polarization. Light carries angular momentum, so the interaction produces an $m_F$-dependent energy shift. The polarization-dependent factor can be written as

$$
\begin{aligned}
\left(i\vec{E}_0^{(-)}\times\vec{E}_0^{(+)}\right)_z
&=\left(i\vec{E}_0^{(-)}\times\vec{E}_0^{(+)}\right)\cdot\hat{z} \\
&=i\left(E_{0x}^*E_{0y}-E_{0y}^*E_{0x}\right).
\end{aligned}
$$

This factor has opposite signs for left- and right-circularly polarized light and vanishes for linearly polarized light.

For left-circularly polarized light,

$$
\vec{E}_L=\frac{E_0}{\sqrt{2}}(1,i,0),
$$

so

$$
\begin{aligned}
\left(i\vec{E}_0^{(-)}\times\vec{E}_0^{(+)}\right)_z
&=i\left[
\frac{E_0}{\sqrt{2}}\left(i\frac{E_0}{\sqrt{2}}\right)
-\left(-i\frac{E_0}{\sqrt{2}}\right)\frac{E_0}{\sqrt{2}}
\right] \\
&=-E_0^2.
\end{aligned}
$$

For right-circularly polarized light,

$$
\vec{E}_R=\frac{E_0}{\sqrt{2}}(1,-i,0),
$$

so

$$
\begin{aligned}
\left(i\vec{E}_0^{(-)}\times\vec{E}_0^{(+)}\right)_z
&=i\left[
\frac{E_0}{\sqrt{2}}\left(-i\frac{E_0}{\sqrt{2}}\right)
-\left(i\frac{E_0}{\sqrt{2}}\right)\frac{E_0}{\sqrt{2}}
\right] \\
&=+E_0^2.
\end{aligned}
$$

#### Tensor term

The tensor term measures the anisotropy of the atomic state relative to the light's polarization orientation. The electric field interacts differently with anisotropic, nonspherical distributions of an atom's angular momentum. This term is symmetric in $m_F$ and applies only for $F\ge1$, because only those states can possess a quadrupolar angular-momentum distribution.

## Vector Shift as an Effective Magnetic Field

The vector shift is linear in the hyperfine magnetic quantum number $m_F$. A conventional MOT exploits the Zeeman effect to localize and trap atoms. The Zeeman contribution to the energy shift is odd in $m_F$, allowing it to raise one stretched state toward resonance and lower the other stretched state away from resonance. This position-dependent energy modification produces a restoring force by favoring absorption from the beam that pushes an atom back toward the trap origin.

The vector AC Stark shift has the same qualitative behavior: it shifts energy levels linearly in $m_F$ and changes sign across $m_F=0$.

The objective of the pMOT is to suppress or eliminate the scalar and tensor terms while retaining a relatively large vector shift. The net shift would then be odd in $m_F$ and resemble a *fictitious magnetic field*.

The Zeeman energy shift is

$$
\Delta E=\mu_BgBm_F,
$$

where $g$ is the Landé $g$-factor and $\mu_B$ is the Bohr magneton.

The vector contribution to the AC Stark shift is

$$
\begin{aligned}
\Delta E
&=-\alpha^{(1)}(F;\omega)
\left(i\vec{E}_0^{(-)}\times\vec{E}_0^{(+)}\right)_z
\frac{m_F}{F} \\
&=-\frac{m_F}{F}
\left(i\vec{E}_0^{(-)}\times\vec{E}_0^{(+)}\right)_z
\sum_{F'}(-1)^{F+F'+1}
\sqrt{\frac{6F(2F+1)}{F+1}}
\begin{Bmatrix}
1&1&1\\
F&F&F'
\end{Bmatrix}
\frac{\omega_{F'F}|\langle F\|\hat{d}\|F'\rangle|^2}
{\hbar\left(\omega_{F'F}^2-\omega^2\right)}.
\end{aligned}
$$

Both the Zeeman and vector AC Stark shifts are linear in $m_F$. The vector shift may therefore be interpreted as an effective magnetic field capable of moving the stretched hyperfine states in opposite directions:

$$
\begin{aligned}
B_{\mathrm{eff}}
\equiv{}&-\frac{1}{\mu_BgF}
\left(i\vec{E}_0^{(-)}\times\vec{E}_0^{(+)}\right)_z \\
&\times\sum_{F'}(-1)^{F+F'+1}
\sqrt{\frac{6F(2F+1)}{F+1}}
\begin{Bmatrix}
1&1&1\\
F&F&F'
\end{Bmatrix}
\frac{\omega_{F'F}|\langle F\|\hat{d}\|F'\rangle|^2}
{\hbar\left(\omega_{F'F}^2-\omega^2\right)}.
\end{aligned}
$$

## Intensity Gradient

Suppose the desired effective magnetic-field gradient is $20\ \mathrm{G/cm}$. In a conventional MOT, the gradient is calculated from $\partial B/\partial z$. The field has explicit $z$ dependence through the Biot–Savart law for coils in an anti-Helmholtz configuration:

$$
B_z(z)
=\frac{1}{2}\mu_0NIR^2
\left[
\left(R^2+\left(z-\frac{d}{2}\right)^2\right)^{-3/2}
-\left(R^2+\left(z+\frac{d}{2}\right)^2\right)^{-3/2}
\right].
$$

The field gradient at the trap center is

$$
\left.\frac{dB}{dz}\right|_{z=0}
=\frac{3\mu_0NIR^2d}
{2\left(R^2+\frac{d^2}{4}\right)^{5/2}}.
$$

For the pMOT, the effective magnetic field is instead identified as

$$
B_{\mathrm{eff}}(z)
=-\frac{1}{Fg\mu_B}
\frac{2I(z)}{c\epsilon_0}
\alpha^{(1)},
$$

where the only $z$ dependence comes from the laser intensity. Therefore,

$$
\frac{dB_{\mathrm{eff}}}{dz}
=\frac{\partial B_{\mathrm{eff}}}{\partial I}
\frac{\partial I}{\partial z}
=-\frac{1}{Fg\mu_B}
\frac{2}{c\epsilon_0}
\alpha^{(1)}
\frac{\partial I}{\partial z}.
$$

For a target gradient of

$$
\frac{\partial B_{\mathrm{eff}}}{\partial z}
=20\ \mathrm{G/cm}
=0.2\ \mathrm{T/m},
$$

the required intensity gradient can be written as

$$
\frac{\partial I}{\partial z}
=\frac{0.2\ \mathrm{T/m}}{B_{\mathrm{eff}}/I},
$$

with the sign determined by the desired restoring direction, the light helicity, and the sign of $\alpha^{(1)}$.


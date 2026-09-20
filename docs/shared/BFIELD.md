\section{Magnetic Field Calculation}
Consider two coils of wire, each centered about the z-axis with radius $R$ and $N$ loops.  The first coil is centered at $z=-R/2$ and has current $I$ in each loop running in the counterclockwise direction when viewed from $+\hat{z}$.  The second coil is centered at $z=+R/2$ and has current $I$ in each loop running in the clockwise direction when viewed from $+\hat{z}$.  Thus, all loops in coil 1 can be said to have current $+I$ and all loops in coil 2 can be said to have current $-I$.

The off-axis magnetic field components can be expressed in closed-form using elliptic integrals.  We will utilize the following declarations of elliptic coordinates:

\begin{equation}
\begin{split}
    & \alpha_+^2 \equiv (R-\rho)^2 + (z+R/2)^2 \\
    & \alpha_-^2 \equiv (R-\rho)^2 + (z-R/2)^2 \\
    & \beta_+^2 = (R+ \rho)^2 + (z+R/2)^2 \\
    & \beta_-^2 = (R+ \rho)^2 + (z-R/2)^2 \\
    & k_+^2 = \frac{4R\rho}{\beta_+^2} \\
    & k_-^2 = \frac{4R\rho}{\beta_-^2}
\end{split}
\end{equation}

The two versions are used for each parameter because it depends on which loop is in consideration.  The loop centered at $z=+R/2$ is associated with parameters with the "+" subscript and the loop centered at $z=-R/2$ is associated with the "-" subscript.

The complete elliptic integrals are:

\begin{equation}
\begin{split}
    & K(k) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1-k^2\sin^2(\theta)}} \\
    & E(k) = \int_0^{\pi /2} d\theta \sqrt{1-k^2\sin^2(\theta)}
\end{split}
\end{equation}

In this configuration, the radial and axial components of the total magnetic field induced by the current in these coils are as follows:

\begin{align}
    B_\rho (\rho, z) &= \frac{\mu_0 NI}{2\pi \rho}\lbrack \frac{(z+R/2)}{\beta_+}(-K(k_+) + \frac{R^2+\rho^2+(z+R/2)^2}{\alpha_+^2}E(k_+)) \\&- \frac{(z-R/2)}{\beta_-}(-K(k_-) + \frac{R^2 + \rho^2 + (z-R/2)^2}{\alpha_-^2}E(k_-)) \rbrack \nonumber
\end{align}

\begin{align}
    B_z (\rho, z) &= \frac{\mu_0 NI}{2\pi}\lbrack \frac{1}{\beta_+}(K(k_+) + \frac{R^2-\rho^2-(z+R/2)^2}{\alpha_+^2}E(k_+)) \\&- \frac{1}{\beta_-}(K(k_-) + \frac{R^2 - \rho^2 - (z-R/2)^2}{\alpha_-^2}E(k_-)) \rbrack \nonumber
\end{align}

Then, the on-axis magnetic field vector at any point $z$ can be found using:

\begin{equation}
    \vec{B}(0,0,z) = \frac{\mu_0 NI R^2}{2} \lbrack \frac{1}{(R^2 + (z+R/2)^2)^{3/2}} - \frac{1}{(R^2 + (z-R/2)^2)^{3/2}} \rbrack \hat{z}
\end{equation}

Clearly, this enforces the vanishing of the magnetic field at the origin:

\begin{equation}
    \vec{B}(0,0,z=0) = 0\hat{z}
\end{equation}

The entire vectorial magnetic field is

\begin{equation}
    \vec{B} = B_{\rho}(\rho, z) \hat{\rho} + B_z (\rho, z)\hat{z}
\end{equation}

\section{Additional Codex Notes}
There exists a coordinate singularity at $\rho=0$ for $B_\rho (\rho,z)$.  So, at $\rho = 0$, use these components:

\begin{equation}
\begin{split}
    & B_x(0,z) = 0 \\
    & B_y(0,z) = 0 \\
    & B_z(0,z) = \frac{\mu_0 NI R^2}{2} \lbrack \frac{1}{(R^2 + (z+R/2)^2)^{3/2}} - \frac{1}{(R^2 + (z-R/2)^2)^{3/2}} \rbrack
\end{split}
\end{equation}

Another singularity exists at the current coils themselves at $(\rho = R, z=\pm R/2)$.  The implementation must not calculate field values at these points.

SciPy uses ellipk and ellipe functions that take as a parameter $m=k^2$.  So, define $m_+ =k_+^2$ and $m_- = k_-^2$ and use the $m$ variables as input for the SciPy functions.

Also, since the code currently uses Cartesian coordinates, it will be important to convert the final magnetic field vector back to Cartesian coordinates.  Since $\rho = \sqrt{x^2 + y^2}$ and $\hat{\rho}=\frac{x}{\rho}\hat{x} + \frac{y}{\rho}\hat{y}$, $B_x = B_\rho \frac{x}{\rho}, B_y = B_\rho \frac{y}{\rho}, B_z = B_z$.  Use this and again, treat the $\rho = 0$ case separately as defined above to avoid divide by zero errors.

All inputs must be in SI units.

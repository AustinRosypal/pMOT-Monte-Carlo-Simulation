### Analytical Solution to Damping Turn-Around Points as Function of Detuning

We seek to solve the equation

$$
\frac{\partial F}{\partial v}=0.
$$

Let's consider the force in the \(x\)-direction:

$$
\frac{\partial F_x}{\partial v_x}=0.
$$

The force in the \(x\)-direction is

$$
\begin{aligned}
F_x
&= \hbar k_{+\hat{x}}R_{+\hat{x}}
 + \hbar k_{-\hat{x}}R_{-\hat{x}} \\
&= \frac{\hbar}{c}
\left[
\omega_{+\hat{x}}R_{+\hat{x}}
+
\omega_{-\hat{x}}R_{-\hat{x}}
\right] \\
&= \frac{2\pi\hbar}{c}
\left[
\nu_{+\hat{x}}R_{+\hat{x}}
+
\nu_{-\hat{x}}R_{-\hat{x}}
\right] \\
&= \frac{2\pi\hbar\nu}{c}
\left[
\frac{s_{+\hat{x}}}
{1+s_{\mathrm{tot}}+\frac{4\Delta_{+\hat{x}}^2}{\Gamma^2}}
-
\frac{s_{-\hat{x}}}
{1+s_{\mathrm{tot}}+\frac{4\Delta_{-\hat{x}}^2}{\Gamma^2}}
\right] \\
&= \frac{2\pi\hbar}{\lambda}
\left[
\frac{s_{+\hat{x}}}
{1+s_{\mathrm{tot}}+\frac{4(\delta_0-k_xv_x)^2}{\gamma^2}}
-
\frac{s_{-\hat{x}}}
{1+s_{\mathrm{tot}}+\frac{4(\delta_0+k_xv_x)^2}{\gamma^2}}
\right].
\end{aligned}
$$

This can be rewritten using the facts that

$$
s_{+\hat{x}}=s_{-\hat{x}}
=
\frac{I_x}{I_{\mathrm{sat}}}
=
25.47,
$$

$$
s_{\mathrm{tot}}=6s=152.82,
$$

$$
\frac{\Delta_0}{2\pi}
=
\delta_0
=
-15\ \mathrm{MHz},
$$

$$
\frac{\Gamma}{2\pi}
=
\gamma
=
6.065\ \mathrm{MHz},
$$

and

$$
\lambda=780.24605\ \mathrm{nm}.
$$

Therefore,

$$
F_x
=
\frac{2\pi\hbar I_x}{\lambda I_{\mathrm{sat}}}
\left[
\frac{1}
{
1+s_{\mathrm{tot}}
+
\frac{4}{(6.065\ \mathrm{MHz})^2}
\left(
-15\ \mathrm{MHz}
-
\frac{v_x}{\lambda}
\right)^2
}
-
\frac{1}
{
1+s_{\mathrm{tot}}
+
\frac{4}{(6.065\ \mathrm{MHz})^2}
\left(
-15\ \mathrm{MHz}
+
\frac{v_x}{\lambda}
\right)^2
}
\right].
$$


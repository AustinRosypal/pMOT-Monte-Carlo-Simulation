# Simplified MOT Instructions

**Author:** Austin Rosypal  
**Date:** July 2026

## Instructions

We must simplify the magneto-optical trap (MOT) simulation to make it easier to verify the physical correctness and numerical implementation of the code.

Approximate each Rb-87 atom as an effective two-level atom using the D2 cooling transition

\[
5s_{1/2} \rightarrow 5p_{3/2}.
\]

This is not a complete description of the real Rb-87 hyperfine and Zeeman structure, but it should reproduce the basic restoring and damping behavior of a MOT.

Begin with all cooling beams detuned by

\[
\Delta_0/(2\pi) = -15\ \text{MHz},
\]

where negative detuning means red detuning.

Continue labeling each beam by its propagation direction and circular-polarization handedness, as in the previous multilevel simulation. In the effective two-level model, assign each beam a polarization sign

\[
\xi_j = \pm 1,
\]

which determines the sign of its effective Zeeman shift. The sign convention must be stated explicitly and tested to ensure that the magnetic force is restoring.

## Scattering Rate

For beam \(j\), use the steady-state scattering rate

\[
R_j =
\frac{\Gamma}{2}
\frac{s_j}
{1+s_{\mathrm{tot}}+\left(\frac{2\Delta_{\mathrm{eff},j}}{\Gamma}\right)^2},
\]

where

\[
s_j = \frac{I_j}{I_{\mathrm{sat}}},
\qquad
s_{\mathrm{tot}} = \sum_j s_j.
\]

The total saturation parameter appears in the denominator because all beams contribute to saturation of the same effective optical transition.

Use one frequency convention consistently:

- If \(\Gamma\), \(\Delta_0\), Doppler shifts, and Zeeman shifts are expressed in angular frequency, use rad/s everywhere.
- If they are expressed in ordinary frequency, use Hz or MHz everywhere.

Do not mix \(\Gamma\) in rad/s with detunings in MHz.

## Effective Detuning

For beam \(j\), define the effective detuning as

\[
\Delta_{\mathrm{eff},j}
=
\Delta_0
-
\mathbf{k}_j\cdot\mathbf{v}
-
\Delta_{B,j}.
\]

Here:

1. \(\Delta_0\) is the zero-field laser detuning.
2. \(\mathbf{k}_j\cdot\mathbf{v}\) is the Doppler shift for beam \(j\).
3. \(\Delta_{B,j}\) is the effective polarization-dependent Zeeman shift.

For an effective two-level MOT model, use

\[
\Delta_{B,j}
=
\xi_j\,\frac{\mu_{\mathrm{eff}} B_{\parallel,j}}{\hbar}
\]

when using angular-frequency units, or

\[
\frac{\Delta_{B,j}}{2\pi}
=
\xi_j\,\frac{\mu_{\mathrm{eff}} B_{\parallel,j}}{h}
\]

when using ordinary-frequency units.

The field projection relevant to beam \(j\) may be approximated as

\[
B_{\parallel,j}
=
\mathbf{B}(\mathbf{r})\cdot\hat{\mathbf{k}}_j.
\]

For the simplest model, one may take

\[
\mu_{\mathrm{eff}} = \mu_B,
\]

which gives

\[
\frac{\mu_B}{h}
=
1.3996245\ \frac{\text{MHz}}{\text{G}}.
\]

Therefore,

\[
\frac{\Delta_{B,j}}{2\pi}
=
\xi_j
\left(
1.3996245\ \frac{\text{MHz}}{\text{G}}
\right)
B_{\parallel,j}[\text{G}].
\]

This effective magnetic moment is a simplifying assumption. A more realistic model would use the difference between the excited-state and ground-state magnetic moments.

Use the same anti-Helmholtz magnetic-field configuration developed previously.

## Mean Radiation-Pressure Force

Do not sample individual photon-scattering events in this simplified model. Instead, calculate the deterministic mean radiation-pressure force from every cooling beam at every force evaluation.

For beam \(j\),

\[
\mathbf{F}_j
=
\hbar\mathbf{k}_j R_j.
\]

The net mean force is

\[
\mathbf{F}_{\mathrm{net}}
=
\sum_j \mathbf{F}_j.
\]

Because \(R_j\) depends on the beam-specific Doppler and Zeeman detuning, the beams closest to resonance will contribute the largest force.

This should be described as a continuous average-force model, not as the atom literally scattering one photon from every beam during every timestep.

The acceleration is

\[
\mathbf{a}
=
\frac{\mathbf{F}_{\mathrm{net}}}{m_{\mathrm{Rb}}},
\]

with

\[
m_{\mathrm{Rb}}
=
1.44316\times10^{-25}\ \text{kg}.
\]

## Equations of Motion

Integrate the coupled equations

\[
\frac{d\mathbf{r}}{dt}
=
\mathbf{v},
\]

\[
\frac{d\mathbf{v}}{dt}
=
\frac{\mathbf{F}_{\mathrm{net}}(\mathbf{r},\mathbf{v})}{m_{\mathrm{Rb}}}.
\]

Use a fourth-order Runge-Kutta method or a reliable adaptive ODE solver.

During every intermediate Runge-Kutta force evaluation, recompute:

- the magnetic field \(\mathbf{B}(\mathbf{r})\),
- each beam intensity \(I_j(\mathbf{r})\),
- each Doppler shift,
- each Zeeman shift,
- each scattering rate,
- and the net force.

Do not compute the acceleration only once at the beginning of a timestep and then reuse it for all Runge-Kutta stages.

The deterministic model will describe the mean cooling and trapping force, but it will not include momentum diffusion from spontaneous emission. Therefore, it will not by itself reproduce the Doppler-temperature equilibrium. Photon-recoil noise can be added later as a separate stochastic term.

## Plots

### Magnetic-Field Maps

Create a \(3\times3\) grid containing:

- \(B_x\) in the \(xy\), \(xz\), and \(yz\) planes,
- \(B_y\) in the \(xy\), \(xz\), and \(yz\) planes,
- \(B_z\) in the \(xy\), \(xz\), and \(yz\) planes.

Use consistent spatial limits, axis labels, units, and color scales where appropriate.

### MOT Geometry

Create a three-dimensional visualization of the six cooling beams along the \(\pm x\), \(\pm y\), and \(\pm z\) directions.

Draw each beam using the same beam-waist or beam-diameter value used previously. Show:

- beam propagation directions,
- polarization labels,
- the trap center,
- the anti-Helmholtz coil axis or magnetic-field axis,
- and the initial atomic position.

### Atomic Dynamics

Create plots of:

- \(x(t)\), \(y(t)\), and \(z(t)\),
- \(v_x(t)\), \(v_y(t)\), and \(v_z(t)\),
- the scattering rate \(R_j(t)\) from each beam,
- the total scattering rate,
- the force components \(F_x(t)\), \(F_y(t)\), and \(F_z(t)\),
- and the three-dimensional atomic trajectory.

The quantity plotted for each beam should be labeled as a scattering rate or mean force contribution, not as a discrete number of scattering events.

## Interactive Controls

Provide interactive controls for sensible simulation parameters, including:

- initial position,
- initial velocity,
- cooling-beam detuning,
- beam intensity or saturation parameter,
- magnetic-field gradient,
- beam waist,
- integration time,
- and timestep or solver tolerance.

Use a **Recompute** button so that changing a slider does not automatically rerun the entire simulation.

For Jupyter widgets, set

```python
continuous_update=False
```

for computationally expensive sliders.

Place plot output inside a dedicated output widget and clear it before redrawing:

```python
with mot_output:
    mot_output.clear_output(wait=True)
    plt.close("all")
    # recompute and redraw plots
```

## Minimum Validation Tests

Before treating the simulation as physically correct, verify the following:

1. At \(\mathbf{r}=0\) and \(\mathbf{v}=0\), equal counterpropagating beams produce approximately zero net force.
2. With the magnetic field disabled, the force opposes small atomic velocities.
3. With velocity set to zero, the magnetic and polarization configuration produces a restoring force toward the origin.
4. Reversing the beam polarizations or magnetic-field gradient reverses the restoring force and makes the trap anti-restoring.
5. With both detuning and magnetic field set to zero, symmetric beams produce no preferred direction.
6. Results converge when the timestep is reduced or the ODE-solver tolerance is tightened.
7. The force magnitude never exceeds the expected order of
   \[
   \hbar k\frac{\Gamma}{2}
   \]
   per strongly saturated beam.
8. The anti-Helmholtz field is zero at the origin and linear near the trap center.

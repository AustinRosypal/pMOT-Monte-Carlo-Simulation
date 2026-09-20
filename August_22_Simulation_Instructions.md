# August 22 Simulation Instructions

We will now run simulations to plot and observe the relationship between various quantities.

## Loading Rate as a Function of Saturation Parameter

The saturation parameter is

\[
s = \frac{I}{I_{\text{sat}}}
\]

where \(I\) is the intensity of the beam and \(I_{\text{sat}}\) is the saturation intensity. We want to observe the effect on loading rate as we vary this parameter.

So, run simulations where

\[
s = n I_{\text{sat}}
\]

for

\[
n = 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.
\]

So, it should have 29 points.

Plot the calculated loading rate in each instance on the y-axis and the \(n\) value on the x-axis, including the equation

\[
s = n I_{\text{sat}}
\]

somewhere on the figure.

I expect the shape to be logarithmic, but we will see. Put this plotting in its own macro, so we can try to fit it with a function later on.

Keep all other parameters and quantities as default. Only change the saturation parameter by changing the intensity of the beams.

The saturation intensity is

\[
I_{\text{sat}} = 1.67\ \text{mW/cm}^2.
\]

The default beam Gaussian diameter is 12.7 mm.

## Loading Rate as a Function of Beam Size

We are interested in how loading rate scales with beam size, \(d\). It will go something like

\[
R \propto d^n
\]

for one or two power laws.

Run simulations where only the beam size is altered. Maintain the same intensity.

Originally, we were doing 20 mW beams with 12.7 mm diameters. Keep that same intensity, meaning the power will change as you change the beam size.

Wisely choose 25 reasonable beam sizes, evenly spaced, that will show the shape of the fit form. Put this plotting in its own macro, so we can try to fit it to a power-law function or functions later on.

This plot will be very useful because it will enable us to compare the geometries of the MOT vs. pMOT. For example:

> "To achieve the same loading rate with the MOT and pMOT, the pMOT requires an aperture of \(x\) size whereas the MOT requires an aperture of \(x/10\) size."

## Restoring and Damping Force Curves as a Function of Detuning

I am interested in how the slope of the restoring force curves at the origin changes as we change the detuning of the beams.

I am also interested in how the location of the turning points of the damping force curves translates as we change the detuning, since detuning is what sets the Doppler speed required to maximize scattering.

For this simulation, provide two plots as a function of detuning:

1. The slope of the restoring force at the origin (\(v=0\)) for each Cartesian axis.
2. The location of the turn-around points, in units of velocity, of the cooling curves at \(r=0\) for each Cartesian axis.

## Final Ensemble Temperature as a Function of Detuning

Plot the final temperature of the ensemble as a function of detuning.

Label the detuning as

\[
\Delta = n\Gamma
\]

where \(\Gamma\) is the linewidth and \(n\) is a real number. Let the x-axis be \(n\), but include the equation somewhere on the figure.

Plot the detuning-dependent Doppler-temperature reference

\[
T_D(\Delta)=-\frac{\hbar\Gamma^2}{8k_B\Delta}
\left[1+s_{\mathrm{eff}}(\Delta)
+\left(\frac{2\Delta}{\Gamma}\right)^2\right],
\qquad \Delta<0,
\]

as a dashed curve. For the multilevel study, use the project convention

\[
s_{\mathrm{eff}}(\Delta)=
\frac{s_0}{1+(2\Delta/\Gamma)^2},
\qquad s_0=I_0/I_{\mathrm{sat}},
\]

where \(I_0\) is the Gaussian peak intensity at the center of one cooling
beam. Recompute \(s_{\mathrm{eff}}\) at every detuning; do not use the constant
\(\hbar\Gamma/(2k_B)\) overlay from the simplified two-level benchmark.

Sample 25 appropriate points for \(n\), the detuning. Plot this relationship in its own macro, so that we can fit it with a function later on.

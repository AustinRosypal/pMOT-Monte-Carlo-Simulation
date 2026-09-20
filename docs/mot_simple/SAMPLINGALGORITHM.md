# MOT Sampling

## 1. Magnetic Field Component Plot Changes

First, make a change to the magnetic-field component plots in `two_level_mot_validation.ipynb`.

We currently plot the \(B_x\), \(B_y\), and \(B_z\) components in three dimensions, with the color of the surface plot also providing information about the magnetic-field component value at each point.

This is a poor visualization because each plot against two spatial variables occurs at a fixed value of the third spatial variable.

Instead, I still want a 3D surface plot with color, but the **vertical axis and color should redundantly represent the same magnetic-field component value**.

For example:

- For the \(B_x\) over the \(x\)-\(y\) plane:
  - Horizontal axes: \(x\) and \(y\)
  - Vertical axis: \(B_x\)
  - Surface color: \(B_x\)

- For the \(B_y\) over the \(y\)-\(z\) plane:
  - Horizontal axes: \(y\) and \(z\)
  - Vertical axis: \(B_y\)
  - Surface color: \(B_y\)

Apply this same convention to all magnetic-field component plots.

---

# 2. Monte Carlo MOT Capture-Velocity Sampling

Now that we have a simple model capable of calculating scattering, position and velocity over time, force, etc., we want to conduct a Monte Carlo sampling procedure to calculate the cutoff **trappable velocity**, or capture velocity, of the MOT.

The goal is to determine the maximum incident atomic velocity, from arbitrary directions around the MOT, for which the atom becomes trapped.

Create a **new Monte Carlo sampling script/module for this task**. Keep all existing programs and functionality intact.

## 2.1 Sampling Parameters

We begin with four primary parameters:

\[
\theta,\quad \phi,\quad r,\quad v_0.
\]

Here:

- \(\theta\) is the initial polar angle.
- \(\phi\) is the initial azimuthal angle.
- \(r\) is the radial distance from the center of the MOT.
- \(v_0\) is an initial trial velocity used by the capture-velocity search.

For now, use

\[
r = 15\ \mathrm{mm},
\]

but make this parameter easily configurable.

Start the capture-velocity search around

\[
v_0 = 20\ \mathrm{m/s}.
\]

---

# 3. Sampling the Incident Direction

Choose

\[
\theta \in \left[0,\frac{\pi}{2}\right],
\]

and

\[
\phi \in \left[0,\frac{\pi}{2}\right].
\]

This spans one octant of a sphere.

Because all eight octants should be symmetric in this MOT geometry, it is redundant to independently Monte Carlo sample all of them.

Each pair

\[
(\theta,\phi)
\]

defines an incident viewing direction toward the MOT.

The geometry should be interpreted from the perspective of this direction. Use the coordinates of the MOT beams to determine how the beam configuration appears from the selected \((\theta,\phi)\) direction.

For example,

\[
(\theta=0,\phi=0)
\]

should correspond to looking directly down one of the cooling-beam axes.

---

# 4. Constructing a Sampling Disc

For each selected incident direction \((\theta,\phi)\), construct a plane perpendicular to that incident direction at radial distance \(r\) from the MOT center.

This plane represents a cross-sectional incident surface from which atoms can enter the MOT.

Sample **100 points** within this disc.

Parameterize positions within the disc using

\[
(\theta',s),
\]

where:

- \(s\) is the radial coordinate within the disc, analogous to the cylindrical \(\rho\) coordinate.
- \(\theta'\) is the angular coordinate within the disc.

Thus,

\[
s = 0
\]

corresponds to the center of the disc, while increasing \(s\) moves outward from the center.

The sampled atomic trajectory should initially point **toward the center of the MOT**, along the incident direction associated with the selected \((\theta,\phi)\).

The maximum allowed value of \(s\) should correspond to the physically relevant cross-sectional region through which an atom could interact with the MOT beams. Make the chosen disc radius configurable.

---

# 5. Capture-Velocity Search

For every sampled point

\[
(\theta,\phi,\theta',s),
\]

determine the maximum incident velocity for which the atom becomes trapped.

Denote this quantity by

\[
v_c.
\]

Use a **binary-search algorithm**.

Start using the initial velocity scale

\[
v_0 = 20\ \mathrm{m/s}.
\]

Determine whether an atom launched from the sampled point at that velocity becomes trapped.

Adjust the velocity according to the trapping result and continue narrowing the velocity interval.

The goal is to determine the boundary separating trapped and non-trapped trajectories.

Continue iterating until the capture velocity is determined to within

\[
0.25\ \mathrm{m/s}.
\]

The resulting quantity should therefore be approximately

\[
v_c(\theta,\phi,\theta',s,r).
\]

If an additional distance variable \(d\) is required by the existing geometry convention, retain and record it explicitly. Otherwise, use \(r\) consistently as the distance of the sampling plane from the MOT center.

### Important binary-search requirement

The algorithm must first establish a valid bracket:

- one velocity that is trapped;
- one velocity that is not trapped.

If the initial \(v_0\) does not provide such a bracket, automatically increase or decrease the trial velocity as necessary until the transition between trapped and non-trapped behavior is bracketed.

Then perform the binary search inside that interval.

---

# 6. Definition of a Trapped Atom

We need a computational criterion for determining whether an atom has become trapped.

For this initial implementation:

> If the atom oscillates within the MOT twice, classify the atom as trapped.

Implement this criterion robustly from the simulated trajectory.

The code should distinguish genuine trapping oscillations from a particle that merely passes through the MOT or crosses the origin several times while escaping.

The trajectory simulation should therefore terminate when either:

1. the atom satisfies the two-oscillation trapping criterion, or
2. the atom clearly escapes the trapping region, or
3. a configurable maximum simulation time is reached.

Record the trapping result and, where useful, the reason the simulation terminated.

---

# 7. Monte Carlo Sampling Size

For each incident direction \((\theta,\phi)\):

1. Construct the corresponding incident disc.
2. Sample 100 points described by \((\theta',s)\).
3. Determine \(v_c\) for each of these points.

Then randomly select another

\[
(\theta,\phi)
\]

within the octant and repeat the entire procedure.

Sample **100 incident discs**.

Therefore:

\[
100\ \text{discs}
\times
100\ \text{points per disc}
=
10,000\ \text{sampled trajectories/positions}.
\]

Note that determining each capture velocity requires multiple trajectory simulations because of the binary search.

---

# 8. Sampling Requirements

Sampling should be statistically appropriate.

For the spherical directions, sample directions uniformly in **solid angle**, rather than simply drawing \(\theta\) uniformly if that would bias the distribution.

For a uniform solid-angle distribution within the octant, the appropriate sampling should account for

\[
d\Omega = \sin\theta\,d\theta\,d\phi.
\]

Similarly, when sampling points uniformly over the area of each disc, account for the area element

\[
dA = s\,ds\,d\theta'.
\]

Therefore, do not simply sample \(s\) uniformly if the goal is uniform sampling over the disc area.

Keep the random seed configurable so that Monte Carlo runs can be reproduced.

---

# 9. Data Storage

Store all results in a format that can easily be analyzed after the simulation has completed.

For every sampled point, record at minimum:

- disc index;
- point index;
- \(\theta\);
- \(\phi\);
- \(\theta'\);
- \(s\);
- \(r\);
- capture velocity \(v_c\);
- capture-velocity uncertainty or binary-search resolution;
- Cartesian initial position \((x_0,y_0,z_0)\);
- Cartesian incident velocity direction;
- final trapped/not-trapped state at the velocity bounds used to determine \(v_c\).

Also store useful metadata describing the simulation configuration, including relevant MOT parameters.

Prefer a standard analysis-friendly format such as CSV, Parquet, or another format already consistent with this project.

Do not store the results only in memory or only in notebook variables.

---

# 10. Diagnostic Plots

Once the data have been collected, produce diagnostic plots.

## Capture Velocity vs. Disc Radius

For every sampled disc, produce a plot of

\[
v_c
\]

as a function of

\[
s.
\]

Each plot should contain the 100 sampled points belonging to that disc.

Therefore, each disc should give a plot of

\[
v_c(s).
\]

Because \(\theta'\) varies between points, preserve \(\theta'\) in the stored data even though the initial diagnostic plot uses \(s\) as its horizontal coordinate.

---

# 11. Disc Geometry Visualization

For each sampled disc, it would also be useful to visualize what the sampled configuration looks like.

Produce a visualization showing:

- the MOT center;
- the relevant cooling-beam geometry;
- the orientation of the selected incident plane;
- the sampled points within the disc;
- the incident direction corresponding to \((\theta,\phi)\).

Where practical, display this geometry visualization next to the corresponding

\[
v_c \text{ vs. } s
\]

plot.

The purpose is to make it visually obvious which MOT cross-section produced the associated capture-velocity distribution.

---

# 12. Code Organization

Do not disrupt or replace the existing MOT simulation.

Instead:

- reuse the existing two-level MOT physics implementation;
- create a new Monte Carlo capture-velocity sampling script/module;
- keep simulation parameters configurable;
- separate trajectory simulation, capture classification, binary search, sampling, data storage, and plotting into sensible functions;
- avoid unnecessarily duplicating physics already implemented elsewhere in the project.

The Monte Carlo driver should call the existing MOT physics rather than implementing a second independent version of the MOT model.

---

# 13. Computational Practicality

The complete requested run contains 10,000 spatial samples, with multiple trajectory simulations required for every binary search.

Therefore, structure the program so that:

- a small test run can be performed first;
- the number of discs is configurable;
- the number of points per disc is configurable;
- the capture-velocity precision is configurable;
- the random seed is configurable;
- progress through the Monte Carlo calculation is visible;
- partial results can be periodically saved so that a long run is not lost if interrupted.

For validation, initially allow configurations such as:

```text
number_of_discs = 2
points_per_disc = 5
```

before executing the full

```text
number_of_discs = 100
points_per_disc = 100
```

run.

---

# 14. Longer-Term Physics Objective

This calculation is ultimately intended to determine the **loading rate of the MOT**.

The MOT loading rate depends on the atomic flux together with the velocity-space and geometric acceptance of the trap.

The capture-velocity data obtained from these different incident cross-sections will eventually be used to determine an effective or net capture behavior for the MOT.

That result can then be combined with the flux and velocity distribution of rubidium atoms to estimate the simulated MOT loading rate.

For now, focus on generating and storing sufficiently complete capture-velocity data so that this later loading-rate calculation can be performed without needing to rerun the entire Monte Carlo simulation.
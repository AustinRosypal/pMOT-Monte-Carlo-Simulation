# Two local-effective-field-axis pMOT trajectories

This is a provisional vector-only stretched-transition diagnostic. At each
time sample the six 1529-nm component fields are reconstructed and summed,
the normalized sum supplies the local quantization-axis proxy, and all 12
cooling/repump polarizations are projected into that spherical basis before
the inherited 24-state rate solve.

- trapping scale: `38.294486173 mW/path`
- central transition-equivalent gradient target: `20 G/cm`
- primary timestep: `2.500 us`
- duration: `25.000 ms`
- gravity: on; recoil diffusion: off

The axial launch is classified `bounded_core_residence` with a
minimum radius of `0.001068 mm`. The nonsymmetric
three-dimensional launch is classified `escaped`
with a minimum radius of `9.925272 mm`.

The axial ray crosses the ideal 2.234-micrometre waist. Consequently its
sampled field proxy reaches `2.31577e+06 G`
and the local-axis history contains large jumps. That peak is not timestep
resolved and must not be interpreted as a quantitative physical field.

The CSV files retain SI trajectory quantities, all six reconstructed beamwise
field vectors, and all sigma+/pi/sigma- fractions for the 12 cooling/repump
and six trapping components at every stored sample.

These runs omit scalar/tensor shifts by the explicitly imposed ideal-magic
assumption, conservative 1529-nm force, trap-light scattering/heating/loss,
coherent interference, measured Jones transformations, and nonadiabatic
dynamics at the fictitious-field zero. A heuristic core-residence result is
not evidence of a dynamically stable three-dimensional pMOT.

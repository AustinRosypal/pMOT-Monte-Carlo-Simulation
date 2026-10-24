# Vector-only effective two-level pMOT

`pmot_vec_2Level` is a controlled diagnostic derived from `pMOT_2Level`.
It uses the same six 1529-nm trapping-beam envelopes, atom-frame wavelength
lookup, vector-defined local quantization axis, six cooling beams, and
deterministic mean-force integrator, but forces the differential scalar and
tensor polarizabilities to zero. Only the differential vector AC Stark shift
changes the cooling-transition detuning.

The package also constructs six 780.232684-nm repump components at zero
detuning and 0.1 mW/component by default. They are intentionally passive in the
strict effective F=2 to F'=3 dynamics: the two-level state space contains no
F=1 population for a repumper to return. Active repumping requires the
state-resolved multilevel population kernel.

This remains a provisional diagnostic, not a physical multilevel pMOT. It does
not include a state-resolved Stark Hamiltonian, conservative dipole force,
trapping-light scattering/heating, coherent standing waves, or nonadiabatic
dynamics at the vector-shift zero.

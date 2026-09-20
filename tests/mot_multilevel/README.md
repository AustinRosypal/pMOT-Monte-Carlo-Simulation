# Multilevel MOT rebuild tests

Tests here validate the replacement physical multilevel kernel: ARC atomic
constants, Section-12 rates, the two-level analytic limit, the 24-state matrix,
force symmetries, RK4 trajectory plumbing, diagnostic persistence and plots,
fail-closed capture bracketing, launch sampling, and notebook integrity.
Archived regression tests live in `tests/mot_error` and only verify
preservation of that code; they do not validate its physics.

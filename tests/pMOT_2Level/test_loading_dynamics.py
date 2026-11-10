from itertools import product

import numpy as np

from pmot.configuration import PLANCK_CONSTANT_J_S
from pmot.pMOT_2Level.loading_dynamics import local, prepare, step
from pmot.pMOT_2Level.loading_scan import (
    GEOMETRIES, RANKS, WAVELENGTHS_NM, Setting, cooling_components, prepare_reference,
    simple_config,
)
from pmot.pMOT_2Level.simulation import evaluate_pmot_2level_local_observable, rk4_step_pmot_2level
from pmot.state import AtomState


def test_cartesian_fast_force_and_signed_gradient_match_reference():
    fallback = np.array([0., 0., 1.])
    points = (
        ([0, 0, 0], [0, 0, 0]),
        ([.00001, -.000017, .000025], [0, 0, 0]),
        ([-.01, .000001, .000002], [45, .17, -1.24]),
        ([.01, .000001, .000002], [-67, -.17, 1.24]),
        ([.00001, .00002, -.009999], [.01, 0, -45]),
        ([.002, -.001, .004], [3.8, 7., 22.]),
    )
    cooling = cooling_components()
    simple = simple_config()
    for ranks, wave, geo, gradient in product(RANKS, WAVELENGTHS_NM, GEOMETRIES, (2, 40)):
        trap, table, powers, _, signed = prepare_reference(Setting(ranks, wave, geo, gradient))
        data = prepare(trap, table, powers)

        def reference(r, v):
            return evaluate_pmot_2level_local_observable(
                cooling, trap, r, v, path_powers_w=powers, previous_axis=fallback,
                simple_config=simple, polarizability_table=table,
            )

        for r, v in points:
            ref = reference(r, v)
            a, n = local(np.asarray(r, dtype=float), np.asarray(v, dtype=float), fallback, *data)
            np.testing.assert_allclose(a, ref.acceleration_m_per_s2, rtol=2e-9, atol=2e-8)
            np.testing.assert_allclose(n, ref.stark.quantization_axis, atol=1e-10)
        d = 1e-7
        jac = np.column_stack([(np.array(reference(dr, [0]*3).stark.net_vector_energy_vector_j)
                                - reference(-dr, [0]*3).stark.net_vector_energy_vector_j)/(2*d)
                               for dr in d*np.eye(3)])
        jac *= 100 / (simple.effective_magnetic_moment_hz_per_t * PLANCK_CONSTANT_J_S)
        np.testing.assert_allclose(jac, np.diag(signed), rtol=1e-8, atol=1e-10)


def test_fast_rk4_step_matches_reference_at_trapping_waist():
    trap, table, powers, _, _ = prepare_reference(Setting('svt', WAVELENGTHS_NM[1], GEOMETRIES[1], 40))
    r, v = np.array([1e-6, 1.5e-6, -9.98e-3]), np.array([-.15, .23, -14.])
    fallback = np.array([0., 0., 1.])
    dt = 2.5e-6
    state, obs = rk4_step_pmot_2level(
        cooling_components(), trap, AtomState(tuple(r), tuple(v)), dt,
        path_powers_w=powers, simple_config=simple_config(), polarizability_table=table,
    )
    rn, vn, nn = step(r, v, dt, fallback, prepare(trap, table, powers))
    np.testing.assert_allclose(rn, state.position_m, atol=1e-14)
    np.testing.assert_allclose(vn, state.velocity_m_per_s, atol=1e-12)
    np.testing.assert_allclose(nn, obs.stark.quantization_axis, atol=1e-10)

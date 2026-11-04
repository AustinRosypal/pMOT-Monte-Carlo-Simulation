"""Inspect an off-core pocket without changing the running capture campaign."""
from pathlib import Path
import os
import json
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/statistics/mot_strange/calibrated_10_10_20Gpcm_full_sphere_r15mm_incell_20261025'
manifest = json.loads((OUTPUT/'campaign_manifest.json').read_text())
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'outputs/compiled_cache'/manifest['source_sha256'])
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.optimize import root
from pmot.mot_strange.accelerated_surrogate import acceleration, field, integrate
from pmot.mot_multilevel.relationship_campaign import write_json


def main():
    with np.load(OUTPUT/'compiled_inputs.npz') as saved:
        data = tuple(saved[f'array_{i}'] for i in range(12))
    with np.load(OUTPUT/'pilot/geometry.npz') as saved:
        r0 = saved['positions_m'][0, 4]; direction = saved['directions'][0]
    attempts = []
    for refinement in (1., 2.):
        began = time.perf_counter()
        result = integrate(r0, 20.*direction, 2., data, refinement)
        attempts.append(dict(refinement=refinement, code=int(result[0]), time_s=float(result[1]),
            position_m=result[2].tolist(), velocity_m_per_s=result[3].tolist(),
            accepted_steps=int(result[6]), wall_s=time.perf_counter()-began))
    seed = np.asarray(attempts[-1]['position_m'])
    axis = field(seed, data); axis /= np.linalg.norm(axis)
    solution = root(lambda r: acceleration(r, np.zeros(3), axis, data), seed, tol=1e-11)
    equilibrium = solution.x
    residual = acceleration(equilibrium, np.zeros(3), axis, data)
    jacobians = []
    for dr, dv in ((1e-7, 1e-3), (5e-8, 5e-4), (2.5e-8, 2.5e-4)):
        matrix = np.zeros((6, 6)); matrix[:3, 3:] = np.eye(3)
        for i in range(3):
            shift = np.eye(3)[i]*dr
            matrix[3:, i] = (acceleration(equilibrium+shift, np.zeros(3), axis, data)
                             - acceleration(equilibrium-shift, np.zeros(3), axis, data))/(2*dr)
            velocity = np.eye(3)[i]*dv
            matrix[3:, 3+i] = (acceleration(equilibrium, velocity, axis, data)
                               - acceleration(equilibrium, -velocity, axis, data))/(2*dv)
        eig = np.linalg.eigvals(matrix)
        jacobians.append(dict(position_step_m=dr, velocity_step_m_per_s=dv,
            full_phase_space_jacobian=matrix.tolist(),
            eigenvalues_per_s=[[float(x.real), float(x.imag)] for x in eig],
            locally_asymptotically_stable=bool(np.max(eig.real) < 0)))
    report = dict(source_sha256=manifest['source_sha256'], disc=0, point=4,
        speed_m_per_s=20., two_second_adaptive_results=attempts,
        equilibrium_position_m=equilibrium.tolist(), equilibrium_radius_m=float(np.linalg.norm(equilibrium)),
        acceleration_residual_m_per_s2=residual.tolist(), root_success=bool(solution.success),
        jacobians=jacobians,
        interpretation='local stability of an off-core equilibrium; does not override the central 2 mm capture definition')
    write_json(OUTPUT/'validation/offcore_equilibrium_probe.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

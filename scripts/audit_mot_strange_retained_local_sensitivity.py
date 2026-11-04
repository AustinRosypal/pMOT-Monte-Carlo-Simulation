"""Read saved traces; compare public kernels and local deterministic Jacobians."""
from pathlib import Path
import shutil
import numpy as np
import mot_strange_retained_capture as model
from pmot.configuration import RB87_MASS_KG


def main():
    original = model.original
    root = model.OUTPUT
    folder = root/'validation/production_failed_node_20261027/endpoint_trace_v1'
    target = folder/'local_sensitivity.json'
    assert not target.exists(), 'Read existing sensitivity evidence instead of duplicating'
    manifest = original.read(root/'campaign_manifest.json')
    assert original.SOURCE_SHA == manifest['source_sha256']
    trace_manifest = original.read(folder/'manifest.json')
    assert trace_manifest['physical_source_sha256'] == original.SOURCE_SHA
    data = model.parent.load_data()
    cfg = original.POINT.config()
    atomic = original.build_rate_equation_model()
    beams = original.build_multilevel_mot_beams(config=cfg)
    rows = []
    for path in sorted(folder.glob('fixed*.npz')):
        with np.load(path) as saved:
            trace = saved['trace']
        for sample in trace:
            t, r, v, axis = sample[0], sample[1:4], sample[4:7], sample[7:10]
            fast = model.fast.observable(r, v, axis, data)
            row = dict(trace_file=path.name, trace_sha256=original.sha(path),
                time_s=float(t), position_m=r.tolist(), velocity_m_per_s=v.tolist(),
                field_norm_t=float(sample[10]), field_epsilon_t=float(data[10][3]),
                accelerated_force_n=fast[0].tolist(),
                spontaneous_scattering_per_s=float(np.dot(data[8], fast[1][8:])))
            try:
                ref = original.rate_equation_observable(atomic, beams, tuple(r), tuple(v),
                    None, cfg, tuple(axis), magnetic_field_function=lambda p: model.fast.field(np.array(p), data),
                    store_rate_matrix=True)
                delta = np.linalg.norm(fast[0]-np.array(ref.force_n))
                row.update(reference_force_n=list(ref.force_n), force_difference_n=float(delta),
                    force_relative_difference=float(delta/max(np.linalg.norm(ref.force_n), 1e-30)),
                    population_max_abs_difference=float(np.max(np.abs(fast[1]-ref.populations))))
            except Exception as error:
                row['reference_error'] = repr(error)
            if t >= .005:
                jacobians, eigenvalues = [], []
                for dr, dv in ((1e-7, 1e-5), (1e-8, 1e-6)):
                    jac = np.zeros((6, 6))
                    jac[:3, 3:] = np.eye(3)
                    for j in range(6):
                        rp, rm, vp, vm = r.copy(), r.copy(), v.copy(), v.copy()
                        h = dr if j < 3 else dv
                        if j < 3:
                            rp[j] += h; rm[j] -= h
                        else:
                            vp[j-3] += h; vm[j-3] -= h
                        jac[3:, j] = (model.fast.acceleration(rp, vp, axis, data)
                                      - model.fast.acceleration(rm, vm, axis, data))/(2*h)
                    eig = np.linalg.eigvals(jac)
                    jacobians.append(jac.tolist())
                    eigenvalues.append([[float(x.real), float(x.imag)] for x in eig])
                row.update(jacobians=jacobians, eigenvalues_per_s=eigenvalues,
                    maximum_real_eigenvalues_per_s=[max(e[0] for e in values) for values in eigenvalues])
            rows.append(row)
    record = dict(at_utc=original.utc(), physical_source_sha256=original.SOURCE_SHA,
        diagnostic_sha256=original.sha(__file__), samples=rows,
        finite_difference_settings=[dict(position_step_m=1e-7,velocity_step_m_s=1e-5),
                                    dict(position_step_m=1e-8,velocity_step_m_s=1e-6)],
        purpose='local sensitivity of the failed trajectory; eigenvalues do not prove global stability',
        campaign_acceptance=False, original_outputs_unchanged=True)
    shutil.copy2(__file__, folder/Path(__file__).name)
    original.write_json(target, record)
    print(dict(samples=len(rows), reference_errors=sum('reference_error' in r for r in rows),
        max_reference_force_relative_difference=max(r.get('force_relative_difference',0) for r in rows),
        selected=[dict(t=r['time_s'], field=r['field_norm_t'],
                       max_real_eigenvalues=r.get('maximum_real_eigenvalues_per_s'))
                  for r in rows if r['time_s'] >= .05]))


if __name__ == '__main__':
    main()

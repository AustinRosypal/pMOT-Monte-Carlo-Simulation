"""Bounded 5 ms convergence diagnostic; never a 2 s loading certificate."""
from pathlib import Path
import shutil
import time
import numpy as np
import mot_strange_retained_capture as model
import mot_strange_retained_numerics as fixed
import mot_strange_retained_budget as adaptive


def main():
    original = model.original
    root = model.OUTPUT
    folder = root/'validation/production_failed_node_20261027/initial_transient'
    folder.mkdir(exist_ok=True)
    assert not (folder/'completion.json').exists(), 'Read completed diagnostic'
    ray = root/'production/rays/d210_p000.npz'
    manifest = original.read(root/'campaign_manifest.json')
    assert original.SOURCE_SHA == manifest['source_sha256']
    with np.load(ray) as a:
        r, direction = a['position_m'], a['direction']
    settings = [('fixed', h*1e-6) for h in (1.25, .625, .3125, .15625, .078125, .0390625, .01953125)]
    settings += [('adaptive', 512.), ('adaptive', 1024.)]
    original.write_json(folder/'manifest.json', dict(duration_s=.005, speed_m_s=33.625,
        settings=settings, physical_source_sha256=original.SOURCE_SHA,
        original_ray_sha256=original.sha(ray), source_sha256=original.sha(__file__),
        fixed_observation_sha256=original.sha(fixed.__file__),
        budgeted_adaptive_sha256=original.sha(adaptive.__file__), campaign_acceptance=False))
    for p in (Path(__file__), Path(fixed.__file__), Path(adaptive.__file__)):
        shutil.copy2(p, folder/p.name)
    data = model.parent.load_data()
    cases = []
    for method, parameter in settings:
        start = time.perf_counter()
        result = (fixed.classify(r, 33.625*direction, parameter, .005, data) if method == 'fixed'
                  else adaptive.retained_adaptive(r, 33.625*direction, .005, data, parameter))
        row = dict(method=method, parameter=parameter, actual_time_s=float(result[1]),
            position_m=result[2].tolist(), velocity_m_s=result[3].tolist(),
            first_core_s=float(result[4]), wall_s=time.perf_counter()-start)
        cases.append(row)
        original.write_json(folder/'progress.json', dict(completed=len(cases), requested=len(settings), cases=cases))
    reference = cases[-1]
    for row in cases:
        row['position_difference_from_adaptive1024_m'] = float(np.linalg.norm(
            np.array(row['position_m'])-reference['position_m']))
        row['velocity_difference_from_adaptive1024_m_s'] = float(np.linalg.norm(
            np.array(row['velocity_m_s'])-reference['velocity_m_s']))
    original.write_json(folder/'completion.json', dict(at_utc=original.utc(), cases=cases,
        campaign_acceptance=False, observation_horizon_s=.005,
        original_ray_unchanged=original.sha(ray) == original.read(folder/'manifest.json')['original_ray_sha256']))
    print(cases)


if __name__ == '__main__':
    main()

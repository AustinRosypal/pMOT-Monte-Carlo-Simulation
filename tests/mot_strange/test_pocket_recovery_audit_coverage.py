"""Recovered pockets require replay evidence even if their trace was lost."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_mot_strange_dense_pocket_recovery import missing_checkpoint_coverage


def test_lost_trace_leaves_a_required_recovered_node_unverified(tmp_path):
    folder = tmp_path / "configuration_062"
    folder.mkdir()
    ray = folder / "disc_0018_point_02.npz"
    pockets = {"124": {"status": "no_nearby_equilibrium", "equilibrium_search_fallback": {}},
               "125": {"status": "no_nearby_equilibrium"}}
    np.savez(ray, pocket_json=json.dumps(pockets))
    assert missing_checkpoint_coverage(tmp_path, []) == [{"checkpoint": str(ray), "node": 124}]
    assert missing_checkpoint_coverage(tmp_path, [{"checkpoint": str(ray), "node": 125}])
    assert missing_checkpoint_coverage(tmp_path, [{"checkpoint": str(ray), "node": 124}]) == []


def test_replay_cache_requires_unchanged_trace_checkpoint_and_auditor(tmp_path, monkeypatch):
    from hashlib import sha256
    import audit_mot_strange_dense_pocket_recovery as auditor
    import mot_strange_pocket_recovery as recovery
    from pmot.mot_strange import loading_scan_retention as retention

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    root = tmp_path / 'output'
    monkeypatch.setattr(auditor, 'ROOT', root)
    monkeypatch.setattr(auditor, 'configure_worker', lambda: None)
    source = Path('scripts/mot_strange_pocket_recovery.py')
    source.parent.mkdir()
    source.write_text('fixture source identity')
    source_sha = sha256(source.read_bytes()).hexdigest()
    revision = root / 'revisions/dense_pocket_recovery_20261108'
    revision.mkdir(parents=True)
    (revision / 'pre_resume_validation.json').write_text(json.dumps({
        'source_sha256': {str(source): source_sha}, 'missing_rays': []}))
    (root / 'pilot_manifest.json').write_text(json.dumps({'source_sha256': {}}))
    published = root / 'pilot/configuration_062'
    published.mkdir(parents=True)
    data = [np.empty(0) for _ in range(12)]
    data[11] = np.r_[np.zeros(15), .014]
    np.savez(published / 'runtime_inputs.npz', *data)
    signature = auditor.digest(data)
    zero = np.zeros(3)
    center = np.array([.005, 0., 0.])
    observation = (2, 2., center, zero, -1., center, .005, .005, 0.)
    packed = np.r_[observation[0], observation[1], observation[2], observation[3],
                   observation[4], observation[5], observation[6:]]
    pocket = {'status': 'no_nearby_equilibrium', 'equilibrium_search_fallback': {
        'rejected_probe_position_m': [.1, 0., 0.]}}
    folder = tmp_path / 'local/pmot-campaigns' / auditor.NAME / 'pilot/configuration_062'
    traces = folder / 'pocket_search_recovery'
    traces.mkdir(parents=True)
    ray = folder / 'disc_0000_point_00.npz'
    payload = dict(prepared_sha256=signature, attempts=np.array([packed]),
                   attempt_keys=np.array([['0', 'adaptive', '2.0']]),
                   pocket_json=json.dumps({'0': pocket}), position=center,
                   direction=np.array([1., 0., 0.]), speeds=np.array([1.]))
    np.savez(ray, **payload)
    trace = traces / 'witness.json'
    record = dict(configuration=62, disc=0, point=0, prepared_sha256=signature,
                  fallback_source_sha256=source_sha, cell_half_length_m=.014,
                  observation=packed.tolist(), pocket=pocket)
    trace.write_text(json.dumps(record))
    calls = []
    monkeypatch.setattr(retention, 'observe', lambda *args: calls.append(1) or observation)
    monkeypatch.setattr(recovery, 'check_pocket', lambda *args: pocket)
    auditor.audit()
    auditor.audit()
    assert len(calls) == 1
    trace.write_text(json.dumps({**record, 'checked_utc': 'changed'}))
    auditor.audit()
    assert len(calls) == 2
    np.savez(ray, **payload, retained_extra_metadata=1)
    auditor.audit()
    assert len(calls) == 3
    path = root / 'dense_pocket_recovery_audit.json'
    report = json.loads(path.read_text())
    path.write_text(json.dumps({**report, 'auditor_sha256': 'different source'}))
    auditor.audit()
    assert len(calls) == 4

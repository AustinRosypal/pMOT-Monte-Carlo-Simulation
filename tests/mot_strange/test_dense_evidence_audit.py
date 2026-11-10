"""The independent dense reducer must retain the declared fail-closed gates."""

from pathlib import Path
import sys
import json
from hashlib import sha256

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_mot_strange_dense_pilot import SPEEDS, edge_neighbours, pair_codes, raw_mask_proof, extra_fixed_proof
from pmot.mot_strange.loading_scan_retention import agreed_category


def observation(code):
    if code == 0:
        return np.array([0, .1, .02, 0, 0, 0, 0, 0, -1, 0, 0, 0, np.inf, 0, 0])
    return np.array([code, 2., .001 if code == 1 else .003, 0, 0, 0, 0, 0,
                     .01 if code == 1 else -1, .001, 0, 0,
                     .0008 if code == 1 else .0029,
                     .0012 if code == 1 else .0031, .0004])


def unpack(row):
    return row[0], row[1], row[2:5], row[5:8], row[8], row[9:12], *row[12:]


@pytest.mark.parametrize("code", [0, 1, 2, 3])
def test_vector_gate_matches_declared_scalar_gate(code):
    a = observation(code)
    rng = np.random.default_rng(927)
    for _ in range(30):
        b = a.copy()
        b[2:5] += rng.normal(size=3) * 1e-5
        expected = agreed_category(unpack(a), unpack(b))
        assert pair_codes(a[None], b[None])[0] == (-1 if expected is None else expected)


def evidence(codes):
    keys, rows = [], []
    required = edge_neighbours(codes) | {0}
    for i, code in enumerate(codes):
        for method, steps in [("adaptive", (1., 2.)), *([("fixed", (5e-6, 2.5e-6))] if i in required else [])]:
            for step in steps:
                keys.append((i, method, step))
                rows.append(observation(code))
    return np.asarray(keys), np.array(rows)


def test_closed_capture_edges_pass():
    codes = np.array([0, 0, 1, 1, 0, 0, 0])
    keys, rows = evidence(codes)
    fixed = raw_mask_proof(codes, keys, rows, [], .014)
    assert edge_neighbours(codes).issubset(fixed)


def test_missing_edge_fixed_evidence_is_rejected():
    codes = np.array([0, 0, 1, 1, 0, 0, 0])
    keys, rows = evidence(codes)
    keep = ~((keys[:, 0] == "4") & (keys[:, 1] == "fixed"))
    with pytest.raises(ValueError):
        raw_mask_proof(codes, keys[keep], rows[keep], [], .014)


def test_mask_change_is_rejected():
    codes = np.zeros(5, dtype=int)
    keys, rows = evidence(codes)
    with pytest.raises(ValueError):
        raw_mask_proof(np.array([0, 0, 1, 0, 0]), keys, rows, [], .014)


def test_false_wall_exit_is_rejected():
    codes = np.zeros(5, dtype=int)
    keys, rows = evidence(codes)
    rows[0, 2] = .001
    with pytest.raises(ValueError, match="cell exit"):
        raw_mask_proof(codes, keys, rows, [], .014)


def test_neighbour_proof_binds_original_inputs_and_category(tmp_path):
    path = tmp_path / "ray.npz"
    path.write_bytes(b"original")
    folder = tmp_path / "fixed_neighbour_review"
    folder.mkdir()
    row = observation(0)
    attempts = [dict(method="fixed", resolution=s, code=0, elapsed_s=float(row[1]),
                     position_m=row[2:5].tolist(), velocity_m_per_s=row[5:8].tolist(),
                     first_core_s=-1., window_anchor_m=row[9:12].tolist(), window_metrics_m=[None, 0., 0.])
                for s in (5e-6, 2.5e-6)]
    report = dict(ray=path.name, original_checkpoint_sha256=sha256(path.read_bytes()).hexdigest(),
                  prepared_sha256="inputs", horizon_s=2., window_s=.1,
                  observer_source_sha256=sha256(Path("src/pmot/mot_strange/loading_scan_retention.py").read_bytes()).hexdigest(),
                  nodes=[dict(node=2, speed_m_per_s=float(SPEEDS[2]), fixed_pair_s=[5e-6, 2.5e-6],
                              fixed_agreement=0, attempts=attempts)])
    source = folder / "ray_test.json"
    source.write_text(json.dumps(report))
    assert extra_fixed_proof(path, "inputs", np.zeros(len(SPEEDS), dtype=int))[0] == {2}
    # A converged fixed label cannot silently replace an original definitive category.
    changed = np.zeros(len(SPEEDS), dtype=int)
    changed[2] = 3
    assert not extra_fixed_proof(path, "inputs", changed)[0]
    report["horizon_s"] = 3.
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="inputs"):
        extra_fixed_proof(path, "inputs", changed)

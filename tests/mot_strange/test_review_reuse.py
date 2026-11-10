"""A saved integration attempt can be reused only with identical provenance."""

import copy
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from mot_strange_review_overlays import verify_reused_attempts


@pytest.fixture
def evidence(tmp_path):
    original = {
        "configuration": 65, "checkpoint_stage": "production", "ray": "ray.npz",
        "prepared_sha256": "inputs", "original_checkpoint_sha256": "checkpoint",
        "nodes": [{"node": 209, "speed_m_per_s": 26.125, "attempts": [{
            "method": "fixed", "resolution": 1.953125e-8, "code": 2,
            "position_m": [0., 0., .006], "worker_seconds": 100.,
        }]}],
    }
    source = tmp_path / "original.json"
    source.write_text(json.dumps(original), encoding="utf-8")
    document = copy.deepcopy(original)
    document["nodes"][0]["attempts"][0]["reused_from"] = {
        "file": source.name, "sha256": sha256(source.read_bytes()).hexdigest(),
    }
    return tmp_path / "new.json", document, source


def test_identical_fixed_attempt_is_accepted(evidence):
    path, document, _ = evidence
    verify_reused_attempts(path, document)


@pytest.mark.parametrize("change", ["bytes", "inputs", "position", "speed", "node", "method", "path", "chain"])
def test_modified_provenance_is_rejected(evidence, change):
    path, document, source = evidence
    attempt = document["nodes"][0]["attempts"][0]
    if change == "bytes":
        source.write_text(source.read_text() + " ", encoding="utf-8")
    elif change == "inputs":
        document["prepared_sha256"] = "different"
    elif change == "position":
        attempt["position_m"][2] += 1e-9
    elif change in ("speed", "node"):
        document["nodes"][0]["speed_m_per_s" if change == "speed" else "node"] += 1
    elif change == "method":
        attempt["method"] = "adaptive"
    elif change == "path":
        attempt["reused_from"]["file"] = "../original.json"
    else:
        original = json.loads(source.read_text())
        original["nodes"][0]["attempts"][0]["reused_from"] = {"file": "older.json"}
        source.write_text(json.dumps(original), encoding="utf-8")
        attempt["reused_from"]["sha256"] = sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verify_reused_attempts(path, document)


def test_original_reports_remain_compatible(evidence):
    path, _, source = evidence
    verify_reused_attempts(path, json.loads(source.read_text()))


def test_final_retention_check_rechecks_original_source(evidence, tmp_path):
    from audit_mot_strange_sampled_production import verify_retained_evidence

    _, document, source = evidence
    folder = tmp_path / "retained"
    reviews = folder / "numerical_review"
    reviews.mkdir(parents=True)
    raw = folder / "disc_0000_point_00.npz"
    raw.write_bytes(b"immutable checkpoint")
    (reviews / source.name).write_bytes(source.read_bytes())
    report = reviews / "disc_0000_point_00_new.json"
    report.write_text(json.dumps(document), encoding="utf-8")
    result = {
        "checkpoint_stream_sha256": sha256(sha256(raw.read_bytes()).digest()).hexdigest(),
        "numerical_review_sha256": {raw.name: {"209": sha256(report.read_bytes()).hexdigest()}},
    }
    spec = {"discs": 1, "points_per_disc": 1}
    verify_retained_evidence(folder, spec, result)
    with (reviews / source.name).open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        verify_retained_evidence(folder, spec, result)

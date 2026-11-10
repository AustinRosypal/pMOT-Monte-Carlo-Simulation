"""Verify independent numerical resolutions without replacing raw checkpoints."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from mot_strange_sampled_design import FINE_GRID
from pmot.mot_strange.loading_scan_retention import OFF_CORE, agreed_category


ADAPTIVE_PAIR = (16., 32.)
FINER_ADAPTIVE_PAIR = (64., 128.)
TIGHT_ADAPTIVE_PAIR = (256., 384.)
VERY_TIGHT_ADAPTIVE_PAIR = (768., 1024.)
ULTRA_TIGHT_ADAPTIVE_PAIR = (2048., 3072.)
FIXED_PAIR_S = (.078125e-6, .0390625e-6)
FINER_FIXED_PAIR_S = (.0390625e-6, .01953125e-6)
FINEST_FIXED_PAIR_S = (.01953125e-6, .009765625e-6)


def record(attempt):
    metrics = [np.inf if value is None else value for value in attempt["window_metrics_m"]]
    return (attempt["code"], attempt["elapsed_s"], np.array(attempt["position_m"]),
            np.array(attempt["velocity_m_per_s"]), attempt["first_core_s"],
            np.array(attempt["window_anchor_m"]), *metrics)


def verify_reused_attempts(path, document):
    """Bind copied fixed observations to immutable original review bytes."""
    for node in document["nodes"]:
        for attempt in node["attempts"]:
            provenance = attempt.get("reused_from")
            if provenance is None:
                continue
            name = provenance["file"]
            if Path(name).name != name or name == path.name:
                raise ValueError("Reused review must name a different local report")
            content = (path.parent / name).read_bytes()
            if sha256(content).hexdigest() != provenance["sha256"]:
                raise ValueError("Reused observation source checksum differs")
            original = json.loads(content)
            for key in ("configuration", "checkpoint_stage", "ray", "prepared_sha256",
                        "original_checkpoint_sha256"):
                if original.get(key) != document.get(key):
                    raise ValueError("Reused observation inputs differ")
            candidates = [a for n in original["nodes"]
                          if n["node"] == node["node"] and n["speed_m_per_s"] == node["speed_m_per_s"]
                          for a in n["attempts"] if a.get("method") == "fixed"
                          and a.get("resolution") == attempt.get("resolution")
                          and "code" in a and "reused_from" not in a]
            copied = {k: v for k, v in attempt.items() if k != "reused_from"}
            if attempt.get("method") != "fixed" or len(candidates) != 1 or copied != candidates[0]:
                raise ValueError("Reused observation differs from its original record")


def audited_resolution(node, report, speed_grid=FINE_GRID):
    if node["speed_m_per_s"] != speed_grid[node["node"]]:
        raise ValueError("Review speed does not match the specified frozen grid")
    values = {(a["method"], a["resolution"]): record(a) for a in node["attempts"]
              if "code" in a}
    fixed_pair = tuple(node.get("fixed_pair_s", FIXED_PAIR_S))
    adaptive_pair = tuple(node.get("adaptive_pair", ADAPTIVE_PAIR))
    if fixed_pair not in (FIXED_PAIR_S, FINER_FIXED_PAIR_S, FINEST_FIXED_PAIR_S):
        return None
    if adaptive_pair not in (
        ADAPTIVE_PAIR, FINER_ADAPTIVE_PAIR, TIGHT_ADAPTIVE_PAIR,
        VERY_TIGHT_ADAPTIVE_PAIR, ULTRA_TIGHT_ADAPTIVE_PAIR,
    ):
        return None
    if adaptive_pair in (VERY_TIGHT_ADAPTIVE_PAIR, ULTRA_TIGHT_ADAPTIVE_PAIR):
        from mot_strange_review_observer import control_proof

        if node.get("adaptive_observer_proof") != control_proof():
            return None
    elif "adaptive_observer_proof" in node:
        return None
    controls = {"adaptive": adaptive_pair, "fixed": fixed_pair}
    categories = []
    for method, pair in controls.items():
        if any((method, resolution) not in values for resolution in pair):
            return None
        result = agreed_category(*(values[(method, resolution)] for resolution in pair))
        if result is None or result != node.get(f"{method}_agreement"):
            return None
        categories.append(result)
    if categories[0] != categories[1]:
        return None
    category = categories[0]
    if agreed_category(values[("adaptive", adaptive_pair[-1])],
                       values[("fixed", fixed_pair[-1])]) != category:
        return None
    pocket = None
    if category == OFF_CORE:
        adaptive, fixed = node.get("adaptive_pocket"), node.get("fixed_pocket")
        if not adaptive or not fixed:
            return None
        accepted = "stable_pocket_and_return_verified"
        if (adaptive["status"] == accepted) != (fixed["status"] == accepted):
            return None
        pocket = fixed
    return category, pocket, report


def load_resolution(path, node_indices, codes, fingerprint, speed_grid=FINE_GRID):
    missing = set(map(int, node_indices[codes < 0]))
    if not missing:
        return {}
    digest = sha256(path.read_bytes()).hexdigest()
    verified = {}
    for source in sorted((path.parent / "numerical_review").glob(f"{path.stem}_*.json")):
        content = source.read_bytes()
        document = json.loads(content)
        if (document.get("original_checkpoint_sha256") != digest
                or document["prepared_sha256"] != fingerprint or document["ray"] != path.name):
            continue
        verify_reused_attempts(source, document)
        for node in document["nodes"]:
            if node["node"] not in missing:
                continue
            accepted = audited_resolution(node, sha256(content).hexdigest(), speed_grid)
            if accepted is None:
                continue
            if node["node"] in verified and verified[node["node"]][:2] != accepted[:2]:
                raise ValueError(f"Independent numerical reviews disagree for {path.name}")
            verified[node["node"]] = accepted
    if set(verified) != missing:
        raise ValueError(f"Numerical review is incomplete for {path.name}, nodes {sorted(missing - set(verified))}")
    return verified


def effective_classification(ray, overlay):
    nodes = ray["node_indices"]
    codes, verified = ray["codes"].copy(), ray["offcore_verified"].copy()
    pockets = json.loads(ray["pocket_json"].item())
    for node, (category, pocket, _) in overlay.items():
        selected = nodes == node
        if not np.any(selected) or not np.all(codes[selected] < 0):
            raise ValueError("A numerical overlay can only resolve an original unknown")
        codes[selected] = category
        if pocket is not None:
            pockets[str(node)] = pocket
            verified[selected] = pocket["status"] == "stable_pocket_and_return_verified"
    return codes, verified, pockets

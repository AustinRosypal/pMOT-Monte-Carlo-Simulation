"""Use impact-stratified discs in an isolated revision of the loading campaign.

The original uniform pilot retains its source and directory. This adapter
configures the same audited checkpoint runner in both parent and spawn
processes, with new sample weights, seeds and a separate manifest.
"""

import hashlib
import json
from pathlib import Path
import sys

import run_pmot_2level_loading as base
from pmot.pMOT_2Level import loading_impact_sampling as sampling


ROOT = base.ROOT
ORIGINAL_OUTPUT = base.OUTPUT
OUTPUT = ORIGINAL_OUTPUT / 'impact_stratified'
_original_fingerprint = base.fingerprint
_original_freeze = base.freeze_counts


def fingerprint():
    files = (Path(__file__), Path(sampling.__file__).with_name('loading_sampling.py'))
    return {**_original_fingerprint(), **{
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }}


def freeze_counts(pilot, max_discs):
    counts = _original_freeze(pilot, max_discs)
    for i, row in enumerate(pilot['settings']):
        if all(row['statistics'][key]['discs_nonzero'] < 3 for key in ('core', 'off_core')):
            # A zero or very sparse pilot cannot estimate relative precision.
            # Reserve a fixed detection sample; never call a zero a precise rate.
            counts[i] = max(counts[i], min(64, max_discs))
    return counts


def configure():
    base.sampling = sampling
    base.OUTPUT = OUTPUT
    base.PILOT_POINTS = 11
    base.PRODUCTION_POINTS = 22
    base.fingerprint = fingerprint
    base.freeze_counts = freeze_counts


def main():
    if sys.argv[1:2] == ['audit']:
        import audit_pmot_2level_loading as audit

        audit.OUTPUT = OUTPUT
        audit.samp = sampling
        audit.fingerprint = fingerprint
        sys.argv.pop(1)
        audit.main()
    elif sys.argv[1:2] == ['report']:
        import report_pmot_2level_loading as report

        report.OUTPUT = OUTPUT
        report.report()
    else:
        if sys.argv[1:2] == ['pilot']:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            controls = OUTPUT / 'controls.json'
            if not controls.exists():
                value = json.loads((ORIGINAL_OUTPUT / 'controls.json').read_text(encoding='utf-8'))
                value['impact_edges_mm'] = [1e3*x for x in sampling.IMPACT_EDGES_M]
                value['impact_points_pilot'] = sampling.POINT_ALLOCATION[11]
                value['impact_points_production'] = sampling.POINT_ALLOCATION[22]
                base.write_json(controls, value)
        base.main()


# Windows spawn reimports the launcher; the worker must use the same reducer.
configure()

if __name__ == '__main__':
    main()

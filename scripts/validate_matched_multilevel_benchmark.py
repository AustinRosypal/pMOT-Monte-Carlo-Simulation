"""Check the physical multilevel kernel at the matched two-level inputs."""
import argparse
import json
from pmot.mot_multilevel.matched_validation import run_preflight


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only",action="store_true")
    args=parser.parse_args()
    from pmot.mot_multilevel.matched_benchmark import definitions
    result=run_preflight(definitions()[0],thorough=not args.baseline_only)
    print(json.dumps({k:result[k] for k in ("status","scope","checked_configurations","elapsed_wall_s")}))

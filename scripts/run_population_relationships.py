"""Project-local dependency bootstrap; always invoke with the prescribed Python."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.venv_pMOT_MC/Lib/site-packages'),str(ROOT/'src')]
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['MPLBACKEND']='Agg'

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('force','validate','pilot','production','all','report'))
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    try:
        if args.stage=='force':
            from pmot.mot_multilevel.relationship_campaign import run_force_sweep
            run_force_sweep()
        else:
            from pmot.mot_multilevel.campaign_execution import main
            main(args.stage,args.workers)
    except Exception as error:
        import traceback
        from pmot.mot_multilevel.relationship_campaign import paths,write_json
        write_json(paths()['statistics']/'run_failure.json',
            {'stage':args.stage,'error':repr(error),'traceback':traceback.format_exc()})
        raise

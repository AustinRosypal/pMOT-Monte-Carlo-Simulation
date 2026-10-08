"""Independent complete fixed-RK4 comparisons for the diameter pilot masks."""
import run_population_diameter_campaign as runner
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np


if __name__=='__main__':
    runner.rc.CAMPAIGN='cooling_diameter_20260930_fixed_intensity_baseline'
    root=runner.rc.paths()['statistics']
    assert runner.rc.source_digest()==runner.CORE_SHA
    manifest=json.loads((root/'campaign_manifest.json').read_text())
    assert manifest['runner_sha256']==hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
    assert (root/'pilot_collection_complete.json').exists()
    runner.rc.write_json(root/'diameter_fixed_validation_process.json',dict(pid=os.getpid(),command=sys.argv,
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=16))
    gate=root/'validation/diameter_fixed_comparison.json'
    assert not gate.exists(), 'Existing comparison must be reviewed before replacement'
    jobs=[]
    for i,diameter in enumerate(runner.DIAMETERS):
        point=runner.DiameterPoint(i,diameter,.027*(diameter/12.7)**2,-2*np.pi*15e6)
        for disc,pindex in ((0,0),(17,31)):
            filename=root/'pilot'/point.key/'rays'/f'd{disc:03d}_p{pindex:03d}.npz'
            jobs.append((point,filename,runner.prepare_diameter(point.config())))
    rows=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        future_inputs={pool.submit(runner.fixed_comparison,job):job[1] for job in jobs}
        for future in as_completed(future_inputs):
            row=future.result()
            row['input_sha256']=hashlib.sha256(future_inputs[future].read_bytes()).hexdigest()
            rows.append(row)
            runner.rc.write_json(root/'validation/diameter_fixed_comparison_progress.json',dict(completed_rays=len(rows),total_rays=len(jobs),rows=rows))
            print(f"fixed comparison {len(rows)}/{len(jobs)}; differences={len(row['differences'])}",flush=True)
    report={'passed':all(not r['differences'] for r in rows),'rows':rows,
        'source_sha256':runner.CORE_SHA,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'checks original early-capture masks only; longer retention qualification remains independently required'}
    runner.rc.write_json(gate,report)
    print('passed:',report['passed'],flush=True)

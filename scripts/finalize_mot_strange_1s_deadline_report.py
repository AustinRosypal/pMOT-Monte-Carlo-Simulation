"""Assemble separate-horizon deadline results and a scientific production plot."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]/'outputs/statistics/mot_strange/calibrated_10_10_20Gpcm_full_sphere_r15mm_incell_20261025_retained1s_v5_deadline_20261028'
    report=root/'FINAL_REPORT.md'
    one=json.loads((root/'reports/final_1s.json').read_text())
    two=json.loads((root/'reports/preserved_2s.json').read_text())
    process=json.loads((root/'process.json').read_text());now=datetime.now(timezone.utc)
    info=dict(finalized_at_utc=now.isoformat(),one_second_runtime_hours=(now-datetime.fromisoformat(process['started_at_utc'])).total_seconds()/3600,
              one_second_result='reports/final_1s.json',two_second_result='reports/preserved_2s.json',
              mixed_horizon_estimate=False,validation_complete=False,visual_review_pending=True)
    (root/'reports/final_index.json').write_text(json.dumps(info,indent=2)+'\n')
    text=report.read_text()
    marker='\n## Preserved 2-second cohort\n'
    text=text.split(marker)[0]
    text+=marker+'\nThis separately reduced cohort was stopped at the user horizon change. It is not combined with the 1-second continuation.\n\n```json\n'+json.dumps(two,indent=2)+'\n```\n'
    text+='\n## Timing and scope\n\n```json\n'+json.dumps(info,indent=2)+'\n```\n'
    if (root/'reports/final_1s.npz').exists():
        with np.load(root/'reports/final_1s.npz') as a:
            v=a['velocity_m_per_s'];spectra=a['disc_all_outcome_spectra_m2'];rates=a['disc_all_outcome_rates_atoms_per_s']
        fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
        from scipy.stats import t
        n=len(spectra)
        for k,label in [(0,'Central final-window retention'),(1,'Off-core endpoint retention')]:
            mean=spectra[:,k].mean(axis=0)*1e6
            half=t.ppf(.975,n-1)*spectra[:,k].std(axis=0,ddof=1)/np.sqrt(n)*1e6 if n>1 else np.zeros_like(mean)
            axes[k].plot(v,mean);axes[k].fill_between(v,np.maximum(0,mean-half),mean+half,alpha=.25)
            axes[k].set(xlabel='Incident speed (m/s)',ylabel='Cross section (mm²)',title=label,xlim=(0,80))
            axes[k].grid(alpha=.2)
        labels=['1 s central','2 s central','1 s off-core','2 s off-core']
        vals=[one['loading_rate_atoms_per_s'],two['loading_rate_atoms_per_s'],one['off_core_retention_integral_atoms_per_s'],two['off_core_retention_integral_atoms_per_s']]
        errors=[one['loading_95_half_width_atoms_per_s'],two['loading_95_half_width_atoms_per_s'],one['off_core_retention_95_half_width_atoms_per_s'],two['off_core_retention_95_half_width_atoms_per_s']]
        axes[2].errorbar(np.asarray(vals)/1e6,np.arange(4),xerr=np.asarray(errors)/1e6,fmt='o',capsize=4)
        axes[2].set(yticks=np.arange(4),yticklabels=labels,xlabel='Rate (million atoms/s)',title='Separate cohorts; nominal 95% intervals')
        axes[2].grid(axis='x',alpha=.2)
        fig.suptitle(f'Deadline-limited surrogate MOT: {n} complete 1 s discs; 2 s cohort separate\nOff-core is not validated 3D trapping; final validation pending',fontsize=11)
        fig.savefig(root/'reports/final_production.png',dpi=160);plt.close(fig)
        text+='\nActual deadline production figure: reports/final_production.png (visual review pending).\n'
    report.write_text(text,encoding='utf-8')
    print(json.dumps(info))


if __name__=='__main__':main()

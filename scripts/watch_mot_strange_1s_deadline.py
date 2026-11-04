"""Independent administrative deadline guard for the separately named 1s runner."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/statistics/mot_strange/calibrated_10_10_20Gpcm_full_sphere_r15mm_incell_20261025_retained1s_v5_deadline_20261028'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def process_family(pid):
    command=f"$p=Get-CimInstance Win32_Process; $p | Where-Object {{$_.ProcessId -eq {pid} -or $_.ParentProcessId -eq {pid}}} | Select-Object ProcessId,ParentProcessId,CommandLine,CreationDate | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    if not result.stdout.strip():return []
    value=json.loads(result.stdout);return value if isinstance(value,list) else [value]


def finalize():
    result=subprocess.run(['uv','run','python','scripts/run_mot_strange_retained1s_deadline.py','finalize'],cwd=ROOT,capture_output=True,text=True)
    write(OUTPUT/'deadline_finalization.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode!=0:raise RuntimeError('Deadline numerical summary failed; see deadline_finalization.json')
    extra=subprocess.run(['uv','run','python','scripts/finalize_mot_strange_1s_deadline_report.py'],cwd=ROOT,capture_output=True,text=True)
    write(OUTPUT/'deadline_report_assembly.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),returncode=extra.returncode,stdout=extra.stdout,stderr=extra.stderr))
    if extra.returncode!=0:raise RuntimeError('Deadline report assembly failed')


def main(pid):
    deadline=datetime.fromisoformat(read(OUTPUT/'user_control.json')['hard_deadline_utc']).timestamp()
    expected='run_mot_strange_retained1s_deadline.py'
    initial=process_family(pid)
    parent=next((p for p in initial if p['ProcessId']==pid),None)
    assert parent and expected in (parent['CommandLine'] or ''), 'Exact1s collector identity required'
    write(OUTPUT/'watchdog.json',dict(watched_pid=pid,at_utc=datetime.now(timezone.utc).isoformat(),hard_deadline_utc=read(OUTPUT/'user_control.json')['hard_deadline_utc'],identity=parent))
    soft_written=False
    while True:
        now=time.time()
        if now>=deadline-300 and not soft_written:
            write(OUTPUT/'STOP_REQUESTED.json',dict(reason='User seven-hour administrative cutoff; checkpoint and drain',at_utc=datetime.now(timezone.utc).isoformat()))
            soft_written=True
        family=process_family(pid)
        collector=next((p for p in family if p['ProcessId']==pid),None)
        if collector is None:
            # Children must also have exited before final data reduction.
            if family and now<deadline-90:
                time.sleep(2);continue
            if family:
                assert all('spawn_main' in (p['CommandLine'] or '') for p in family), 'Unknown descendant; refuse unrelated termination'
        else:
            assert collector['CreationDate']==parent['CreationDate'] and expected in (collector['CommandLine'] or ''), 'PID identity changed'
            if now<deadline-90:
                time.sleep(min(30,max(1,deadline-300-now)) if not soft_written else 2);continue
        if family:
            assert all(p['ProcessId']==pid or ('spawn_main' in (p['CommandLine'] or '') and p['ParentProcessId']==pid) for p in family)
            ids=[pid]+[p['ProcessId'] for p in family if p['ProcessId']!=pid]
            command='Stop-Process -Id '+','.join(map(str,ids))+' -ErrorAction SilentlyContinue'
            subprocess.run(['powershell','-NoProfile','-Command',command],check=True)
            write(OUTPUT/'deadline_interrupted_inflight.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),verified_processes=family,meaning='Cooperative drain grace exhausted. Only saved atomic checkpoints count; in-flight evaluations incomplete.'))
            for _ in range(30):
                if not process_family(pid):break
                time.sleep(1)
            assert not process_family(pid),'Verified campaign processes have not exited'
        finalize()
        write(OUTPUT/'deadline_guard_complete.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),all_campaign_processes_exited=True,hard_deadline_utc=read(OUTPUT/'user_control.json')['hard_deadline_utc'],report=str(OUTPUT/'FINAL_REPORT.md')))
        break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('pid',type=int);args=parser.parse_args()
    try:main(args.pid)
    except Exception as error:
        write(OUTPUT/'deadline_guard_failure.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),error=repr(error)))
        raise

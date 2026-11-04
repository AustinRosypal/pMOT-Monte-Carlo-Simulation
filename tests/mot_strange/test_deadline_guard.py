import importlib
from pathlib import Path
import json
import pytest


def setup_guard(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    guard=importlib.import_module('watch_mot_strange_1s_deadline')
    monkeypatch.setattr(guard,'OUTPUT',tmp_path)
    (tmp_path/'user_control.json').write_text(json.dumps({'hard_deadline_utc':'2026-10-28T13:43:52+00:00'}))
    return guard


def test_guard_refuses_unrelated_process(monkeypatch,tmp_path):
    guard=setup_guard(monkeypatch,tmp_path)
    monkeypatch.setattr(guard,'process_family',lambda pid:[dict(ProcessId=123,CommandLine='jupyter lab')])
    with pytest.raises(AssertionError,match='identity'):guard.main(123)
    assert not (tmp_path/'STOP_REQUESTED.json').exists()


def test_guard_stops_only_verified_family_and_finalizes(monkeypatch,tmp_path):
    guard=setup_guard(monkeypatch,tmp_path)
    parent=dict(ProcessId=123,ParentProcessId=1,CommandLine='uv run python run_mot_strange_retained1s_deadline.py run',CreationDate='same')
    child=dict(ProcessId=456,ParentProcessId=123,CommandLine='spawn_main(parent_pid=123)',CreationDate='child')
    sequence=iter([[parent,child],[parent,child],[],[]])
    monkeypatch.setattr(guard,'process_family',lambda pid:next(sequence))
    deadline=guard.datetime.fromisoformat('2026-10-28T13:43:52+00:00').timestamp()
    monkeypatch.setattr(guard.time,'time',lambda:deadline-60)
    calls=[]
    monkeypatch.setattr(guard.subprocess,'run',lambda args,**kwargs:calls.append(args))
    monkeypatch.setattr(guard,'finalize',lambda:calls.append('finalize'))
    guard.main(123)
    assert calls[0][-1]=='Stop-Process -Id 123,456 -ErrorAction SilentlyContinue'
    assert calls[-1]=='finalize'
    assert json.loads((tmp_path/'deadline_guard_complete.json').read_text())['all_campaign_processes_exited']

"""Audit each completed production setting while later settings run."""

import json
import msvcrt
import os
from pathlib import Path
import subprocess
from time import sleep

from run_mot_strange_loading_pilot import NAME, ROOT, now, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_audit(cache, args, label):
    log = cache / f"audit_{label}.log"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\nAudit started at {now()}\n")
        stream.flush()
        result = subprocess.run(
            ["uv", "run", "python", "-u", "scripts/audit_mot_strange_sampled_production.py", *args],
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    return result.returncode, str(log)


def monitor(cache):
    plan = read(ROOT / "sampled_production_plan.json")
    expected = plan["expected_settings"]
    state = cache / "audit_watch_progress.json"
    previous = read(state) if state.exists() else {}
    failed = previous.get("needs_review", {})
    while True:
        complete = set()
        for index in range(expected):
            folder = f"configuration_{index:03d}"
            published = ROOT / "production" / folder / "audit.json"
            if published.is_file() and read(published)["status"] == "passed":
                complete.add(index)
                failed.pop(str(index), None)
                continue
            checkpointed = cache / folder / "summary.json"
            if not checkpointed.is_file() or str(index) in failed:
                continue
            result = read(checkpointed)
            if result["status"] != "checkpointed_pending_audit":
                failed[str(index)] = {"reason": result["status"]}
                continue
            code, log = run_audit(cache, [str(index)], f"{index:03d}")
            if code:
                failed[str(index)] = {"returncode": code, "log": log}
            else:
                complete.add(index)
            write_json(state, {"status": "watching", "audited_settings": sorted(complete),
                               "needs_review": failed, "updated_utc": now()})
        production = cache / "progress.json"
        if production.is_file() and read(production)["status"] == "checkpointed_pending_audit":
            if len(complete) != expected or failed:
                write_json(state, {"status": "needs_review", "audited_settings": sorted(complete),
                                   "needs_review": failed, "updated_utc": now()})
                return
            code, log = run_audit(cache, ["--finalize"], "final")
            if code:
                failed["final"] = {"returncode": code, "log": log}
            else:
                with (cache / "report.log").open("a", encoding="utf-8") as report_log:
                    report = subprocess.run(
                        ["uv", "run", "python", "-u", "scripts/report_mot_strange_loading_scans.py"],
                        stdout=report_log, stderr=subprocess.STDOUT, check=False,
                    )
                if report.returncode:
                    failed["report"] = {"returncode": report.returncode, "log": str(cache / "report.log")}
            write_json(state, {"status": "needs_review" if failed else "reports_ready_for_visual_qa",
                               "audited_settings": sorted(complete), "needs_review": failed,
                               "updated_utc": now()})
            return
        write_json(state, {"status": "watching", "audited_settings": sorted(complete),
                           "needs_review": failed, "updated_utc": now()})
        sleep(30)


if __name__ == "__main__":
    cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME / "sampled_production"
    with (cache / "audit_watch.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        monitor(cache)

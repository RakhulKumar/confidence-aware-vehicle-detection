#!/usr/bin/env python3
"""Watchdog for the v0.6 queue runner (run every 15 min by the scheduled task confdet_v06_watchdog).

  python scripts/queue_watchdog.py [--queue <paths_local.yaml: queue>] [--task confdet_v06_queue] [--dry-run]

Starts the runner -- always through its own scheduled task, never directly -- only if the
heartbeat is older than 5 minutes, no live process holds the runner lock or wrote the heartbeat,
the runner's task is not running, no job has failed or is held by the disk check, and there is a
job to run (utils/jobqueue.watchdog_decision). Three layers stop a second runner: this check,
the task's "do not start a new instance" setting, and the runner's own lock.
One line per run goes to <logs>/watchdog.log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.utils.io_utils import site_paths  # noqa: E402
from confdet.utils.jobqueue import load_queue, now, watchdog_decision  # noqa: E402


def task_running(name: str) -> bool:
    r = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "CSV", "/NH"], capture_output=True, text=True)
    return r.returncode == 0 and '"Running"' in r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default=site_paths()["queue"])
    ap.add_argument("--task", default="confdet_v06_queue")
    ap.add_argument("--max-age-s", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="decide and log, never start")
    a = ap.parse_args()
    qpath = Path(a.queue)
    q = load_queue(qpath)
    hb_path, lock_path = qpath.with_name("heartbeat.json"), qpath.with_suffix(".lock")
    hb = json.loads(hb_path.read_text(encoding="utf-8")) if hb_path.exists() else None
    try:
        lock_pid = int(lock_path.read_text().strip() or 0) if lock_path.exists() else None
    except ValueError:
        lock_pid = None
    start, why = watchdog_decision(q, hb, time.time(), lock_pid, task_running(a.task), a.max_age_s)
    action = "no action"
    if start and not a.dry_run:
        r = subprocess.run(["schtasks", "/Run", "/TN", a.task], capture_output=True, text=True)
        action = f"STARTED {a.task}" if r.returncode == 0 else f"START FAILED ({r.returncode}): {r.stdout or r.stderr}".strip()
    elif start:
        action = "would start (dry run)"
    line = f"[{now()}] {action}: {why}"
    logs = Path(q.get("logs", qpath.parent.parent / "logs"))
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "watchdog.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()

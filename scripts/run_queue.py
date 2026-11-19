#!/usr/bin/env python3
"""Run the v0.6 job queue, one job at a time, restart-safe (see src/confdet/utils/jobqueue.py).

  python scripts/run_queue.py [--queue <paths_local.yaml: queue>] [--once]
  python scripts/run_queue.py --status

The runner takes the job left `running` (after a crash or restart it is
resumed) or else the next `pending` job whose prerequisites are done, marks
it `running`, runs it with its output appended to <logs>/<job id>.log,
verifies its outputs and marks it `done` -- or `failed`, which stops the
queue until a human looks. `held` jobs are never started.

If a job was started outside the runner (`pid:` in the queue file) and that
process is still alive, the runner waits for it instead of starting a second copy.

After the PC restarts, start the runner again (the same command); it continues
where it stopped.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import threading  # noqa: E402

from confdet.utils.jobqueue import (  # noqa: E402
    Lock, disk_check, heartbeat_status, load_queue, next_job, now, pid_alive, remove_last_checkpoint, save_queue,
    set_status, status_table, verify, write_heartbeat,
)

# drives the disk check watches: C: and D: on the Windows PC, the data root elsewhere (Beocat)
DISK_DRIVES = ("C:", "D:") if os.name == "nt" else (os.environ.get("CONFDET_ROOT") or str(Path.home()),)
_CURRENT = {"job": None, "pid": None}


def _heartbeat_loop(path: Path, stop: threading.Event) -> None:
    """Write <queue folder>/heartbeat.json every minute while the runner lives."""
    while not stop.is_set():
        try:
            write_heartbeat(path, _CURRENT["job"], _CURRENT["pid"])
        except OSError:
            pass
        stop.wait(60)

from confdet.utils.io_utils import site_paths  # noqa: E402

DEFAULT_QUEUE = site_paths()["queue"]          # configs/paths_local.yaml, site CONFDET_SITE


def run_one(job: dict, q: dict, qpath: Path, logs: Path, python: str) -> bool:
    if job["status"] == "running" and pid_alive(job.get("pid")):
        print(f"[{now()}] {job['id']} is already running as PID {job['pid']}; waiting for it")
        while pid_alive(job.get("pid")):
            time.sleep(30)
    else:
        resumed = job["status"] == "running"
        ok_disk, why = disk_check(job, drives=DISK_DRIVES)
        if not ok_disk:
            set_status(q, job["id"], "held", note=f"held by the disk check {now()}: {why}")
            save_queue(q, qpath)
            print(f"[{now()}] {job['id']} HELD: {why}; queue stopped")
            return False
        _CURRENT["job"] = job["id"]
        set_status(q, job["id"], "running", started_at=job.get("started_at") or now(),
                   attempts=int(job.get("attempts", 0)) + 1, pid=None)
        save_queue(q, qpath)
        log = logs / f"{job['id']}.log"
        print(f"[{now()}] {'resuming' if resumed else 'starting'} {job['id']} -> {log}")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {now()} {'RESUME' if resumed else 'START'} {job['id']}: {' '.join(job['cmd'])}\n")
            f.flush()
            proc = subprocess.Popen([python, *job["cmd"]], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
            set_status(q, job["id"], "running", pid=proc.pid)
            save_queue(q, qpath)
            _CURRENT["pid"] = proc.pid
            code = proc.wait()
            f.write(f"===== {now()} EXIT CODE {code}\n")
        if code != 0:
            q = load_queue(qpath)
            set_status(q, job["id"], "failed", finished_at=now(), error=f"exit code {code}", pid=None)
            save_queue(q, qpath)
            print(f"[{now()}] {job['id']} FAILED (exit code {code}); queue stopped. See {log}")
            return False
    q = load_queue(qpath)
    ok, msg = verify(next(j for j in q["jobs"] if j["id"] == job["id"]).get("verify"), REPO)
    removed = remove_last_checkpoint(next(j for j in q["jobs"] if j["id"] == job["id"])) if ok else None
    set_status(q, job["id"], "done" if ok else "failed", finished_at=now(), pid=None,
               **({} if ok else {"error": f"verification: {msg}"}),
               **({"removed_last_pt": removed} if removed else {}))
    save_queue(q, qpath)
    _CURRENT.update(job=None, pid=None)
    print(f"[{now()}] {job['id']} {'done' if ok else 'FAILED verification: ' + msg}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--once", action="store_true", help="run one job, then stop")
    args = ap.parse_args()
    qpath = Path(args.queue)
    q = load_queue(qpath)
    if args.status:
        lines, t = status_table(q)
        print("\n".join(lines))
        print(heartbeat_status(qpath.with_name("heartbeat.json")))
        print(f"\ndone {t['done']} | running {t['running']} | pending {t['pending']} | failed {t['failed']} | "
              f"held {t['held']} -> about {t['hours_left']:.1f} h left in released jobs "
              f"(+{t['hours_held']:.1f} h held)")
        return
    logs = Path(q.get("logs", qpath.parent.parent / "logs"))
    logs.mkdir(parents=True, exist_ok=True)
    python = q.get("python") or sys.executable
    lock = Lock(qpath.with_suffix(".lock"))
    if not lock.acquire():
        raise SystemExit(f"another runner holds {lock.path}")
    stop = threading.Event()
    write_heartbeat(qpath.with_name("heartbeat.json"), None)      # at once, then every minute from the thread
    threading.Thread(target=_heartbeat_loop, args=(qpath.with_name("heartbeat.json"), stop), daemon=True).start()
    try:
        while True:
            q = load_queue(qpath)
            job = next_job(q)
            if job is None:
                _, t = status_table(q)
                print(f"[{now()}] nothing to run: done {t['done']}, held {t['held']}, failed {t['failed']}, "
                      f"pending (waiting on prerequisites) {t['pending']}")
                return
            if not run_one(job, q, qpath, logs, python) or args.once:
                return
    finally:
        stop.set()
        lock.release()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()

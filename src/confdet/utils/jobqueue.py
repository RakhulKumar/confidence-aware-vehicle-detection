"""Restart-safe job queue (v0.6).   Used by scripts/run_queue.py.

The queue is one YAML file. Each job:

    id: train_yolo26_v06              unique name
    cmd: [scripts/04_train_detector.py, --config, ..., --resume]
                                      run as `python <cmd...>` from the project folder
    after: [build_bmd_v06]            jobs that must be done first
    est_hours: 14                     for --status
    status: pending | running | done | failed | held
    verify: {...}                     what "done" means (see `verify`)

A job is set to `running` (and the file saved) before it starts and to `done`
only after its outputs pass `verify`. After a crash or restart, the runner
picks the `running` job up again: the commands themselves are restart-safe
(training resumes from last.pt, prediction skips finished files, results are
written atomically), so re-running a job continues it. `held` jobs are never
started: they wait for a human to change them to `pending`.

Every write of the queue file is atomic (temporary file + rename).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

STATUSES = ("pending", "running", "done", "failed", "held")


def load_queue(path: str | Path) -> dict:
    q = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    q.setdefault("jobs", [])
    ids = [j["id"] for j in q["jobs"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate job ids in {path}")
    for j in q["jobs"]:
        j.setdefault("status", "pending")
        j.setdefault("after", [])
        if j["status"] not in STATUSES:
            raise ValueError(f"job {j['id']}: unknown status {j['status']!r}")
        unknown = set(j["after"]) - set(ids)
        if unknown:
            raise ValueError(f"job {j['id']} waits for unknown job(s) {sorted(unknown)}")
    return q


def save_queue(q: dict, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(q, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, path)


def next_job(q: dict) -> dict | None:
    """A job left `running` by a crash comes first (it is resumed); then the first
    `pending` job whose prerequisites are all done. `failed` stops the queue."""
    by_id = {j["id"]: j for j in q["jobs"]}
    if any(j["status"] == "failed" for j in q["jobs"]):
        return None
    for j in q["jobs"]:
        if j["status"] == "running":
            return j
    for j in q["jobs"]:
        if j["status"] == "pending" and all(by_id[a]["status"] == "done" for a in j["after"]):
            return j
    return None


def set_status(q: dict, job_id: str, status: str, **fields) -> None:
    if status not in STATUSES:
        raise ValueError(status)
    for j in q["jobs"]:
        if j["id"] == job_id:
            j["status"] = status
            j.update(fields)
            return
    raise KeyError(job_id)


def verify(spec: dict | None, root: Path) -> tuple[bool, str]:
    """Check a job's outputs. spec keys (all optional, all must pass):
      files:      [path, ...]                       each exists and is non-empty
      min_files:  {dir: path, glob: '*.png', count: n}   at least n matches
      checkpoint: path                              loads with torch (CPU)
      npz:        [path, ...]                       each loads as prediction records
      json:       [path, ...]                       each parses as JSON
    Relative paths are taken from `root` (the project folder)."""
    import json
    spec = spec or {}

    def P(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    for f in spec.get("files", []):
        if not P(f).exists() or P(f).stat().st_size == 0:
            return False, f"missing or empty: {f}"
    mf = spec.get("min_files")
    if mf:
        n = len(list(P(mf["dir"]).glob(mf.get("glob", "*")))) if P(mf["dir"]).exists() else 0
        if n < int(mf["count"]):
            return False, f"{mf['dir']}: {n} files matching {mf.get('glob', '*')}, need {mf['count']}"
    if spec.get("checkpoint"):
        try:
            import torch
            torch.load(str(P(spec["checkpoint"])), map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            return False, f"checkpoint {spec['checkpoint']} does not load: {e}"
    for f in spec.get("npz", []):
        try:
            from confdet.evaluation.records import load_records
            if len(load_records(P(f)).records) == 0:
                return False, f"{f} holds no images"
        except Exception as e:  # noqa: BLE001
            return False, f"{f} does not load: {e}"
    for f in spec.get("json", []):
        try:
            json.loads(P(f).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return False, f"{f} is not valid JSON: {e}"
    return True, "ok"


def status_table(q: dict) -> tuple[list[str], dict]:
    """Lines for --status and totals (hours left excludes held jobs, listed separately)."""
    lines, tot = [], {s: 0 for s in STATUSES}
    hours_left = hours_held = 0.0
    for j in q["jobs"]:
        tot[j["status"]] += 1
        h = float(j.get("est_hours", 0) or 0)
        if j["status"] in ("pending", "running", "failed"):
            hours_left += h
        elif j["status"] == "held":
            hours_held += h
        extra = f"  [{j.get('note')}]" if j.get("note") else ""
        lines.append(f"{j['status']:<8} {h:>6.1f} h  {j['id']}{extra}")
    return lines, {**tot, "hours_left": hours_left, "hours_held": hours_held}


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).status() != "zombie"
    except Exception:  # noqa: BLE001
        return False


class Lock:
    """One runner at a time: a lock file holding the runner's PID; a stale lock
    (its PID is gone, e.g. after a restart) is taken over."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def acquire(self) -> bool:
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip() or 0)
            except ValueError:
                pid = 0
            if pid and pid != os.getpid() and pid_alive(pid):
                return False
        self.path.write_text(str(os.getpid()))
        return True

    def release(self) -> None:
        try:
            if self.path.exists() and self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink()
        except OSError:
            pass


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# v0.6 additions: free-disk check, heartbeat, last.pt clean-up
# ---------------------------------------------------------------------------
def free_gb(drive: str) -> float:
    import shutil
    root = drive if drive.endswith(("\\", "/")) else drive + "\\"
    return shutil.disk_usage(root).free / 2 ** 30


def disk_check(job: dict, drives=("C:", "D:"), min_free_gb: float = 50.0, free_fn=free_gb) -> tuple[bool, str]:
    """A job may start only if, after writing its estimated output (`est_output_gb` on
    `output_drive`, default D:), every drive keeps at least `min_free_gb` free."""
    est = float(job.get("est_output_gb", 0) or 0)
    target = str(job.get("output_drive", "D:")).upper()
    for d in drives:
        left = free_fn(d) - (est if d.upper() == target else 0.0)
        if left < min_free_gb:
            return False, (f"{d} would have {left:.0f} GB free after this job (estimated output {est:.1f} GB), "
                           f"below the {min_free_gb:.0f} GB minimum")
    return True, "ok"


def write_heartbeat(path: str | Path, job_id: str | None, job_pid: int | None = None) -> None:
    import json
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"time": now(), "epoch_s": time.time(), "runner_pid": os.getpid(),
                               "job": job_id, "job_pid": job_pid}), encoding="utf-8")
    os.replace(tmp, path)


def heartbeat_status(path: str | Path, max_age_s: float = 180.0) -> str:
    import json
    path = Path(path)
    if not path.exists():
        return "runner: no heartbeat file (not started since heartbeats were added)"
    hb = json.loads(path.read_text(encoding="utf-8"))
    age = time.time() - float(hb["epoch_s"])
    alive = age <= max_age_s and pid_alive(hb.get("runner_pid"))
    return (f"runner: {'ALIVE' if alive else 'NOT RUNNING'} (last heartbeat {hb['time']}, {age / 60:.1f} min ago, "
            f"PID {hb.get('runner_pid')}, job {hb.get('job') or '-'})")


def remove_last_checkpoint(job: dict) -> str | None:
    """After a training job is verified done: delete its rolling last.pt -- that file only,
    only inside the job's own run folder (`run_dir`), only if the job asks for it
    (`cleanup_last_pt: true`) and the canonical best.pt exists. Returns the path removed."""
    run_dir = job.get("run_dir")
    if not run_dir or not job.get("cleanup_last_pt"):
        return None
    run_dir = Path(run_dir).resolve()
    last = (run_dir / "train" / "weights" / "last.pt").resolve()
    if run_dir not in last.parents or last.name != "last.pt" or not last.exists():
        return None
    if not (run_dir / "weights" / "best.pt").exists():
        return None                                   # never remove the only checkpoint
    last.unlink()
    return str(last)


# ---------------------------------------------------------------------------
# Watchdog (v0.6, 19 Nov): restart a runner that died, never start a second one
# ---------------------------------------------------------------------------
def watchdog_decision(q: dict, heartbeat: dict | None, now_s: float, lock_pid: int | None,
                      task_running: bool, max_age_s: float = 300.0, alive=pid_alive) -> tuple[bool, str]:
    """Should the watchdog start the runner? Only if every check says the runner is gone AND it
    would have work it is allowed to do. Any doubt -> do not start."""
    if lock_pid and alive(lock_pid):
        return False, f"runner alive (lock PID {lock_pid})"
    if heartbeat and alive(heartbeat.get("runner_pid")):
        return False, f"runner alive (heartbeat PID {heartbeat.get('runner_pid')})"
    if task_running:
        return False, "the runner's scheduled task is already running"
    age = (now_s - float(heartbeat["epoch_s"])) if heartbeat else float("inf")
    if age <= max_age_s:
        return False, f"heartbeat {age:.0f} s old (<= {max_age_s:.0f} s)"
    if any(j["status"] == "failed" for j in q["jobs"]):
        return False, "a job failed: the queue waits for a human"
    if any(j["status"] == "held" and "disk check" in str(j.get("note", "")) for j in q["jobs"]):
        return False, "a job is held by the disk check: the queue waits for a human"
    job = next_job(q)
    if job is None:
        return False, "nothing to run"
    return True, f"runner gone (heartbeat {age / 60:.1f} min old); next job {job['id']}"

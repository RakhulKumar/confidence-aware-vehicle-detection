"""Restart-safe job queue (scripts/run_queue.py)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from confdet.utils.jobqueue import Lock, load_queue, next_job, save_queue, status_table, verify  # noqa: E402


def _q(jobs):
    return {"jobs": [dict(j) for j in jobs]}


def test_next_job_resumes_running_then_respects_prerequisites_held_and_failed():
    q = _q([{"id": "a", "status": "done"}, {"id": "b", "after": ["a"]}, {"id": "c", "after": ["b"]},
            {"id": "h", "status": "held"}])
    for j in q["jobs"]:
        j.setdefault("status", "pending"); j.setdefault("after", [])
    assert next_job(q)["id"] == "b"
    q["jobs"][2]["status"] = "running"                 # c was running when the PC restarted
    assert next_job(q)["id"] == "c"
    q["jobs"][2]["status"] = "pending"
    q["jobs"][1]["status"] = "failed"
    assert next_job(q) is None                         # a failure stops the queue
    q["jobs"][1]["status"] = "done"; q["jobs"][2]["status"] = "done"
    assert next_job(q) is None                         # only a held job is left
    lines, t = status_table(q)
    assert t["held"] == 1 and t["done"] == 3 and len(lines) == 4


def test_load_queue_rejects_bad_files(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text(yaml.safe_dump({"jobs": [{"id": "a", "cmd": []}, {"id": "a", "cmd": []}]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_queue(p)
    p.write_text(yaml.safe_dump({"jobs": [{"id": "a", "after": ["zz"]}]}))
    with pytest.raises(ValueError, match="unknown job"):
        load_queue(p)
    p.write_text(yaml.safe_dump({"jobs": [{"id": "a", "status": "maybe"}]}))
    with pytest.raises(ValueError, match="status"):
        load_queue(p)


def test_verify_specs(tmp_path):
    import torch
    (tmp_path / "d").mkdir()
    for i in range(3):
        (tmp_path / "d" / f"{i}.png").write_bytes(b"x")
    (tmp_path / "r.json").write_text('{"a": 1}')
    torch.save({"epoch": 3}, tmp_path / "w.pt")
    ok, _ = verify({"files": ["r.json"], "min_files": {"dir": "d", "glob": "*.png", "count": 3},
                    "checkpoint": "w.pt", "json": ["r.json"]}, tmp_path)
    assert ok
    assert not verify({"min_files": {"dir": "d", "glob": "*.png", "count": 4}}, tmp_path)[0]
    assert not verify({"files": ["nope.json"]}, tmp_path)[0]
    (tmp_path / "bad.pt").write_bytes(b"not a checkpoint")
    assert not verify({"checkpoint": "bad.pt"}, tmp_path)[0]
    (tmp_path / "bad.json").write_text("{")
    assert not verify({"json": ["bad.json"]}, tmp_path)[0]


def test_lock_blocks_a_second_runner_and_takes_over_a_stale_one(tmp_path):
    lk = Lock(tmp_path / "q.lock")
    assert lk.acquire()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "q.lock").write_text(str(other.pid))     # a live runner holds it
        assert not Lock(tmp_path / "q.lock").acquire()
    finally:
        other.kill(); other.wait()
    assert Lock(tmp_path / "q.lock").acquire()               # its PID is gone: stale, taken over


def _runner(qpath, *extra):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "run_queue.py"), "--queue", str(qpath), *extra],
                          capture_output=True, text=True, cwd=REPO)


def test_runner_end_to_end_with_crash_and_resume(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    flag = tmp_path / "crash_once"
    flag.write_text("1")
    # job b crashes on its first attempt (simulated restart), and resumes on the second
    b_code = (f"import os,sys; p=r'{flag}'\n"
              f"if os.path.exists(p): os.remove(p); sys.exit(3)\n"
              f"open(r'{out / 'b.txt'}','w').write('ok')")
    q = {"logs": str(tmp_path / "logs"), "jobs": [
        {"id": "a", "cmd": ["-c", f"open(r'{out / 'a.txt'}','w').write('ok')"], "est_hours": 1,
         "verify": {"files": [str(out / "a.txt")]}},
        {"id": "b", "after": ["a"], "cmd": ["-c", b_code], "est_hours": 2, "verify": {"files": [str(out / "b.txt")]}},
        {"id": "v", "after": ["b"], "cmd": ["-c", "pass"], "verify": {"files": [str(out / "never.txt")]}},
        {"id": "h", "status": "held", "cmd": ["-c", "raise SystemExit(9)"], "est_hours": 5},
    ]}
    qpath = tmp_path / "q.yaml"
    save_queue(q, qpath)
    r = _runner(qpath)
    s = {j["id"]: j["status"] for j in load_queue(qpath)["jobs"]}
    assert s == {"a": "done", "b": "failed", "v": "pending", "h": "held"}, r.stdout + r.stderr
    st = _runner(qpath, "--status").stdout
    assert "failed" in st and "h left" in st
    # a human looks, resets b to running (= what a restart mid-job leaves behind) and restarts the runner
    qd = load_queue(qpath)
    next(j for j in qd["jobs"] if j["id"] == "b")["status"] = "running"
    save_queue(qd, qpath)
    _runner(qpath)
    s = {j["id"]: j for j in load_queue(qpath)["jobs"]}
    assert s["b"]["status"] == "done" and s["b"]["attempts"] == 2
    assert s["v"]["status"] == "failed" and "verification" in s["v"]["error"]   # exit 0 is not enough
    assert s["h"]["status"] == "held"
    assert "RESUME b" in (tmp_path / "logs" / "b.log").read_text()
    assert not (tmp_path / "q.lock").exists() and not list(tmp_path.glob("*.tmp"))


def test_disk_check_holds_a_job_that_would_leave_less_than_50_gb():
    from confdet.utils.jobqueue import disk_check
    free = {"C:": 120.0, "D:": 80.0}
    fn = lambda d: free[d.upper()]  # noqa: E731
    assert disk_check({"est_output_gb": 10}, free_fn=fn)[0]
    ok, why = disk_check({"est_output_gb": 40}, free_fn=fn)                 # D: would be left with 40 GB
    assert not ok and "D:" in why and "50" in why
    assert disk_check({"est_output_gb": 40, "output_drive": "C:"}, free_fn=fn)[0]
    free["C:"] = 45.0
    assert not disk_check({}, free_fn=fn)[0]                                  # C: already below the minimum


def test_heartbeat_reports_alive_and_stale(tmp_path):
    import json
    import time
    from confdet.utils.jobqueue import heartbeat_status, write_heartbeat
    hb = tmp_path / "heartbeat.json"
    assert "no heartbeat" in heartbeat_status(hb)
    write_heartbeat(hb, "train_x", 123)
    assert "ALIVE" in heartbeat_status(hb) and "train_x" in heartbeat_status(hb)
    d = json.loads(hb.read_text()); d["epoch_s"] = time.time() - 600
    hb.write_text(json.dumps(d))
    assert "NOT RUNNING" in heartbeat_status(hb)


def test_last_pt_is_removed_only_after_done_and_only_inside_its_run(tmp_path):
    from confdet.utils.jobqueue import remove_last_checkpoint
    run = tmp_path / "runs" / "exp" / "det"
    (run / "train" / "weights").mkdir(parents=True)
    (run / "weights").mkdir()
    last = run / "train" / "weights" / "last.pt"
    last.write_bytes(b"L")
    other = tmp_path / "runs" / "exp" / "other" / "train" / "weights"
    other.mkdir(parents=True)
    (other / "last.pt").write_bytes(b"O")
    job = {"run_dir": str(run), "cleanup_last_pt": True}
    assert remove_last_checkpoint(job) is None and last.exists()            # no best.pt yet: keep the only checkpoint
    (run / "weights" / "best.pt").write_bytes(b"B")
    assert remove_last_checkpoint({**job, "cleanup_last_pt": False}) is None and last.exists()
    assert remove_last_checkpoint(job) == str(last.resolve()) and not last.exists()
    assert (other / "last.pt").exists() and (run / "weights" / "best.pt").exists()
    assert remove_last_checkpoint(job) is None                               # nothing left to remove


def test_runner_holds_a_job_that_fails_the_disk_check_and_writes_a_heartbeat(tmp_path):
    q = {"logs": str(tmp_path / "logs"), "jobs": [
        {"id": "big", "cmd": ["-c", "pass"], "est_output_gb": 10 ** 7},
        {"id": "after", "after": ["big"], "cmd": ["-c", "pass"]}]}
    qpath = tmp_path / "q.yaml"
    save_queue(q, qpath)
    r = _runner(qpath)
    jobs = {j["id"]: j for j in load_queue(qpath)["jobs"]}
    assert jobs["big"]["status"] == "held" and "disk check" in jobs["big"]["note"], r.stdout + r.stderr
    assert jobs["after"]["status"] == "pending"
    assert (tmp_path / "heartbeat.json").exists()
    assert "runner:" in _runner(qpath, "--status").stdout


def test_watchdog_starts_only_a_dead_runner_with_allowed_work():
    from confdet.utils.jobqueue import watchdog_decision

    def _q(jobs):
        return {"jobs": [{"after": [], "status": "pending", **j} for j in jobs]}
    q = _q([{"id": "a", "cmd": ["x"], "status": "running"}, {"id": "b", "cmd": ["x"], "after": ["a"]}])
    old = {"epoch_s": 0.0, "runner_pid": 11}
    dead = lambda pid: False                                            # noqa: E731
    live = lambda pid: pid == 11                                        # noqa: E731
    assert watchdog_decision(q, old, 1000.0, 11, False, alive=dead)[0] is True
    assert watchdog_decision(q, None, 1000.0, None, False, alive=dead)[0] is True       # no heartbeat file
    # never a second runner: live lock / heartbeat PID, or the task still running
    assert watchdog_decision(q, old, 1000.0, 11, False, alive=live)[0] is False
    assert watchdog_decision(q, {"epoch_s": 0.0, "runner_pid": 11}, 1000.0, None, False, alive=live)[0] is False
    assert watchdog_decision(q, old, 1000.0, None, True, alive=dead)[0] is False
    # fresh heartbeat (<= 5 min)
    assert watchdog_decision(q, {"epoch_s": 800.0, "runner_pid": 11}, 1000.0, None, False, alive=dead)[0] is False
    # stops that wait for a human are respected
    qf = _q([{"id": "a", "cmd": ["x"], "status": "failed"}, {"id": "b", "cmd": ["x"]}])
    assert watchdog_decision(qf, old, 1000.0, None, False, alive=dead)[0] is False
    qd = _q([{"id": "a", "cmd": ["x"], "status": "held", "note": "held by the disk check ..."}, {"id": "b", "cmd": ["x"]}])
    assert watchdog_decision(qd, old, 1000.0, None, False, alive=dead)[0] is False
    qh = _q([{"id": "a", "cmd": ["x"], "status": "held", "note": "held until review"}, {"id": "b", "cmd": ["x"]}])
    assert watchdog_decision(qh, old, 1000.0, None, False, alive=dead)[0] is True    # a review hold is not a stop
    qn = _q([{"id": "a", "cmd": ["x"], "status": "done"}])
    assert watchdog_decision(qn, old, 1000.0, None, False, alive=dead) == (False, "nothing to run")

"""The Kaggle helpers on the local machine: a fake kaggle CLI, local processes, no Kaggle call."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

KAGGLE = Path(__file__).resolve().parents[1] / "kaggle"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, KAGGLE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = load("kaggle_common", "common.py")
queue = load("kaggle_queue", "queue.py")


def test_sh_timeout_kills_the_whole_command(tmp_path):
    pid_file = tmp_path / "pid"
    with pytest.raises(subprocess.TimeoutExpired):
        common.sh(f"sleep 30 & echo $! > {pid_file}; wait", timeout=1)
    pid = int(pid_file.read_text())
    for _ in range(50):  # the killed child is reaped by init shortly after
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the command's child process survived the timeout")


def test_sh_raises_on_failure():
    with pytest.raises(subprocess.CalledProcessError):
        common.sh("exit 3", timeout=10)
    common.sh("true", timeout=10)


def test_remaining_is_never_zero(monkeypatch):
    monkeypatch.setattr(common, "START", time.time() - 2 * common.DEADLINE)
    assert common.remaining() == 1.0


def done_proc(code, out=""):
    return subprocess.CompletedProcess(["kaggle"], code, out, "")


@pytest.mark.parametrize("answer, expected", [
    (done_proc(0, "spritker/monodist-cv-f0-a has status \"KernelWorkerStatus.RUNNING\"\n"),
     "spritker/monodist-cv-f0-a has status \"KernelWorkerStatus.RUNNING\""),
    (done_proc(1, f"Cannot access kernel 'x' ({queue.NOT_FOUND}). The most likely cause ..."), ""),  # never pushed
    (done_proc(1, "ConnectionError: Max retries exceeded"), None),  # unknown: must not push
    (None, None),  # CLI timeout
])
def test_status(monkeypatch, answer, expected):
    monkeypatch.setattr(queue, "kaggle", lambda *a: answer)
    assert queue.status("spritker/monodist-cv-f0-a") == expected


def write_job(out, folder, name, seqs, commit="abc123"):
    (out / folder).mkdir(parents=True)
    (out / folder / "cv.jsonl").write_text(json.dumps({"step": "setup", "commit": commit}) + "\n")
    for h in ("nms", "e2e"):
        d = out / folder / "det" / f"{name}_{h}"
        d.mkdir(parents=True)
        for s in seqs:
            np.savez(d / f"{s}.npz", frame=np.array([0, 1]), box=np.zeros((2, 4)), score=np.array([.9, .8]),
                     cls=np.array([0, 1]), image_hw=np.array([375, 1242]))


def test_done_checks_files_content_and_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "OUT", tmp_path)
    seqs = ["0000", "0001"]
    write_job(tmp_path, "f0-a", "f0_a", seqs)
    assert queue.done("f0-a", "f0-a", seqs, "abc123")
    assert not queue.done("f0-a", "f0-a", seqs, "other")  # output of another commit
    assert not queue.done("f0-a", "f0-a", seqs + ["0002"], "abc123")  # a sequence missing
    bad = tmp_path / "f0-a" / "det" / "f0_a_e2e" / "0001.npz"
    bad.write_bytes(b"truncated")
    assert not queue.done("f0-a", "f0-a", seqs, "abc123")
    np.savez(bad, frame=np.array([5000]), box=np.zeros((1, 4)), score=np.array([.9]), cls=np.array([0]),
             image_hw=np.array([375, 1242]))  # frame outside sequence 0001
    assert not queue.done("f0-a", "f0-a", seqs, "abc123")


def test_failed_jobs_give_a_failing_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "OUT", tmp_path)
    monkeypatch.setattr(queue, "status", lambda slug: "has status \"KernelWorkerStatus.ERROR\"")
    monkeypatch.setattr(queue.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["queue.py", "--commit", "abc123", "--jobs", "f0-a"])
    assert queue.main() == 1


def test_unknown_status_is_not_pushed(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "OUT", tmp_path)
    answers = iter([None, "has status \"KernelWorkerStatus.CANCEL_ACKNOWLEDGED\""])
    monkeypatch.setattr(queue, "status", lambda slug: next(answers))
    monkeypatch.setattr(queue.time, "sleep", lambda s: None)
    pushes = []
    real_run = subprocess.run
    monkeypatch.setattr(queue.subprocess, "run", lambda cmd, **kw: pushes.append(cmd) or real_run(["true"]))
    monkeypatch.setattr(sys, "argv", ["queue.py", "--commit", "abc123", "--jobs", "f0-a"])
    assert queue.main() == 1 and pushes == []


def test_finished_queue_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "OUT", tmp_path)
    write_job(tmp_path, "f0-a", "f0_a", queue.kitti.SEQUENCES)
    monkeypatch.setattr(sys, "argv", ["queue.py", "--commit", "abc123", "--jobs", "f0-a"])
    assert queue.main() == 0

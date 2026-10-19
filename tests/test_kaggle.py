"""The Kaggle helpers on the local machine: a fake kaggle CLI, local processes and files, no Kaggle call."""
import hashlib
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
verify = load("kaggle_verify", "verify.py")


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


@pytest.mark.parametrize("names", [[], ["Tesla T4"], ["A100", "A100"]])
def test_remote_setup_refuses_the_wrong_gpu_allocation(monkeypatch, names):
    monkeypatch.setattr(common, "LOCAL", None)
    with pytest.raises(RuntimeError, match="expected two T4 GPUs"):
        common.check_gpus(names)


def test_remote_setup_accepts_two_t4s(monkeypatch):
    monkeypatch.setattr(common, "LOCAL", None)
    common.check_gpus(["Tesla T4", "Tesla T4"])


def test_local_check_can_use_one_laptop_gpu(monkeypatch):
    monkeypatch.setattr(common, "LOCAL", "/tmp/local-check")
    common.check_gpus(["NVIDIA RTX 1000 Ada Generation Laptop GPU"])


def done_proc(code, out=""):
    return subprocess.CompletedProcess(["kaggle"], code, out, "")


@pytest.mark.parametrize("answer, expected", [
    (done_proc(0, "spritker/monodist-train has status \"KernelWorkerStatus.RUNNING\"\n"),
     "spritker/monodist-train has status \"KernelWorkerStatus.RUNNING\""),
    (done_proc(1, f"Cannot access kernel 'x' ({verify.NOT_FOUND}). The most likely cause ..."), ""),  # never pushed
    (done_proc(1, "ConnectionError: Max retries exceeded"), None),  # unknown
    (None, None),  # CLI timeout
])
def test_status(monkeypatch, answer, expected):
    monkeypatch.setattr(verify, "kaggle", lambda *a: answer)
    assert verify.status("spritker/monodist-train") == expected


def write_output(out, jobs, commit="abc123", mode="full", failed=()):
    """A downloaded train-kernel output: log, weights and detections of every job."""
    out.mkdir(parents=True, exist_ok=True)
    log = [{"step": "setup", "commit": commit, "mode": mode}, {"step": "end", "failed": list(failed)}]
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    for job, seqs in jobs.items():
        sha = None
        if job != "coco":
            w = out / "runs" / job / "weights"
            w.mkdir(parents=True)
            (w / "best.pt").write_bytes(job.encode())
            sha = hashlib.sha256(job.encode()).hexdigest()
        for h in ("nms", "e2e"):
            d = out / "det" / f"{job}_{h}"
            d.mkdir(parents=True)
            (d / "run.json").write_text(json.dumps({"weights": {"name": "best.pt", "sha256": sha}, "nms": h == "nms",
                                                    "frames": 20 if mode == "smoke" else None}))
            for s in seqs:
                np.savez(d / f"{s}.npz", frame=np.array([0, 1]), box=np.zeros((2, 4)), score=np.array([.9, .8]),
                         cls=np.array([0, 1]), image_hw=np.array([375, 1242]))


def test_check_accepts_a_complete_output(tmp_path):
    write_output(tmp_path, verify.expected_jobs())
    assert verify.check(tmp_path, "abc123", "full") == []


def test_check_finds_every_kind_of_problem(tmp_path):
    jobs = verify.expected_jobs()
    write_output(tmp_path, jobs, failed=["f1_b"])
    assert verify.check(tmp_path, "other", "full")[0].startswith("log: setup")       # another commit
    assert any("failed jobs" in p for p in verify.check(tmp_path, "abc123", "full"))
    write_output(tmp_path / "b", jobs)
    (tmp_path / "b" / "det" / "f0_a_e2e" / "0001.npz").write_bytes(b"truncated")
    (tmp_path / "b" / "det" / "coco_nms" / "0000.npz").unlink()
    np.savez(tmp_path / "b" / "det" / "final_a_nms" / "0001.npz", frame=np.array([5000]), box=np.zeros((1, 4)),
             score=np.array([.9]), cls=np.array([0]), image_hw=np.array([375, 1242]))  # frame outside the sequence
    (tmp_path / "b" / "runs" / "f2_b" / "weights" / "best.pt").write_bytes(b"retrained")  # detections of other weights
    problems = verify.check(tmp_path / "b", "abc123", "full")
    for text in ("f0_a_e2e: missing or invalid", "coco_nms: missing or invalid", "final_a_nms: missing or invalid",
                 "f2_b_nms: detections were made with other weights"):
        assert any(p.startswith(text) for p in problems), text
    assert verify.check(tmp_path / "b", "abc123", "smoke")  # settings of a full run are not a smoke run


def test_failing_check_gives_exit_code_1(tmp_path, monkeypatch):
    write_output(tmp_path, {"coco": ["0000"]})
    monkeypatch.setattr(sys, "argv", ["verify.py", "--commit", "abc123", "--dir", str(tmp_path)])
    assert verify.main() == 1  # every other job is missing
    monkeypatch.setattr(sys, "argv", ["verify.py", "--commit", "abc123", "--dir", str(tmp_path), "--jobs", "coco"])
    monkeypatch.setattr(verify, "expected_jobs", lambda: {"coco": ["0000"]})
    assert verify.main() == 0


def test_download_retries_a_failed_cli_call_and_reports_it(tmp_path, monkeypatch, capsys):
    calls = []

    def fake(*args, timeout=None):
        calls.append(args)
        if args[:2] == ("kernels", "status"):
            return done_proc(0, "x has status \"KernelWorkerStatus.COMPLETE\"\n")
        return done_proc(1, "HTTPError 429")  # the download fails every time
    monkeypatch.setattr(verify, "kaggle", fake)
    monkeypatch.setattr(verify.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["verify.py", "--commit", "abc123", "--dir", str(tmp_path), "--download"])
    assert verify.main() == 1
    assert sum(c[:2] == ("kernels", "output") for c in calls) == verify.DOWNLOAD_TRIES
    assert "download failed" in capsys.readouterr().out

#!/usr/bin/env python3
"""Continue the authorized C3 matrix, resuming interrupted runs from their checkpoints."""
import collections
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
PROJECT = HERE / "MT_DS_PL_C3"
PYTHON = HERE.parent / ".conda/envs/SSL29/bin/python"
MAX_ITERATIONS = 15000
MIN_FREE_MIB = 9000
MAX_ACTIVE = 2
MAX_ATTEMPTS = 3
POLL_SECONDS = 30
SETTLE_SECONDS = 20

sys.path.insert(0, str(PROJECT / "code"))
from provenance import source_manifest


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def completed(job):
    output = Path(job["output_dir"])
    training = output / "training_metrics.csv"
    try:
        with training.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        last = int(rows[-1]["iteration"]) if rows else 0
    except (OSError, ValueError, KeyError):
        return False
    return (last >= MAX_ITERATIONS and (output / "best_metrics.json").is_file()
            and (output / "test_metrics.json").is_file())


def trim_logs_to_checkpoint(output, iteration):
    """Keep only metrics covered by the checkpoint; preserve original files in a backup."""
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("pre_resume_%Y%m%d_%H%M%S_%f")
    backup = output / stamp
    changed = []
    for name in ("training_metrics.csv", "diagnostics.csv", "boundary_metrics.csv", "transfer_metrics.csv"):
        path = output / name
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        kept = [row for row in rows if int(row["iteration"]) <= iteration]
        if len(kept) != len(rows):
            changed.append((path, fieldnames, kept))
    path = output / "validation_metrics.jsonl"
    if path.is_file():
        lines = path.read_text().splitlines()
        kept_lines = [line for line in lines if int(json.loads(line)["iteration"]) <= iteration]
        if len(kept_lines) != len(lines):
            changed.append((path, None, kept_lines))
    if changed:
        backup.mkdir()
        for path, fieldnames, kept in changed:
            shutil.copy2(path, backup / path.name)
            if fieldnames is None:
                path.write_text("".join(line + "\n" for line in kept))
            else:
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(kept)
        return str(backup)
    return None


def prepare_resume(job):
    import torch

    output = Path(job["output_dir"])
    checkpoint_path = output / "last_checkpoint.pth"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"No resumable checkpoint for {job['output_dir']}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["config"].get("source_sha256") != source_manifest(PROJECT):
        raise RuntimeError(f"Checkpoint source hash mismatch: {checkpoint_path}")
    iteration = int(checkpoint["iteration"])
    backup = trim_logs_to_checkpoint(output, iteration)
    job["resume_checkpoint"] = str(checkpoint_path)
    job["resume_iteration"] = iteration
    if backup:
        job["pre_resume_metrics_backup"] = backup
    return iteration


def gpu_free_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    available = {}
    for line in result.stdout.splitlines():
        index, memory = (part.strip() for part in line.split(",", 1))
        available[int(index)] = int(memory)
    return available


def main():
    if not PYTHON.is_file():
        raise SystemExit(f"SSL29 Python not found: {PYTHON}")
    old_manifest = PROJECT / "Results_transfer/launch_logs/20261113_170940_047368/manifest.json"
    plan = json.loads(old_manifest.read_text())
    expected_sources = plan["source_sha256"]
    if source_manifest(PROJECT) != expected_sources:
        raise SystemExit("C3 source differs from the frozen source used by the original runs")

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    logs = PROJECT / "Results_transfer/launch_logs" / f"continuation_{stamp}"
    logs.mkdir(parents=True, exist_ok=False)
    (logs / "parent_pid.txt").write_text(f"{os.getpid()}\n")

    jobs = plan["jobs"]
    pending = collections.deque()
    for job in jobs:
        job["status"] = "pending"
        job["attempts"] = 0
        if completed(job):
            job["status"] = "finished"
            job["reconciled_at"] = now()
            job["reconciled_from_stale_status"] = True
            continue
        output = Path(job["output_dir"])
        if output.exists():
            prepare_resume(job)
            job["attempts"] = 1
            job["resume_from_previous_queue"] = True
        pending.append(job)

    active = {}
    stop_on_failure = False

    def counts():
        return {state: sum(job["status"] == state for job in jobs)
                for state in ("pending", "running", "finished", "failed")}

    def persist(phase="running"):
        plan["continuation_started_at"] = plan.get("continuation_started_at", now())
        plan["continuation_source_manifest"] = str(old_manifest)
        plan["continuation_limits"] = dict(min_free_mib=MIN_FREE_MIB, max_active=MAX_ACTIVE,
                                           max_attempts=MAX_ATTEMPTS)
        plan["updated_at"] = now()
        save_json(logs / "manifest.json", plan)
        status = dict(manifest=str(logs / "manifest.json"), counts=counts(), phase=phase,
                      updated_at=now(), min_free_mib=MIN_FREE_MIB, max_active=MAX_ACTIVE)
        save_json(PROJECT / "queue_status.json", status)

    def refresh():
        nonlocal stop_on_failure
        for process, gpu, job, log_path in list(active.values()):
            code = process.poll()
            if code is None:
                continue
            active.pop(job["output_dir"], None)
            job["exit_code"] = code
            job["finished_at"] = now()
            if code == 0 and completed(job):
                job["status"] = "finished"
                print(f"{now()} finished {job['mode']} {job['dataset']} seed={job['seed']}", flush=True)
                continue
            tail = log_path.read_text(errors="replace")[-12000:].lower()
            is_oom = "out of memory" in tail or "outofmemoryerror" in tail
            if is_oom and job["attempts"] < MAX_ATTEMPTS:
                try:
                    prepare_resume(job)
                    job["status"] = "pending"
                    pending.appendleft(job)
                    print(f"{now()} OOM; queued resume at iteration {job['resume_iteration']}: "
                          f"{job['mode']} {job['dataset']} seed={job['seed']}", flush=True)
                except Exception as error:
                    job["status"] = "failed"
                    job["error"] = str(error)
                    stop_on_failure = True
            else:
                job["status"] = "failed"
                job["error"] = "OOM after retry limit" if is_oom else "nonzero exit or missing final metrics"
                stop_on_failure = True
                print(f"{now()} failed {job['mode']} {job['dataset']} seed={job['seed']} code={code}", flush=True)
        persist("failed" if stop_on_failure else "running")

    persist()
    print(f"Continuation queue: {counts()}; manifest={logs / 'manifest.json'}", flush=True)
    while active or (pending and not stop_on_failure):
        refresh()
        if not stop_on_failure and len(active) < MAX_ACTIVE and pending:
            try:
                free = gpu_free_memory()
            except (subprocess.CalledProcessError, FileNotFoundError) as error:
                print(f"{now()} GPU query unavailable ({error}); retrying", flush=True)
                free = {}
            active_gpus = {entry[1] for entry in active.values()}
            candidates = [gpu for gpu, memory in free.items()
                          if memory >= MIN_FREE_MIB and gpu not in active_gpus]
            if candidates:
                gpu = max(candidates, key=free.get)
                job = pending.popleft()
                command = list(job["command"])
                if job.get("resume_checkpoint"):
                    command.extend(["--resume", job["resume_checkpoint"]])
                logfile = logs / (f"{job['mode']}_{job['dataset']}_seed{job['seed']}"
                                  f"_attempt{job['attempts'] + 1}.log")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                                   MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
                handle = logfile.open("x")
                process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                           env=environment)
                handle.close()
                job["status"] = "running"
                job["gpu"] = gpu
                job["pid"] = process.pid
                job["log"] = str(logfile)
                job["started_at"] = now()
                job["attempts"] += 1
                active[job["output_dir"]] = (process, gpu, job, logfile)
                print(f"{now()} started {job['mode']} {job['dataset']} seed={job['seed']} "
                      f"gpu={gpu} pid={process.pid} free_mib={free[gpu]}", flush=True)
                persist()
                time.sleep(SETTLE_SECONDS)
                continue
        time.sleep(POLL_SECONDS)

    if stop_on_failure or counts()["failed"]:
        persist("failed")
        raise SystemExit("Continuation stopped after a failed run; inspect the continuation manifest/logs")
    if counts()["finished"] == len(jobs):
        report = subprocess.run([str(PYTHON), str(PROJECT / "scripts/report_transfer.py")],
                                cwd=PROJECT, capture_output=True, text=True)
        (logs / "report_stdout.txt").write_text(report.stdout)
        (logs / "report_stderr.txt").write_text(report.stderr)
        persist("reported" if report.returncode == 0 else "report_failed")
        if report.returncode:
            raise SystemExit(f"All jobs finished but report generation failed; see {logs}")
        print(f"All 24 runs finished. Report: {PROJECT / 'summary/transfer_report.md'}", flush=True)
    else:
        persist("incomplete")


if __name__ == "__main__":
    main()

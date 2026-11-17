#!/usr/bin/env python3
"""Plan/queue the fixed E4 sham/Top/Random/Bottom matrix; fail clearly without a GPU."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from provenance import source_manifest


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["none", "sham", "cbcd", "random", "bottom"],
                        default=["sham", "cbcd", "random", "bottom"])
    parser.add_argument("--datasets", nargs="+", choices=["LA", "Pancreas"], default=["LA", "Pancreas"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 500, 1337])
    parser.add_argument("--level", choices=["e3", "e4"], default="e4")
    parser.add_argument("--output_root", type=Path, default=PROJECT / "Results_transfer")
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--memory_mib", type=int, default=7500)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_seconds", type=float, default=30)
    parser.add_argument("--settle_seconds", type=float, default=30)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if min(args.memory_mib, args.max_per_gpu, args.poll_seconds) <= 0 or args.settle_seconds < 0:
        parser.error("Memory, concurrency and poll interval must be positive; settle interval nonnegative")
    if any(len(v) != len(set(v)) for v in (args.modes, args.datasets, args.seeds)):
        parser.error("Duplicate modes, datasets or seeds")
    root = args.output_root.resolve()
    jobs = []
    for seed in args.seeds:
        for dataset in args.datasets:
            for mode in args.modes:
                output = root / f"{mode}_{args.level}" / f"seed_{seed}" / f"{dataset}_{8 if dataset == 'LA' else 6}l"
                if output.exists():
                    raise FileExistsError(f"Refusing to overwrite run: {output}")
                command = [sys.executable, "-u", str(PROJECT / "code/train.py"), "--dataset", dataset,
                           "--probe_level", args.level, "--transfer_mode", mode, "--seed", str(seed),
                           "--output_dir", str(output)]
                jobs.append(dict(mode=mode, seed=seed, dataset=dataset, output_dir=str(output), command=command,
                                 status="pending"))
    sources = source_manifest(PROJECT)
    plan = dict(created_at=now(), source_sha256=sources, jobs=jobs)
    if args.plan:
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open("x") as handle:
            json.dump(plan, handle, indent=2)
            handle.write("\n")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: no training launched. The --dry_run plan can be reviewed on CPU.")
    logs = root / "launch_logs" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    logs.mkdir(parents=True, exist_ok=False)
    processes = []
    def refresh():
        failures = []
        for gpu, process, item in processes:
            code = process.poll()
            if code is not None:
                item.update(status="finished" if code == 0 else "failed", exit_code=code)
                if code != 0:
                    failures.append(item["log"])
        temporary = logs / "manifest.json.tmp"
        temporary.write_text(json.dumps(plan, indent=2) + "\n")
        temporary.replace(logs / "manifest.json")
        if failures:
            raise RuntimeError(f"Training failed; pending jobs not launched. Inspect {failures}")
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    for job in jobs:
        while True:
            refresh()
            if source_manifest(PROJECT) != sources:
                raise RuntimeError("Sources changed after planning; pending jobs were not launched")
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
            free = {int(index): int(memory) for index, memory in (line.split(",") for line in raw.splitlines())}
            active = {gpu: sum(p.poll() is None and g == gpu for g, p, item in processes) for gpu in free}
            candidates = [gpu for gpu in free if free[gpu] >= args.memory_mib and active[gpu] < args.max_per_gpu
                          and (args.gpus is None or gpu in args.gpus)]
            if candidates:
                gpu = max(candidates, key=free.get)
                break
            if args.gpus is not None and not set(args.gpus) & free.keys():
                raise SystemExit(f"Requested GPUs {args.gpus} not found; available indices: {sorted(free)}")
            print(f"{now()} waiting for GPU memory; free={free}", flush=True)
            time.sleep(args.poll_seconds)
        logfile = logs / f"{job['mode']}_{job['seed']}_{job['dataset']}.log"
        with logfile.open("x") as handle:
            process = subprocess.Popen(job["command"], cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpu)))
        processes.append((gpu, process, job))
        job.update(gpu=gpu, pid=process.pid, log=str(logfile), started_at=now(), status="running")
        refresh()
        print(f"{now()} started {job['mode']} {job['dataset']} seed={job['seed']} gpu={gpu} pid={process.pid}", flush=True)
        time.sleep(args.settle_seconds)
    print(f"All {len(jobs)} jobs started. Monitoring until finished. Manifest: {logs / 'manifest.json'}", flush=True)
    while any(p.poll() is None for gpu, p, item in processes):
        refresh()
        time.sleep(args.poll_seconds)
    refresh()
    print(f"All {len(jobs)} jobs finished successfully.", flush=True)


if __name__ == "__main__":
    main()

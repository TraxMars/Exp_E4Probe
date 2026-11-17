#!/usr/bin/env python3
"""Offline CBCD analysis of the paired, final C1 Student/probe checkpoints."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
C1 = PROJECT.parent / "MT_DS_PL_C1"
sys.path[:0] = [str(PROJECT / "code"), str(C1 / "code")]
from boundary_probe import BoundaryProbe, LEVEL_CHANNELS, label_to_boundary
from channel_analysis import channel_scores, encoder_feature, occlusion_deltas, sample_patch
from data import SPECS, case_path, check_case, load_case
from networks import VNetDS


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*keys):
    return int.from_bytes(hashlib.sha256(json.dumps(keys).encode()).digest()[:4], "little")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def collect_jobs(args):
    jobs, cases, data_files = [], {}, set()
    for level in args.levels:
        for dataset in args.datasets:
            for seed in args.seeds:
                run = C1 / f"Results_C1_{level.upper()}" / f"seed_{seed}" / f"{dataset}_{SPECS[dataset]['labels']}l"
                config = json.loads((run / "config.json").read_text())
                stored = config["args"]
                if stored["probe_level"] != level or stored["dataset"] != dataset or stored["seed"] != seed:
                    raise ValueError(f"Run identity mismatch: {run}")
                root = Path(stored["data_root"]) / SPECS[dataset]["directory"]
                description = dict(ids=config["labeled_ids"], patch_size=stored["patch_size"], root=str(root))
                if dataset in cases and cases[dataset] != description:
                    raise ValueError(f"Different labeled cases, crop size or data root for {dataset}")
                cases[dataset] = description
                data_files.update(case_path(root, dataset, case) for case in description["ids"])
                jobs.append(dict(level=level, dataset=dataset, seed=seed, run=str(run),
                                 checkpoint=str(run / "last_checkpoint.pth")))
    return jobs, cases, data_files


def build_patches(args, cases, output):
    patches, manifest = {}, []
    for dataset, desc in cases.items():
        for case in desc["ids"]:
            path = case_path(desc["root"], dataset, case)
            image, label = load_case(path)
            check_case(image, label, path)
            for index in range(args.crops_per_case):
                seed = stable_seed(args.sampling_seed, dataset, case, index)
                inputs, target, crop = sample_patch(image, label, desc["patch_size"],
                                                   SPECS[dataset]["rotation"], np.random.RandomState(seed))
                patches[dataset, case, index] = (inputs, target)
                manifest.append(dict(dataset=dataset, case=case, crop=index, crop_seed=seed, **crop))
    write_json(output / "patches.json", manifest)
    return patches


def analyze_run(job, args, cases, patches, output):
    run_name = f"{job['level']}_{job['dataset']}_seed_{job['seed']}"
    destination = output / run_name
    if args.resume and (destination / "summary.json").is_file():
        print(f"Using completed run {run_name}", flush=True)
        return json.loads((destination / "summary.json").read_text())
    destination.mkdir(exist_ok=True)
    checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    stored = checkpoint["config"]["args"]
    public = json.loads((Path(job["run"]) / "config.json").read_text())
    if checkpoint["config"] != public or checkpoint["iteration"] != stored["max_iterations"]:
        raise ValueError(f"Checkpoint is mismatched or unfinished: {run_name}")
    if checkpoint.get("probe") is None:
        raise ValueError(f"No paired boundary probe: {run_name}")
    with torch.random.fork_rng(devices=[]):
        student = VNetDS(has_dropout=True)
        probe = BoundaryProbe(LEVEL_CHANNELS[job["level"]])
    student.load_state_dict(checkpoint["student"], strict=True)
    probe.load_state_dict(checkpoint["probe"], strict=True)
    del checkpoint
    student.to(args.device).eval().requires_grad_(False)
    probe.to(args.device).eval().requires_grad_(False)
    before = {name: value.detach().clone() for name, value in probe.state_dict().items()}
    rows = []
    score_fields = ["case", "crop", "channel", "cbcd", "rank_descending"]
    fields = ["case", "crop", "status", "boundary_fraction", "reference_loss", "top_delta", "bottom_delta",
              "random_delta", "random_std", "top_minus_random", "top_minus_bottom"]
    with (destination / "channel_scores.csv").open("w", newline="") as score_handle, \
            (destination / "occlusion.csv").open("w", newline="", buffering=1) as occlusion_handle:
        score_writer = csv.DictWriter(score_handle, fieldnames=score_fields)
        writer = csv.DictWriter(occlusion_handle, fieldnames=fields)
        score_writer.writeheader()
        writer.writeheader()
        for case in cases[job["dataset"]]["ids"]:
            for index in range(args.crops_per_case):
                inputs, labels = patches[job["dataset"], case, index]
                target = label_to_boundary(labels.to(args.device))
                row = dict(case=case, crop=index, status="valid", boundary_fraction=target.mean().item())
                if not target.any():
                    row["status"] = "empty_boundary"
                else:
                    feature = encoder_feature(student, inputs.to(args.device), job["level"])
                    scores = channel_scores(probe, feature, target)
                    if scores.max() == scores.min():
                        row["status"] = "flat_scores"
                    else:
                        rank = scores[0].argsort(descending=True, stable=True).argsort()
                        for channel in range(scores.shape[1]):
                            score_writer.writerow(dict(case=case, crop=index, channel=channel,
                                                       cbcd=scores[0, channel].item(),
                                                       rank_descending=rank[channel].item()))
                        generator = torch.Generator().manual_seed(stable_seed(
                            args.sampling_seed, job["level"], job["dataset"], case, index, "occlusion"))
                        row.update(occlusion_deltas(probe, feature, target, scores, args.topk,
                                                   args.random_repeats, generator))
                    del feature, scores
                writer.writerow(row)
                rows.append(row)
                print(f"{run_name} {case} crop={index}: " + (
                    f"top-random={row['top_minus_random']:.5f}, top-bottom={row['top_minus_bottom']:.5f}"
                    if row["status"] == "valid" else row["status"]), flush=True)
    for name, value in probe.state_dict().items():
        if not torch.equal(value, before[name]):
            raise RuntimeError(f"Analysis changed probe state: {name}")
    if any(p.grad is not None for module in (student, probe) for p in module.parameters()):
        raise RuntimeError("Analysis wrote parameter gradients")
    valid = [r for r in rows if r["status"] == "valid"]
    keys = ["top_delta", "bottom_delta", "random_delta", "top_minus_random", "top_minus_bottom"]
    summary = dict(**job, iteration=stored["max_iterations"], patches=len(rows), valid_patches=len(valid),
                   skipped_patches=len(rows) - len(valid), rows=rows,
                   means={k: statistics.mean(r[k] for r in valid) if valid else None for k in keys})
    write_json(destination / "summary.json", summary)
    return summary


def summarize(runs, args, cases):
    groups = []
    for level in args.levels:
        for dataset in args.datasets:
            selected = [run for run in runs if run["level"] == level and run["dataset"] == dataset]
            metrics = {}
            complete = {r["seed"] for r in selected} == {42, 500, 1337}
            complete &= all(any(row["case"] == case and row["status"] == "valid" for row in run["rows"])
                            for run in selected for case in cases[dataset]["ids"])
            for key in ("top_minus_random", "top_minus_bottom"):
                case_values = []
                for case in cases[dataset]["ids"]:
                    by_seed = [[row[key] for row in run["rows"] if row["case"] == case and row["status"] == "valid"]
                               for run in selected]
                    if all(by_seed):
                        case_values.append(statistics.mean(statistics.mean(v) for v in by_seed))
                rng = np.random.default_rng(stable_seed(args.sampling_seed, dataset, "bootstrap"))
                if len(case_values) >= 2:
                    sampled = rng.choice(case_values, size=(5000, len(case_values)), replace=True).mean(axis=1)
                    lower, upper = map(float, np.quantile(sampled, [0.025, 0.975]))
                    mean = statistics.mean(case_values)
                else:
                    lower = upper = mean = None
                means = [run["means"][key] for run in selected]
                metrics[key] = dict(mean=mean, case_bootstrap_95=[lower, upper], cases=len(case_values),
                                    seed_means={str(r["seed"]): r["means"][key] for r in selected})
                metrics[key]["pass"] = bool(complete and all(v is not None and v > 0 for v in means)
                                            and lower is not None and lower > 0)
            groups.append(dict(level=level, dataset=dataset, complete=bool(complete), metrics=metrics,
                               passed=all(v["pass"] for v in metrics.values())))
    levels = {level: bool({g["dataset"] for g in groups if g["level"] == level} == {"LA", "Pancreas"}
                         and all(g["passed"] for g in groups if g["level"] == level)) for level in args.levels}
    return dict(status="finished", groups=groups, level_pass=levels, completed_runs=len(runs))


def report(runs, summary, output):
    lines = ["# C2 CBCD 离线遮挡诊断", "", "配套 final Student + probe；eval；仅有标签训练 patch；无训练更新。",
             "损失差单位为 BCE + soft Dice。下表为有效 patch 均值；Random 为每 patch 多次随机遮挡的均值。", "",
             "| 层级 | 数据集 | seed | 有效/总 patch | Top Δ | Random Δ | Bottom Δ | Top−Random |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        values = [run["means"][key] for key in ("top_delta", "random_delta", "bottom_delta", "top_minus_random")]
        lines.append(f"| {run['level']} | {run['dataset']} | {run['seed']} | {run['valid_patches']}/{run['patches']} | "
                     + " | ".join(f"{v:.5f}" if v is not None else "无有效 patch" for v in values) + " |")
    lines += ["", "病例内先平均 crop 和 seed，然后按病例 bootstrap；区间仅描述这些训练病例的 probe 敏感性。", "",
              "| 层级 | 数据集 | 对比 | 病例均值 | 病例 bootstrap 95% | 完整组合 | 判据 |",
              "|---|---|---|---:|---|---|---|"]
    for group in summary["groups"]:
        for key, result in group["metrics"].items():
            value = f"{result['mean']:.5f}" if result["mean"] is not None else "NA"
            interval = ", ".join(f"{v:.5f}" if v is not None else "NA" for v in result["case_bootstrap_95"])
            lines.append(f"| {group['level']} | {group['dataset']} | {key} | {value} | [{interval}] | "
                         f"{group['complete']} | {'通过' if result['pass'] else '不通过/不完整'} |")
    lines += ["", "## 承接判断", ""]
    for level, passed in summary["level_pass"].items():
        lines.append(f"- {level.upper()}：{'通过通道排序的前置诊断，可进入最小迁移试验。' if passed else '未通过完整前置判据，见上述具体环节。'}")
    lines += ["", "遮挡实验检验 probe 对通道的依赖，尚不能证明无标签迁移或分割收益。",
              "现有 GPU 在本会话不可用；本报告没有产生新的训练 Dice。", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=list(LEVEL_CHANNELS), default=["e4", "e3"])
    parser.add_argument("--datasets", nargs="+", choices=["LA", "Pancreas"], default=["LA", "Pancreas"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 500, 1337])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--sampling_seed", type=int, default=20261113)
    parser.add_argument("--crops_per_case", type=int, default=2)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--random_repeats", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.cpu_threads, args.crops_per_case, args.random_repeats, args.topk) < 1:
        parser.error("Threads, crops, repeats and k must be positive")
    if any(args.topk >= LEVEL_CHANNELS[level] for level in args.levels):
        parser.error("k must be smaller than the feature channel count")
    if any(len(v) != len(set(v)) for v in (args.levels, args.datasets, args.seeds)):
        parser.error("Duplicate levels, datasets or seeds")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this session")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}; use --resume for matching unfinished work")
    jobs, cases, data_files = collect_jobs(args)
    inputs = {Path(job["checkpoint"]) for job in jobs} | {Path(job["run"]) / "config.json" for job in jobs} | data_files
    sources = [Path(__file__), PROJECT / "code/channel_analysis.py", PROJECT / "README.md",
               C1 / "code/boundary_probe.py", C1 / "code/networks/vnet_ds.py", C1 / "code/data.py"]
    print(f"Hashing {len(inputs)} input files and {len(sources)} source files", flush=True)
    parameters = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("resume", "output_dir")}
    config = dict(parameters=parameters, torch_version=torch.__version__, numpy_version=np.__version__,
                  inputs={str(p.resolve()): sha256(p) for p in sorted(inputs)},
                  sources={str(p.resolve()): sha256(p) for p in sources})
    if args.resume:
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume rejected: parameters, environment, source or input files changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config)
    start = time.monotonic()
    patches = build_patches(args, cases, output)
    runs = []
    for job in jobs:
        runs.append(analyze_run(job, args, cases, patches, output))
        write_json(output / "progress.json", dict(completed_runs=len(runs), total_runs=len(jobs),
                   updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                   elapsed_seconds=time.monotonic() - start))
    summary = summarize(runs, args, cases)
    summary["elapsed_seconds"] = time.monotonic() - start
    write_json(output / "summary.json", summary)
    report(runs, summary, output)
    print(json.dumps(summary["level_pass"]), f"report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the pre-specified C3 E4 matrix; incomplete runs cannot pass the gate."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (42, 500, 1337)
MODES = ("sham", "cbcd", "random", "bottom")


def load_run(path):
    config = json.loads((path / "config.json").read_text())
    test = json.loads((path / "test_metrics.json").read_text())
    validations = {r["iteration"]: r for r in map(json.loads, (path / "validation_metrics.jsonl").read_text().splitlines())}
    if max(validations, default=0) != config["args"]["max_iterations"]:
        raise ValueError(f"Unfinished validation sequence: {path}")
    if test.get("split") != "test" or len(test["per_case"]) != 20:
        raise ValueError(f"Unexpected test protocol: {path}")
    return dict(config=config, test=test,
                last5=statistics.mean(validations[i]["dice"] * 100 for i in sorted(validations)[-5:]))


def compare(top, control):
    cases, seed_deltas = [], []
    for seed in SEEDS:
        a, b = top[seed], control[seed]
        if a["config"]["labeled_ids"] != b["config"]["labeled_ids"]:
            raise ValueError("Different labeled case split")
        left = {c["case"]: c["dice"] for c in a["test"]["per_case"]}
        right = {c["case"]: c["dice"] for c in b["test"]["per_case"]}
        if left.keys() != right.keys():
            raise ValueError("Different test case identities")
        cases.append({case: 100 * (left[case] - right[case]) for case in left})
        seed_deltas.append(100 * (a["test"]["dice"] - b["test"]["dice"]))
    values = [statistics.mean(d[case] for d in cases) for case in sorted(cases[0])]
    rng = np.random.default_rng(20261113)
    bootstrap = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return dict(mean_delta=statistics.mean(seed_deltas), seed_deltas=seed_deltas,
                case_bootstrap_95=list(map(float, np.quantile(bootstrap, [.025, .975]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "Results_transfer")
    parser.add_argument("--output", type=Path, default=PROJECT / "summary/transfer_report.md")
    args = parser.parse_args()
    groups, missing = {}, []
    for dataset, labels in (("LA", 8), ("Pancreas", 6)):
        for mode in (*MODES, "C1"):
            groups[dataset, mode] = {}
            for seed in SEEDS:
                base = (PROJECT.parent / "MT_DS_PL_C1/Results_C1_E4" if mode == "C1"
                        else args.root / f"{mode}_e4")
                path = base / f"seed_{seed}" / f"{dataset}_{labels}l"
                needed = ("config.json", "test_metrics.json", "validation_metrics.jsonl")
                if not all((path / name).is_file() for name in needed):
                    missing.append(str(path))
                    continue
                run = load_run(path)
                if mode != "C1":
                    params = run["config"]["args"]
                    required = dict(probe_level="e4", transfer_mode=mode, transfer_topk=16,
                                    transfer_strength=.1, transfer_start=2000, transfer_rampup=2000,
                                    max_iterations=15000, warmup_iterations=2000, boundary_weight=.03,
                                    consistency=.3, base_lr=.01, ema_decay=.99, batch_size=4, labeled_bs=2,
                                    patch_size=[112, 112, 80] if dataset == "LA" else [96, 96, 96])
                    if any(params.get(k) != v for k, v in required.items()):
                        raise ValueError(f"Run differs from the pre-specified protocol: {path}")
                groups[dataset, mode][seed] = run
    lines = ["# C3 最小迁移实验", "", "预设 E4 × 2 数据集 × 3 seed × sham/Top/Random/Bottom；Dice 差值单位为百分点。", ""]
    results = dict(status="incomplete" if missing else "finished", missing_runs=missing, datasets={})
    if missing:
        lines += [f"尚缺 {len(missing)} 次完整实验，未作收益判定。", ""] + [f"- {path}" for path in missing]
    else:
        hashes = [run["config"]["source_sha256"] for (dataset, mode), runs in groups.items()
                  if mode != "C1" for run in runs.values()]
        if any(value != hashes[0] for value in hashes):
            raise ValueError("C3 groups used different source versions")
        for dataset in ("LA", "Pancreas"):
            comparisons = {mode: compare(groups[dataset, "cbcd"], groups[dataset, mode])
                           for mode in ("sham", "C1", "random", "bottom")}
            last5_delta = statistics.mean(groups[dataset, "cbcd"][s]["last5"] - groups[dataset, "sham"][s]["last5"] for s in SEEDS)
            passed = all(comparisons[m]["mean_delta"] > .4 and comparisons[m]["case_bootstrap_95"][0] > 0
                         for m in ("sham", "C1"))
            passed &= all(comparisons[m]["mean_delta"] > 0 for m in ("random", "bottom")) and last5_delta >= 0
            results["datasets"][dataset] = dict(comparisons=comparisons, last5_delta_vs_sham=last5_delta, passed=passed)
            lines += [f"## {dataset}", "", "| 对照 | Top−对照均值 | 病例 bootstrap 95% |", "|---|---:|---|"]
            for mode, value in comparisons.items():
                low, high = value["case_bootstrap_95"]
                lines.append(f"| {mode} | {value['mean_delta']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
            lines += ["", f"最后 5 次验证相对 sham：{last5_delta:+.4f}；本数据集判据：{'通过' if passed else '未通过'}。", ""]
        results["passed"] = all(v["passed"] for v in results["datasets"].values())
        lines += [f"跨数据集判据：{'通过' if results['passed'] else '尚未支持迁移收益'}。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"{results['status']}: {args.output}")


if __name__ == "__main__":
    main()

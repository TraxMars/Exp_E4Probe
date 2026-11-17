"""MT_DS_PL: Mean Teacher + four-scale deep supervision on DuCiSC's VNet with pseudo-label consistency.

L = L_sup + 0.3 * L_pl from iteration 2000 (L_sup before), where both are the four-scale mean of
(CE + Dice) / 2: L_sup to the nearest-neighbour GT, L_pl to each scale's own Teacher argmax.

The GPU is chosen by the caller through CUDA_VISIBLE_DEVICES (scripts/launch.py sets it).
"""
import argparse
import copy
import csv
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from boundary_probe import LEVEL_CHANNELS, BoundaryProbe, boundary_loss, encoder_parameters, gradient_ratio, \
    label_to_boundary
from boundary_transfer import BoundaryTransfer, contribution_scores, ramped_strength
from data import SPECS, VolumeDataset, TwoStreamBatchSampler
from diagnostics import boundary_diagnostics, gradient_diagnostics, labeled_band_ce
from evaluate import ducisc_line, evaluate
from networks import VNetDS
from objectives import pseudo_label_consistency, supervised_loss, update_ema
from provenance import snapshot, source_manifest


PROJECT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(SPECS), required=True)
    parser.add_argument("--data_root", type=Path, default=PROJECT.parents[1] / "Dataset")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--labelnum", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    for name, default in (("seed", 42), ("batch_size", 4), ("labeled_bs", 2), ("max_iterations", 15000),
                          ("warmup_iterations", 2000), ("base_lr", 0.01), ("ema_decay", 0.99),
                          ("consistency", 0.3), ("num_workers", 2), ("cpu_threads", 4), ("log_interval", 10),
                          ("val_interval", 200), ("val_start", 800), ("checkpoint_interval", 1000)):
        parser.add_argument(f"--{name}", type=type(default), default=default)
    # Logging-only diagnostics (diagnostics.csv) every N iterations; 0 disables them.
    parser.add_argument("--diag_interval", type=int, default=50)
    # Keep the training volumes in memory (sampling and augmentation are unchanged).
    parser.add_argument("--cache_volumes", action=argparse.BooleanOptionalAction, default=True)
    # Resuming restores the configuration stored in the checkpoint; other arguments are ignored.
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--patch_size", nargs=3, type=int)
    # C1 encoder boundary probe (MT-DuCiSC BoundaryProbeW003); "none" is the unchanged MT_DS_PL baseline.
    parser.add_argument("--probe_level", choices=["none", *LEVEL_CHANNELS], default="none")
    parser.add_argument("--boundary_weight", type=float, default=0.03)
    # Logging-only ||grad(λ·L_b)|| / ||grad(L_sup)|| over the encoder up to the probed level; 0 disables it.
    parser.add_argument("--ratio_interval", type=int, default=200)
    parser.add_argument("--transfer_mode", choices=["none", "sham", "cbcd", "random", "bottom"], default="none")
    parser.add_argument("--transfer_topk", type=int, default=16)
    parser.add_argument("--transfer_strength", type=float, default=0.10)
    parser.add_argument("--transfer_start", type=int, default=2000)
    parser.add_argument("--transfer_rampup", type=int, default=2000)
    args = parser.parse_args()
    if not 0 < args.labeled_bs < args.batch_size:
        parser.error("Require 0 < labeled_bs < batch_size")
    if args.transfer_mode != "none":
        if args.probe_level == "none":
            parser.error("Transfer requires an E3/E4 boundary probe")
        if not 0 < args.transfer_topk < LEVEL_CHANNELS[args.probe_level]:
            parser.error("Require 0 < transfer_topk < feature channel count")
        if args.transfer_start < args.warmup_iterations or args.transfer_rampup < 0 or args.transfer_strength < 0:
            parser.error("Transfer must start after warmup; rampup and strength must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this session; use CPU only for checks")
    return args


def git_commit():
    result = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"],
                            cwd=PROJECT, capture_output=True, text=True)
    return result.stdout.strip() or None


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train(args):
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"].get("source_sha256") != source_manifest(PROJECT):
            raise ValueError("Resume requires a C3 checkpoint from identical source files")
        args = argparse.Namespace(**dict(checkpoint["config"]["args"], resume=args.resume))
        args.data_root, args.output_dir = Path(args.data_root), Path(args.output_dir)
    spec = SPECS[args.dataset]
    args.data_root = args.data_root.resolve()
    root = args.data_root / spec["directory"]
    args.labelnum = args.labelnum if args.labelnum is not None else spec["labels"]
    args.patch_size = list(args.patch_size or spec["patch"])
    dataset = VolumeDataset(root, args.dataset, args.patch_size, cache=args.cache_volumes)
    if not args.labeled_bs <= args.labelnum <= len(dataset) - (args.batch_size - args.labeled_bs):
        raise ValueError("Invalid labeled count for this dataset/batch size")
    sampler = TwoStreamBatchSampler(list(range(args.labelnum)), list(range(args.labelnum, len(dataset))),
                                    args.batch_size, args.labeled_bs)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    student = VNetDS(n_channels=1, n_classes=2, has_dropout=True).to(args.device)
    teacher = copy.deepcopy(student).requires_grad_(False)
    edge_probe = None
    transfer = None
    if args.probe_level != "none":
        # Initialized (on the CPU) from its own seed inside fork_rng, so the global CPU RNG (DataLoader base seed,
        # sampling) and the CUDA RNG (dropout) follow exactly the baseline's streams. torch.manual_seed would also
        # reseed CUDA, which fork_rng(devices=[]) does not restore; seed only the CPU generator.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(args.seed + 1000003)
            edge_probe = BoundaryProbe(LEVEL_CHANNELS[args.probe_level]).to(args.device)
    if args.transfer_mode != "none":
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(args.seed + 2000003)
            transfer = BoundaryTransfer(LEVEL_CHANNELS[args.probe_level], args.transfer_topk,
                                        args.transfer_mode, seed=args.seed + 3000003).to(args.device)
    # Same SGD for the probe as in MT-DuCiSC (one optimizer, same lr / momentum / weight decay).
    optimizer = torch.optim.SGD(list(student.parameters()) + (list(edge_probe.parameters()) if edge_probe else [])
                                + (list(transfer.parameters()) if transfer else []),
                                lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    config = dict(args=dict({key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()}, resume=None),
                  git_commit=git_commit(), source_sha256=source_manifest(PROJECT),
                  base_commit="6add5ca23a7b95037ac989215d92bcbb50042d4d", labeled_ids=dataset.ids[:args.labelnum],
                  unlabeled_ids=dataset.ids[args.labelnum:])
    iteration, best_dice = 0, -1.0
    if checkpoint is not None:
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        if edge_probe is not None:
            edge_probe.load_state_dict(checkpoint["probe"])
        if transfer is not None:
            transfer.load_state_dict(checkpoint["transfer"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iteration, best_dice = checkpoint["iteration"], checkpoint["best_dice"]
        if (args.output_dir / "best_metrics.json").is_file():
            # best_model.pth may be newer than the last periodic checkpoint.
            best_dice = max(best_dice, json.loads((args.output_dir / "best_metrics.json").read_text())["dice"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if args.device == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        snapshot(PROJECT, args.output_dir, config["source_sha256"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(args.output_dir / "log.txt"), logging.StreamHandler(sys.stdout)])
    logging.info("config: %s", config)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                        pin_memory=args.device == "cuda", worker_init_fn=worker_init)

    def save_checkpoint():
        torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(),
                        probe=edge_probe.state_dict() if edge_probe is not None else None,
                        transfer=transfer.state_dict() if transfer is not None else None,
                        optimizer=optimizer.state_dict(), iteration=iteration, best_dice=best_dice,
                        config=config, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None),
                   args.output_dir / "last_checkpoint.pth.tmp")
        (args.output_dir / "last_checkpoint.pth.tmp").replace(args.output_dir / "last_checkpoint.pth")

    start, data_wait = time.monotonic(), 0.0
    fields = ["iteration", "loss_supervised", "ce", "dice", "loss_pl", "pl_ce", "pl_dice",
              "consistency_weight", "loss_total", "gradient_norm", "lr", "peak_memory_mib", "elapsed_seconds",
              "data_wait_seconds"]
    with (args.output_dir / "training_metrics.csv").open("a", newline="", buffering=1) as handle, \
            (args.output_dir / "diagnostics.csv").open("a", newline="", buffering=1) as diagnostics_handle:
        metrics, diagnostics = csv.DictWriter(handle, fieldnames=fields), None
        if handle.tell() == 0:
            metrics.writeheader()
        boundary_handle = boundary_metrics = None
        transfer_handle = transfer_metrics = None
        if transfer is not None:
            transfer_handle = (args.output_dir / "transfer_metrics.csv").open("a", newline="", buffering=1)
            transfer_metrics = csv.DictWriter(transfer_handle, fieldnames=[
                "iteration", "donor_mode", "strength", "mask_ratio", "gate_mean", "delta_norm",
                "residual_ratio", "valid_donors"])
            if transfer_handle.tell() == 0:
                transfer_metrics.writeheader()
        if edge_probe is not None:
            # Separate file, so training_metrics.csv keeps the baseline's columns.
            boundary_handle = (args.output_dir / "boundary_metrics.csv").open("a", newline="", buffering=1)
            boundary_metrics = csv.DictWriter(boundary_handle, fieldnames=[
                "iteration", "loss_boundary", "boundary_bce", "boundary_dice", "boundary_fraction",
                "grad_sup", "grad_boundary", "grad_ratio", "grad_cos"])
            if boundary_handle.tell() == 0:
                boundary_metrics.writeheader()
        ratio = {}
        while iteration < args.max_iterations:
            student.train()
            teacher.train()
            if edge_probe is not None:
                edge_probe.train()
            fetch = time.monotonic()
            for batch in loader:
                data_wait += time.monotonic() - fetch
                images = batch["image"].to(args.device, non_blocking=True)
                # Only labeled GT enters the objective.
                labels = batch["label"][:args.labeled_bs].to(args.device, non_blocking=True)
                if edge_probe is None:
                    supervised_outputs = student(images[:args.labeled_bs], deepsup=True)
                else:
                    supervised_outputs, features = student(images[:args.labeled_bs], deepsup=True, features=True)
                    boundary_target = label_to_boundary(labels)
                    loss_boundary, boundary_bce, boundary_dice = boundary_loss(
                        edge_probe(features[args.probe_level], output_size=labels.shape[1:]), boundary_target)
                    probe_feature = features[args.probe_level]
                    del features
                loss_sup, ce, dice = supervised_loss(supervised_outputs, labels)
                strength = ramped_strength(iteration, args.transfer_start, args.transfer_rampup,
                                            args.transfer_strength) if transfer is not None else 0.0
                transfer_stats = dict(strength=0.0, mask_ratio=0.0, gate_mean=0.0,
                                      delta_norm=0.0, residual_ratio=0.0, valid_donors=0)
                if transfer is not None and strength > 0:
                    # All active modes (including sham) share this forward order and dropout schedule.
                    scores = contribution_scores(loss_boundary, probe_feature)
                    encoded = list(student.forward_features(images[args.labeled_bs:]))
                    with torch.no_grad():
                        teacher_outputs = teacher(images[args.labeled_bs:], deepsup=True)
                        probability = teacher_outputs[-1].softmax(dim=1)
                        entropy = -(probability * probability.clamp_min(1e-6).log()).sum(dim=1, keepdim=True) \
                            / np.log(probability.shape[1])
                        pseudo_boundary = label_to_boundary(teacher_outputs[-1].argmax(dim=1))
                    level_index = {"e3": 2, "e4": 3}[args.probe_level]
                    encoded[level_index], transfer_stats = transfer(
                        probe_feature, scores, boundary_target, encoded[level_index], pseudo_boundary, entropy, strength)
                    unlabeled_outputs = student.decode_features(encoded, deepsup=True)
                    del encoded, scores
                else:
                    # Disabled/zero strength and every warmup step retain the C1 forward order exactly.
                    unlabeled_outputs = student(images[args.labeled_bs:], deepsup=True)
                    with torch.no_grad():
                        teacher_outputs = teacher(images[args.labeled_bs:], deepsup=True)
                loss_pl, pl_ce, pl_dice = pseudo_label_consistency(unlabeled_outputs, teacher_outputs)
                weight = args.consistency if iteration >= args.warmup_iterations else 0.0
                loss = loss_sup
                if weight > 0:
                    loss = loss + weight * loss_pl
                if edge_probe is not None:
                    # As in MT-DuCiSC: constant weight from the first iteration.
                    loss = loss + args.boundary_weight * loss_boundary
                    if not torch.isfinite(loss_boundary):
                        raise FloatingPointError("Nonfinite boundary loss")
                if not torch.isfinite(torch.stack([loss_sup, loss_pl, loss])).all():
                    raise FloatingPointError("Nonfinite supervised/pseudo-label/total loss")
                optimizer.zero_grad(set_to_none=True)
                if args.diag_interval and (iteration + 1) % args.diag_interval == 0:
                    # The hidden GT of the unlabeled images is read only here, after the objective is built.
                    # It enters only L_true, an oracle that is differentiated by autograd.grad for the
                    # logged floats and then dropped, and a no_grad boundary function.
                    hidden = batch["label"][args.labeled_bs:].to(args.device)
                    loss_true = supervised_loss(unlabeled_outputs, hidden)[0]
                    probe = dict(iteration=iteration + 1, **gradient_diagnostics(
                        loss_sup, args.consistency * loss_pl,
                        dict(b7=student.block_seven.conv[-3].weight, b3=student.block_three.conv[-3].weight),
                        loss_true))
                    del loss_true
                    if any(parameter.grad is not None for parameter in student.parameters()):
                        raise RuntimeError("Gradient diagnostics wrote parameter gradients")
                    probe.update(boundary_diagnostics(unlabeled_outputs[-1], teacher_outputs[-1], hidden))
                    probe.update(labeled_band_ce(supervised_outputs[-1], labels))
                    if not all(isinstance(value, (int, float)) for value in probe.values()):
                        raise RuntimeError("Diagnostics must return plain numbers")
                    if diagnostics is None:
                        diagnostics = csv.DictWriter(diagnostics_handle, fieldnames=list(probe))
                        if diagnostics_handle.tell() == 0:
                            diagnostics.writeheader()
                    diagnostics.writerow(probe)
                if boundary_metrics is not None and args.ratio_interval and (iteration + 1) % args.ratio_interval == 0:
                    ratio = gradient_ratio(loss_sup, args.boundary_weight * loss_boundary,
                                           encoder_parameters(student, args.probe_level))
                    if any(parameter.grad is not None for parameter in student.parameters()):
                        raise RuntimeError("Gradient ratio logging wrote parameter gradients")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float("inf"),
                                                               error_if_nonfinite=True)
                optimizer.step()
                update_ema(student, teacher, args.ema_decay, iteration)
                iteration += 1
                if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.max_iterations:
                    peak = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0
                    row = dict(iteration=iteration, loss_supervised=loss_sup.item(), ce=ce.item(), dice=dice.item(),
                               loss_pl=loss_pl.item(), pl_ce=pl_ce.item(), pl_dice=pl_dice.item(),
                               consistency_weight=weight, loss_total=loss.item(),
                               gradient_norm=gradient_norm.item(), lr=optimizer.param_groups[0]["lr"],
                               peak_memory_mib=peak, elapsed_seconds=time.monotonic() - start,
                               data_wait_seconds=data_wait)
                    metrics.writerow(row)
                    if transfer_metrics is not None:
                        transfer_metrics.writerow(dict(iteration=iteration, donor_mode=args.transfer_mode,
                                                       **transfer_stats))
                    logging.info("iteration %d: sup=%.6f pl=%.6f weight=%.3f grad=%.4f memory=%.0fMiB", iteration,
                                 row["loss_supervised"], row["loss_pl"], weight, row["gradient_norm"], peak)
                    if boundary_metrics is not None:
                        boundary_metrics.writerow(dict(
                            iteration=iteration, loss_boundary=loss_boundary.item(), boundary_bce=boundary_bce.item(),
                            boundary_dice=boundary_dice.item(), boundary_fraction=boundary_target.mean().item(),
                            **{key: ratio.get(key, float("nan"))
                               for key in ("grad_sup", "grad_boundary", "grad_ratio", "grad_cos")}))
                        ratio = {}
                if args.val_interval and iteration >= args.val_start and iteration % args.val_interval == 0:
                    result = dict(evaluate(student, root, args.dataset, args.patch_size, args.device), iteration=iteration)
                    with (args.output_dir / "validation_metrics.jsonl").open("a") as validation_log:
                        validation_log.write(json.dumps(result, allow_nan=False) + "\n")
                    logging.info("validation iteration %d: Dice=%.6f", iteration, result["dice"])
                    if result["dice"] > best_dice:
                        best_dice = result["dice"]
                        torch.save(student.state_dict(), args.output_dir / "best_model.pth")
                        (args.output_dir / "best_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
                if iteration == 1 or iteration % args.checkpoint_interval == 0 or iteration == args.max_iterations:
                    save_checkpoint()
                if iteration >= args.max_iterations:
                    break
                fetch = time.monotonic()
    if boundary_handle is not None:
        boundary_handle.close()
    if transfer_handle is not None:
        transfer_handle.close()
    if (args.output_dir / "best_model.pth").is_file():
        student.load_state_dict(torch.load(args.output_dir / "best_model.pth", map_location=args.device,
                                           weights_only=True))
        # Main result without post-processing (DuCiSC's default nms=0); "nms1" adds getLargestCC.
        result = evaluate(student, root, args.dataset, args.patch_size, args.device, split="test", nms1=True)
        (args.output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        logging.info("test nms0: %s | nms1: %s", ducisc_line(result), ducisc_line(result["nms1"]))
    logging.info("finished at iteration %d, best validation Dice %.6f", iteration, best_dice)


if __name__ == "__main__":
    train(arguments())

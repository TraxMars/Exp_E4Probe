"""Check the port against C1 and the old transfer, then exercise the real training/checkpoint path."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from boundary_transfer import BoundaryTransfer, ramped_strength
from networks import VNetDS


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check_network():
    original = module("c1_vnet", PROJECT.parent / "MT_DS_PL_C1/code/networks/vnet_ds.py").VNetDS
    for training in (False, True):
        torch.manual_seed(42)
        old = original(has_dropout=True).train(training)
        torch.manual_seed(42)
        new = VNetDS(has_dropout=True).train(training)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
        image = torch.randn(2, 1, 32, 32, 32)
        state = torch.get_rng_state()
        expected = old(image, deepsup=True)
        torch.set_rng_state(state)
        actual = new(image, deepsup=True)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        if training:
            sum(o.square().mean() for o in expected).backward()
            sum(o.square().mean() for o in actual).backward()
            for left, right in zip(old.parameters(), new.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    print("PASS: C1/C3 initialization, all four outputs and backward gradients match bitwise in train/eval")


def check_transfer():
    old_class = module("old_boundary", ROOT / "MT+DuCiSC+BGCE/MT-DuCiSC/code/networks/boundary_probe.py").BoundaryGuidedFeatureTransfer
    torch.manual_seed(7)
    donor = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
    receiver = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
    scores = torch.rand(2, 16)
    band = torch.zeros(2, 1, 16, 16, 16)
    band[:, :, 5:11, 5:11, 5:11] = 1
    entropy = torch.rand_like(band)
    for mode in ("cbcd", "bottom"):
        new = BoundaryTransfer(16, 4, mode)
        old = old_class(16, 4, donor_mode=mode)
        old.load_state_dict({k: v for k, v in new.state_dict().items() if k != "_extra_state"})
        expected, _ = old(donor, scores, band, receiver, band, entropy, .1)
        actual, _ = new(donor, scores, band, receiver, band, entropy, .1)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        actual.square().mean().backward()
        assert donor.grad is None and receiver.grad is not None and receiver.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
    print("PASS: global Top/Bottom transfer matches first-version values; donor detached, receiver/module gradients finite")
    random = BoundaryTransfer(16, 4, "random", seed=8)
    global_rng = torch.get_rng_state().clone()
    saved = copy.deepcopy(random.state_dict())
    expected, _ = random(donor, scores, band, receiver, band, entropy, .1)
    random.load_state_dict(saved)
    actual, _ = random(donor, scores, band, receiver, band, entropy, .1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(global_rng, torch.get_rng_state())
    for masks, uncertainty, strength in [(torch.zeros_like(band), entropy, .1), (band, torch.zeros_like(entropy), .1), (band, entropy, 0)]:
        actual, _ = random(donor, scores, masks, receiver, band, uncertainty, strength)
        assert actual is receiver
    sham = BoundaryTransfer(16, 4, "sham")
    assert sham(donor, scores, band, receiver, band, entropy, .1)[0] is receiver
    assert ramped_strength(1999, 2000, 2000, .1) == 0
    assert ramped_strength(2000, 2000, 2000, .1) == .1 / 2000
    assert ramped_strength(3999, 2000, 2000, .1) == .1
    print("PASS: random routing checkpoint restores exactly, global RNG untouched, empty/zero/sham paths are identities")


def make_data(root, dataset, invert_hidden=False):
    folder = root / dataset
    folder.mkdir(parents=True)
    names = [f"case_{i}" for i in range(4)]
    (folder / "train.list").write_text("\n".join(names) + "\n")
    rng = np.random.default_rng(42)
    for i, name in enumerate(names):
        image = rng.normal(size=(32, 32, 32)).astype(np.float32)
        label = np.zeros(image.shape, dtype=np.int64)
        label[6 + i:22 + i, 7:23, 8:24] = 1
        if invert_hidden and i >= 2:
            label = 1 - label
        path = (folder / "2018LA_Seg_TrainingSet" / name / "mri_norm2.h5" if dataset == "LA"
                else folder / "Pancreas_h5" / f"{name}_norm.h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle["image"], handle["label"] = image, label


def integration():
    with tempfile.TemporaryDirectory(prefix="e4probe_c3_") as temporary:
        tmp = Path(temporary)
        make_data(tmp / "data", "LA")
        make_data(tmp / "data", "Pancreas")
        make_data(tmp / "inverted", "LA", invert_hidden=True)
        common = ["--device", "cpu", "--cpu_threads", "2", "--labelnum", "2", "--num_workers", "0",
                  "--patch_size", "32", "32", "32", "--max_iterations", "4", "--warmup_iterations", "1",
                  "--val_interval", "0", "--diag_interval", "1", "--ratio_interval", "2", "--checkpoint_interval", "2"]
        states = {}
        def run(name, project=PROJECT, mode=None, strength=None, data="data", dataset="LA"):
            output = tmp / "runs" / name
            cmd = [sys.executable, str(project / "code/train.py"), "--dataset", dataset, "--probe_level", "e4",
                   "--data_root", str(tmp / data), "--output_dir", str(output), *common]
            if mode:
                cmd += ["--transfer_mode", mode, "--transfer_start", "1", "--transfer_rampup", "2"]
            if strength is not None:
                cmd += ["--transfer_strength", str(strength)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(f"{name} failed:\n{result.stdout}\n{result.stderr}")
            checkpoint = torch.load(output / "last_checkpoint.pth", map_location="cpu", weights_only=False)
            assert checkpoint["iteration"] == 4
            states[name] = checkpoint
            return output
        run("c1", PROJECT.parent / "MT_DS_PL_C1")
        run("none", mode="none")
        run("zero", mode="cbcd", strength=0)
        for name in ("none", "zero"):
            for key in ("student", "teacher", "probe"):
                for param, value in states["c1"][key].items():
                    torch.testing.assert_close(states[name][key][param], value, rtol=0, atol=0)
        print("PASS: real CLI 4-step none/zero-strength trajectories exactly reproduce C1, including Teacher/probe BN")
        run("sham", mode="sham")
        top_output = run("cbcd", mode="cbcd")
        run("random", mode="random")
        run("bottom", mode="bottom")
        run("inverted", mode="cbcd", data="inverted")
        for key in ("student", "teacher", "probe", "transfer"):
            for param, value in states["cbcd"][key].items():
                torch.testing.assert_close(states["inverted"][key][param], value, rtol=0, atol=0)
        assert any(not torch.equal(value, states["sham"]["student"][key]) for key, value in states["cbcd"]["student"].items())
        resumed = subprocess.run([sys.executable, str(PROJECT / "code/train.py"), "--dataset", "LA",
                                 "--output_dir", str(top_output), "--device", "cpu", "--resume",
                                 str(top_output / "last_checkpoint.pth")], capture_output=True, text=True)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        run("pancreas", mode="cbcd", dataset="Pancreas")
        print("PASS: sham/Top/Random/Bottom CLI train, active transfer changes Student, LA/Pancreas smoke and checkpoint reload")
        print("PASS: inverting all hidden unlabeled GT leaves Student/Teacher/probe/transfer state bitwise unchanged")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_network()
    check_transfer()
    integration()

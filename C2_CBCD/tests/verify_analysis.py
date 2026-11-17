"""Independent formula/masking checks and preservation of model state and RNG."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "code"), str(PROJECT.parent / "MT_DS_PL_C1/code")]
from boundary_probe import BoundaryProbe, boundary_loss, label_to_boundary
from channel_analysis import channel_scores, encoder_feature, occlusion_deltas, sample_patch
from networks import VNetDS


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    student = VNetDS(has_dropout=True).eval()
    image = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        _, features = student(image, deepsup=True, features=True)
        for level in ("e3", "e4"):
            torch.testing.assert_close(encoder_feature(student, image, level), features[level], rtol=0, atol=0)
    print("PASS: encoder-only feature equals existing full-forward E3/E4 exactly")

    probe = BoundaryProbe(8, hidden_channels=4).train()
    probe.head[1].eval()  # Preserve mixed modes as well as BN running statistics.
    feature = torch.randn(1, 8, 4, 4, 4)
    labels = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    labels[:, 2:6, 2:6, 2:6] = 1
    target = label_to_boundary(labels)
    before = copy.deepcopy(probe.state_dict())
    modes = [m.training for m in probe.modules()]
    rng = torch.get_rng_state().clone()
    scores = channel_scores(probe, feature, target)
    reference_probe = copy.deepcopy(probe).eval()
    leaf = feature.clone().requires_grad_()
    loss = boundary_loss(reference_probe(leaf, target.shape[2:]), target)[0]
    gradient = torch.autograd.grad(loss, leaf)[0]
    expected = (gradient * leaf).abs().mean((2, 3, 4))
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    result = occlusion_deltas(probe, feature, target, scores, 2, 4, torch.Generator().manual_seed(123))
    with torch.no_grad():
        base = boundary_loss(reference_probe(feature, target.shape[2:]), target)[0]
        def delta(indices):
            masked = feature.clone()
            masked[0, indices[0]] = 0
            return (boundary_loss(reference_probe(masked, target.shape[2:]), target)[0] - base).item()
        assert result["top_delta"] == delta(scores.topk(2, dim=1).indices)
        assert result["bottom_delta"] == delta(scores.topk(2, dim=1, largest=False).indices)
        generator = torch.Generator().manual_seed(123)
        random = [delta(torch.randperm(8, generator=generator)[:2][None]) for _ in range(4)]
        assert result["random_delta"] == float(np.mean(random))
    for name, value in probe.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert modes == [m.training for m in probe.modules()]
    assert all(p.grad is None for p in probe.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    assert not scores.requires_grad
    print("PASS: CBCD and Top/Bottom/Random interventions match independent calculations")
    print("PASS: parameter gradients, mixed train/eval modes, BN state and global RNG untouched")

    data = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
    labels_np = (data % 3 == 0).astype(np.int64)
    before_data, before_label = data.copy(), labels_np.copy()
    first = sample_patch(data, labels_np, (16, 16, 16), True, np.random.RandomState(4))
    second = sample_patch(data, labels_np, (16, 16, 16), True, np.random.RandomState(4))
    assert first[2] == second[2] and first[0].shape == (1, 1, 16, 16, 16)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    np.testing.assert_array_equal(data, before_data)
    np.testing.assert_array_equal(labels_np, before_label)
    assert not label_to_boundary(torch.zeros_like(labels)).any()
    print("PASS: deterministic local crops, source arrays preserved, empty boundary recognized")

    # Closed-form channel scoring on a linear objective catches averaging/absolute-value mistakes.
    class LinearProbe(nn.Module):
        def forward(self, x, output_size):
            return F.interpolate(x.sum(1, keepdim=True), size=output_size, mode="trilinear", align_corners=False)
    controlled = torch.zeros(1, 8, 4, 4, 4)
    controlled[:, 0] = 1
    measured = channel_scores(LinearProbe(), controlled, target)
    assert measured[0, 0] > 0 and torch.equal(measured[0, 1:], torch.zeros(7))
    print("PASS: only the active channel receives a nonzero CBCD score")


if __name__ == "__main__":
    main()

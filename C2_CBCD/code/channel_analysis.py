"""Frozen-checkpoint CBCD and equal-budget channel occlusion; no optimizer or hidden GT."""
from contextlib import contextmanager

import numpy as np
import torch

from boundary_probe import boundary_loss


@contextmanager
def evaluation(module):
    """Restore each submodule's mode, including a deliberately mixed BN/dropout setup."""
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, mode in modes:
            child.training = mode


def sample_patch(image, label, patch_size, rotation, rng):
    """C1 augmentation/padding/crop rules, using an explicit local RandomState."""
    k, axis = (int(rng.randint(0, 4)), int(rng.randint(0, 2))) if rotation else (0, None)
    if rotation:
        image = np.flip(np.rot90(image, k), axis=axis).copy()
        label = np.flip(np.rot90(label, k), axis=axis).copy()
    pads = [0, 0, 0]
    if any(size <= patch for size, patch in zip(label.shape, patch_size)):
        pads = [max((patch - size) // 2 + 3, 0) for size, patch in zip(label.shape, patch_size)]
        image = np.pad(image, [(pad, pad) for pad in pads])
        label = np.pad(label, [(pad, pad) for pad in pads])
    starts = [int(rng.randint(0, size - patch + 1)) for size, patch in zip(label.shape, patch_size)]
    crop = tuple(slice(start, start + patch) for start, patch in zip(starts, patch_size))
    return (torch.from_numpy(np.ascontiguousarray(image[crop][None, None])),
            torch.from_numpy(np.ascontiguousarray(label[crop][None])),
            dict(rotation_k=k, flip_axis=axis, padding=pads, starts=starts, patch_size=list(patch_size)))


@torch.no_grad()
def encoder_feature(student, image, level):
    """Exactly the existing encoder up to the probe; decoder/dropout are unnecessary here."""
    if level not in ("e3", "e4"):
        raise ValueError(f"Unsupported probe level: {level}")
    x = student.block_one(image)
    x = student.block_two(student.block_one_dw(x))
    x = student.block_three(student.block_two_dw(x))
    if level == "e4":
        x = student.block_four(student.block_three_dw(x))
    return x


def channel_scores(probe, feature, target):
    """Magnitude scores are sensitivities, not signed evidence of a useful segmentation channel."""
    if feature.ndim != 5 or target.ndim != 5 or feature.shape[0] != target.shape[0]:
        raise ValueError("Expected matching 5-D feature/target batches")
    with evaluation(probe), torch.enable_grad():
        leaf = feature.detach().requires_grad_(True)
        loss = boundary_loss(probe(leaf, target.shape[2:]), target)[0]
        gradient = torch.autograd.grad(loss, leaf, create_graph=False)[0]
        scores = (gradient * leaf).abs().mean(dim=(2, 3, 4)).detach()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite CBCD scores")
    return scores


@torch.no_grad()
def occlusion_deltas(probe, feature, target, scores, k, random_repeats, generator):
    """Same sample, target and frozen probe for all interventions; return loss differences."""
    if feature.shape[:2] != scores.shape or not 0 < k < feature.shape[1] or random_repeats < 1:
        raise ValueError("Invalid score shape, occlusion budget or random repeat count")
    with evaluation(probe):
        reference = boundary_loss(probe(feature, target.shape[2:]), target)[0]

        def difference(indices):
            mask = torch.ones_like(scores).scatter_(1, indices.to(scores.device), 0)
            logits = probe(feature * mask[..., None, None, None], target.shape[2:])
            return (boundary_loss(logits, target)[0] - reference).item()

        top = difference(scores.topk(k, dim=1).indices)
        bottom = difference(scores.topk(k, dim=1, largest=False).indices)
        random = [difference(torch.stack([
            torch.randperm(feature.shape[1], generator=generator)[:k] for _ in range(feature.shape[0])
        ])) for _ in range(random_repeats)]
    if not np.isfinite([reference.item(), top, bottom, *random]).all():
        raise FloatingPointError("Nonfinite occlusion loss")
    return dict(reference_loss=reference.item(), top_delta=top, bottom_delta=bottom,
                random_delta=float(np.mean(random)), random_std=float(np.std(random)),
                top_minus_random=top - float(np.mean(random)), top_minus_bottom=top - bottom)

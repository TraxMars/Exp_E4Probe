"""C1: the encoder boundary probe of MT-DuCiSC (BoundaryProbeW003), trained jointly with the Student.

Ported unchanged from MT+DuCiSC+BGCE/MT-DuCiSC/results/LA_BoundaryProbeW003_8_labeled_0.01_0.99_seed1337/
teacher_0.3_none/code (utils/boundary.py, networks/boundary_probe.py): a Conv3-BN-ReLU-Conv1 head on a labeled
encoder feature (E4 = x4 originally; E3 = x3 is the added level), its logits trilinearly upsampled to the patch
and supervised with the GT boundary (3x3x3 neighbourhood max != min) by batch-balanced BCE + soft Dice.
Only labeled GT enters it; inference uses the plain VNetDS.
"""
import torch
from torch import nn
import torch.nn.functional as F


LEVEL_CHANNELS = {"e3": 64, "e4": 128}


def label_to_boundary(label, radius=1):
    """Voxels whose (2·radius+1)³ neighbourhood holds more than one label. Max-pool pads with -inf, so
    patch borders do not count as object borders."""
    if label.ndim != 4:
        raise ValueError(f"Expected [B, D, H, W] labels, got {tuple(label.shape)}")
    kernel = 2 * radius + 1
    x = label.float().unsqueeze(1)
    local_max = F.max_pool3d(x, kernel_size=kernel, stride=1, padding=radius)
    local_min = -F.max_pool3d(-x, kernel_size=kernel, stride=1, padding=radius)
    return (local_max != local_min).float()


class BoundaryProbe(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature, output_size):
        logits = self.head(feature)
        return F.interpolate(logits, size=output_size, mode="trilinear", align_corners=False)


def boundary_loss(logits, target, eps=1e-6):
    """Balanced BCE (pos_weight = negatives/positives over the batch, clamped to [1, 100]) plus soft Dice."""
    target = target.float()
    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = (negative / (positive + eps)).clamp(min=1.0, max=100.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3, 4))
    denominator = prob.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()
    return bce + dice, bce.detach(), dice.detach()


def encoder_parameters(student, level):
    """Student parameters the probe gradient reaches: blocks up to and including the probed level."""
    names = ["block_one", "block_one_dw", "block_two", "block_two_dw", "block_three"]
    if level == "e4":
        names += ["block_three_dw", "block_four"]
    return [parameter for name in names for parameter in getattr(student, name).parameters()]


def gradient_ratio(loss_sup, weighted_boundary, parameters):
    """||grad(λ·L_b)|| / ||grad(L_sup)|| and their cosine over the shared encoder parameters.
    Logging only: autograd.grad leaves every .grad untouched."""
    g_sup = torch.autograd.grad(loss_sup, parameters, retain_graph=True)
    g_b = torch.autograd.grad(weighted_boundary, parameters, retain_graph=True)
    flat_sup = torch.cat([g.flatten() for g in g_sup])
    flat_b = torch.cat([g.flatten() for g in g_b])
    norm_sup, norm_b = flat_sup.norm(), flat_b.norm()
    cosine = (flat_sup @ flat_b) / (norm_sup * norm_b).clamp_min(1e-30)
    return dict(grad_sup=norm_sup.item(), grad_boundary=norm_b.item(),
                grad_ratio=(norm_b / norm_sup.clamp_min(1e-30)).item(), grad_cos=cosine.item())

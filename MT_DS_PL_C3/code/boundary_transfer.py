"""C3 minimal global boundary-statistic transfer; no BDD or donor matching yet."""
import torch
from torch import nn
import torch.nn.functional as F


def contribution_scores(loss_boundary, feature):
    gradient = torch.autograd.grad(loss_boundary, feature, retain_graph=True, create_graph=False)[0]
    return (gradient * feature).abs().mean(dim=(2, 3, 4)).detach()


def ramped_strength(iteration, start, rampup, maximum):
    if iteration < start or maximum == 0:
        return 0.0
    return maximum * min(1.0, (iteration - start + 1) / rampup) if rampup else maximum


class BoundaryTransfer(nn.Module):
    def __init__(self, channels, topk=16, donor_mode="cbcd", seed=0, boundary_dilation=2):
        super().__init__()
        if not 0 < topk < channels or donor_mode not in ("cbcd", "random", "bottom", "sham"):
            raise ValueError("Invalid channel budget or donor mode")
        self.channels, self.topk = channels, topk
        self.donor_mode, self.boundary_dilation = donor_mode, boundary_dilation
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.donor_to_delta = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels),
                                            nn.ReLU(inplace=True), nn.Linear(channels, channels))
        self.gate = nn.Sequential(nn.LayerNorm(2 * channels), nn.Linear(2 * channels, channels), nn.Sigmoid())

    def get_extra_state(self):
        return self.generator.get_state()

    def set_extra_state(self, state):
        self.generator.set_state(state.cpu())

    def _pool(self, feature, boundary):
        if self.boundary_dilation:
            r = self.boundary_dilation
            boundary = F.max_pool3d(boundary.float(), 2 * r + 1, stride=1, padding=r)
        mask = F.interpolate(boundary.float(), size=feature.shape[2:], mode="nearest")
        count = mask.sum(dim=(2, 3, 4))
        return (feature * mask).sum(dim=(2, 3, 4)) / count.clamp_min(1), mask, count[:, 0] > 0

    def forward(self, donor_feature, scores, donor_boundary, receiver_feature, pseudo_boundary, entropy, strength):
        if scores.shape != donor_feature.shape[:2] or donor_feature.shape[1] != self.channels \
                or receiver_feature.shape[1] != self.channels:
            raise ValueError("Mismatched transfer features/scores")
        stats = dict(strength=float(strength), mask_ratio=0.0, gate_mean=0.0,
                     delta_norm=0.0, residual_ratio=0.0, valid_donors=0)
        if strength == 0 or self.donor_mode == "sham":
            return receiver_feature, stats
        donor_values, _, valid = self._pool(donor_feature.detach(), donor_boundary.detach())
        stats["valid_donors"] = valid.sum().item()
        if not valid.any():
            return receiver_feature, stats
        if self.donor_mode == "random":
            indices = torch.stack([torch.randperm(self.channels, generator=self.generator)[:self.topk]
                                   for _ in range(donor_values.shape[0])]).to(scores.device)
        else:
            indices = scores.topk(self.topk, dim=1, largest=self.donor_mode == "cbcd").indices
        selected = torch.zeros_like(scores).scatter_(1, indices, 1)
        token = (donor_values * selected)[valid].mean(dim=0).detach()
        receiver_values, band, _ = self._pool(receiver_feature, pseudo_boundary.detach())
        uncertainty = F.interpolate(entropy.detach(), size=receiver_feature.shape[2:],
                                    mode="trilinear", align_corners=False)
        mask = band * uncertainty
        stats["mask_ratio"] = mask.mean().item()
        if not mask.any():
            return receiver_feature, stats
        delta = self.donor_to_delta(token)[None]
        gate = self.gate(torch.cat([receiver_values, token[None].expand(receiver_values.shape[0], -1)], dim=1))
        residual = strength * mask * gate[..., None, None, None] * delta[..., None, None, None]
        stats.update(gate_mean=gate.detach().mean().item(), delta_norm=delta.detach().norm().item(),
                     residual_ratio=(residual.detach().norm() / receiver_feature.detach().norm().clamp_min(1e-12)).item())
        return receiver_feature + residual, stats

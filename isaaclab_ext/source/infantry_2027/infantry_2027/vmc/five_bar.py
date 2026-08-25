"""Batched five-bar kinematics calibrated against infantry_2027_v0."""

from __future__ import annotations

from typing import NamedTuple

import torch


L1 = 0.215
L2 = 0.2537
PHI1_OFFSET = 2.749420977758278
PHI4_OFFSET = 0.31053494255178626


class VmcOutput(NamedTuple):
    length: torch.Tensor
    length_rate: torch.Tensor
    angle: torch.Tensor
    angle_rate: torch.Tensor
    jacobian: torch.Tensor
    singular: torch.Tensor


def _geometry(phi1: torch.Tensor, phi4: torch.Tensor):
    xb, yb = L1 * torch.cos(phi1), L1 * torch.sin(phi1)
    xd, yd = L1 * torch.cos(phi4), L1 * torch.sin(phi4)
    bd2 = (xd - xb).square() + (yd - yb).square()
    a0, b0 = 2.0 * L2 * (xd - xb), 2.0 * L2 * (yd - yb)
    root = torch.sqrt(torch.clamp(a0.square() + b0.square() - bd2.square(), min=0.0))
    phi2 = 2.0 * torch.atan2(b0 + root, a0 + bd2)
    xc, yc = xb + L2 * torch.cos(phi2), yb + L2 * torch.sin(phi2)
    phi3 = torch.atan2(yc - yd, xc - xd)
    return xc, yc, phi2, phi3


def wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


class TorchFiveBarVmc:
    def __init__(self, num_legs: int, device: str | torch.device):
        self.num_legs = num_legs
        self.device = torch.device(device)

    def update(
        self,
        phi1: torch.Tensor,
        phi4: torch.Tensor,
        phi1_rate: torch.Tensor,
        phi4_rate: torch.Tensor,
    ) -> VmcOutput:
        xc, yc, phi2, phi3 = _geometry(phi1, phi4)
        length = torch.hypot(xc, yc)
        geometric_angle = torch.atan2(yc, xc)
        s12 = torch.sin(phi1 - phi2)
        s34 = torch.sin(phi3 - phi4)
        s32 = torch.sin(phi3 - phi2)
        singular = (s32.abs() < 1.0e-6) | (length < 1.0e-6)
        safe_s32 = torch.where(singular, torch.ones_like(s32), s32)
        safe_length = torch.where(singular, torch.ones_like(length), length)
        jacobian = torch.empty((self.num_legs, 2, 2), device=self.device, dtype=phi1.dtype)
        jacobian[:, 0, 0] = L1 * torch.sin(geometric_angle - phi3) * s12 / safe_s32
        jacobian[:, 0, 1] = L1 * torch.sin(geometric_angle - phi2) * s34 / safe_s32
        jacobian[:, 1, 0] = L1 * torch.cos(geometric_angle - phi3) * s12 / (safe_length * safe_s32)
        jacobian[:, 1, 1] = L1 * torch.cos(geometric_angle - phi2) * s34 / (safe_length * safe_s32)
        jacobian[singular] = 0.0
        rates = torch.bmm(jacobian, torch.stack((phi1_rate, phi4_rate), dim=-1).unsqueeze(-1)).squeeze(-1)
        return VmcOutput(
            length,
            rates[:, 0],
            wrap_angle(geometric_angle - torch.pi / 2.0),
            rates[:, 1],
            jacobian,
            singular,
        )

    @staticmethod
    def force_to_torque(jacobian: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        return torch.bmm(jacobian.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)

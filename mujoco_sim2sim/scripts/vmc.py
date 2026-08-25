"""NumPy five-bar VMC matching the calibrated Isaac Lab implementation."""

from __future__ import annotations

import math

import numpy as np


L1 = 0.215
L2 = 0.2537
PHI1_OFFSET = 2.749420977758278
PHI4_OFFSET = 0.31053494255178626


def wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def state(phi1: float, phi4: float, phi1_rate: float, phi4_rate: float):
    xb, yb = L1 * math.cos(phi1), L1 * math.sin(phi1)
    xd, yd = L1 * math.cos(phi4), L1 * math.sin(phi4)
    bd2 = (xd - xb) ** 2 + (yd - yb) ** 2
    a0, b0 = 2.0 * L2 * (xd - xb), 2.0 * L2 * (yd - yb)
    root = math.sqrt(max(0.0, a0 * a0 + b0 * b0 - bd2 * bd2))
    phi2 = 2.0 * math.atan2(b0 + root, a0 + bd2)
    xc, yc = xb + L2 * math.cos(phi2), yb + L2 * math.sin(phi2)
    phi3 = math.atan2(yc - yd, xc - xd)
    length = math.hypot(xc, yc)
    geometric_angle = math.atan2(yc, xc)
    s12, s34, s32 = math.sin(phi1 - phi2), math.sin(phi3 - phi4), math.sin(phi3 - phi2)
    singular = abs(s32) < 1.0e-6 or length < 1.0e-6
    jacobian = np.zeros((2, 2), dtype=np.float64)
    if not singular:
        jacobian[0, 0] = L1 * math.sin(geometric_angle - phi3) * s12 / s32
        jacobian[0, 1] = L1 * math.sin(geometric_angle - phi2) * s34 / s32
        jacobian[1, 0] = L1 * math.cos(geometric_angle - phi3) * s12 / (length * s32)
        jacobian[1, 1] = L1 * math.cos(geometric_angle - phi2) * s34 / (length * s32)
    rates = jacobian @ np.array((phi1_rate, phi4_rate))
    return length, wrap(geometric_angle - math.pi / 2.0), rates, jacobian, singular

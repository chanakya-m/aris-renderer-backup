"""See Assignment 2 descriptions"""

from typing import Union

import torch
from torch import Tensor


def sample_uniform_square(uv: Tensor) -> Tensor:
    return uv * 2 - 1


def pdf_uniform_square(p: Tensor) -> Tensor:
    return torch.ones_like(p[:, 0]) / 4


def sample_tent(uv: Tensor) -> Tensor:
    def sample_tent_1d(u: Tensor) -> Tensor:
        return torch.where(
            u < 0.5,
            torch.sqrt(2 * u) - 1,
            1 - torch.sqrt(2 - 2 * u)
        )

    return torch.stack([sample_tent_1d(uv[:, 0]), sample_tent_1d(uv[:, 1])], dim=1)

def pdf_tent(p: Tensor) -> Tensor:
    return (1 - torch.abs(p[:, 0])) * (1 - torch.abs(p[:, 1]))


def sample_uniform_disk(uv: Tensor) -> Tensor:
    u = uv[:, 0]
    v = uv[:, 1]

    r = torch.sqrt(u)
    theta = 2 * torch.pi * v

    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

def pdf_uniform_disk(p: Tensor) -> Tensor:
    return torch.ones_like(p[:, 0]) / torch.pi


def sample_uniform_sphere(uv: Tensor) -> Tensor:
    u = uv[:, 0]
    v = uv[:, 1]

    phi = 2 * torch.pi * v
    theta = torch.acos(1 - 2*u)

    return torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ], dim=1)

def pdf_uniform_sphere(p: Tensor) -> Tensor:
    return torch.ones_like(p[:, 0]) / (4 * torch.pi)


def sample_uniform_hemisphere(uv: Tensor) -> Tensor:
    u = uv[:, 0]
    v = uv[:, 1]

    phi = 2 * torch.pi * v
    theta = torch.acos(1 - u)

    return torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ], dim=1)

def pdf_uniform_hemisphere(p: Tensor) -> Tensor:
    return torch.ones_like(p[:, 0]) / (2 * torch.pi)


def sample_cosine_hemisphere(uv: Tensor) -> Tensor:
    u = uv[:, 0]
    v = uv[:, 1]

    r = torch.sqrt(u)
    phi = 2 * torch.pi * v

    x_disk = r * torch.cos(phi)
    y_disk = r * torch.sin(phi)

    z = torch.sqrt(1 - r*r + 1e-8)

    return torch.stack([x_disk, y_disk, z], dim=1)

def pdf_cosine_hemisphere(p: Tensor) -> Tensor:
    return torch.clamp(p[:, 2], min=0.0) / torch.pi


def sample_beckmann(uv: Tensor, alpha: float) -> Tensor:
    u = uv[:, 0]
    v = uv[:, 1]

    phi = 2 * torch.pi * v

    log_u = torch.log((1 - u) + 1e-8)
    cos_sq_theta = 1 / (1 - alpha * alpha * log_u)

    cos_theta = torch.sqrt(cos_sq_theta)
    sin_theta = torch.sqrt(torch.clamp(1 - cos_sq_theta, min=0.0))

    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta

    return torch.stack([x, y, z], dim=1)

def pdf_beckmann(p: Tensor, alpha: float) -> Tensor:
    cos_theta = p[:, 2]
    cos_theta = torch.clamp(cos_theta, min=1e-8)

    cos_cubed_theta = cos_theta * cos_theta * cos_theta

    tan_sq_theta = (1 - cos_theta * cos_theta) / (cos_theta * cos_theta)

    alpha_sq = alpha * alpha

    exp = -tan_sq_theta / alpha_sq
    numerator = torch.exp(exp)
    denominator = torch.pi * alpha_sq * cos_cubed_theta

    pdf = numerator / denominator
    pdf[p[:, 2] < 0] = 0.0

    return pdf

import logging

import torch
from torch import Tensor
import torch.nn.functional as F

from aris.core.scene import Scene
from aris.integrator import Integrator, integrator_registry
from aris.utils.tensor_utils import dot

logger = logging.getLogger(__name__)


class PointLightIntegrator(Integrator):
    def __init__(self, position: list[float], energy: list[float]) -> None:
        super().__init__()
        assert len(position) == 3, "position must be [x, y, z]"
        assert len(energy) == 3, "energy must be [r, g, b]"

        self.position = torch.tensor(position, dtype=torch.float32).view(1, 3)
        self.energy = torch.tensor(energy, dtype=torch.float32).view(1, 3)

    def render(self, scene: Scene, rays_o: Tensor, rays_d: Tensor) -> Tensor:
        result = torch.zeros_like(rays_o)
        # YOUR TASK: implement the point light integrator

        geometry = scene.geometry.ray_intersect(rays_o, rays_d)
        hit_mask = geometry.mask     # (N,)
        if not hit_mask.any():
            return result

        points = geometry.points[hit_mask] # (M,3)
        sh_normals = geometry.sh_normals[hit_mask]

        # Shadow rays
        x2p = self.position - points  # (M,3)
        x2p_norm = F.normalize(x2p, p=2, dim=1)
        offset = 1e-4
        shadow_ray_o = points + offset
        shadow_ray_d = x2p_norm
        shadow_geometry = scene.geometry.ray_intersect(shadow_ray_o, shadow_ray_d)

        lit_mask = torch.zeros_like(hit_mask)
        lit_mask[hit_mask] = ~shadow_geometry.mask # all points where light source is visible

        if not lit_mask.any():
            return result

        points_lit = geometry.points[lit_mask] # (K,3)
        normals_lit = geometry.sh_normals[lit_mask] # (K,1)

        x2p_lit = self.position - points_lit
        x2p_sq_norm = torch.sum(x2p_lit * x2p_lit, dim=1, keepdim=True)
        x2p_norm_lit = x2p_lit / torch.sqrt(x2p_sq_norm + 1e-8) # (K,3)

        cos_theta = dot(normals_lit, x2p_norm_lit) # (K,1)
        cos_theta_clamped = torch.clamp(cos_theta, min=0.0)

        denominator = 4 * (torch.pi ** 2) * x2p_sq_norm # (K,1)
        radiance = self.energy * cos_theta_clamped / denominator

        result[lit_mask] = radiance

        return result


integrator_registry.add("point", PointLightIntegrator)

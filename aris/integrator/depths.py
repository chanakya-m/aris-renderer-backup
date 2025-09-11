import torch
from torch import Tensor

from aris.core.scene import Scene
from aris.integrator import Integrator, integrator_registry


class DepthsIntegrator(Integrator):
    def render(self, scene: Scene, rays_o: Tensor, rays_d: Tensor) -> Tensor:
        geometry = scene.geometry.ray_intersect(rays_o, rays_d)
        colors = torch.zeros_like(rays_o)
        hit_mask = geometry.mask
        if hit_mask.any():
            points_h = geometry.points[hit_mask]
            rays_o_h = geometry.rays_o[hit_mask]
            rays_d_h = geometry.rays_d[hit_mask]

            # t = ((p - o) * d) / (d * d)

            vec_o_to_p = points_h - rays_o_h

            numerator = torch.sum(vec_o_to_p * rays_d_h, dim=1)
            denominator = torch.sum(rays_d_h * rays_d_h, dim=1)
            depths_for_hits = numerator / (denominator + 1e-8)
            inv_depths = 1.0 / (depths_for_hits + 1e-8)

            hit_colors = inv_depths.unsqueeze(1).expand(-1, 3)
            colors[hit_mask] = hit_colors

        return colors

integrator_registry.add("depths", DepthsIntegrator)

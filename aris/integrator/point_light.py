import logging

import torch
from torch import Tensor

from aris.core.scene import Scene
from aris.integrator import Integrator, integrator_registry

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
        hitmask = geometry.mask
        if hitmask.any():
            points = geometry.points[hitmask]
            normals = geometry.geo_normals[hitmask]
            rays_o_h = geometry.rays_h[hitmask]
            rays_d_h = geometry.rays_d[hitmask]

        return result


integrator_registry.add("point", PointLightIntegrator)

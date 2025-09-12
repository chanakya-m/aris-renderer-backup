import torch
from torch import Tensor
import torch.nn.functional as F

from aris.camera import Camera, camera_registry
from aris.camera.perspective import PerspectiveCamera


class ThinLensCamera(Camera):
    def __init__(self,
                 width: int,
                 height: int,
                 fov: float,
                 c2w: list[list[float]],
                 aperture: float,
                 focal_distance: float,
                 ) -> None:
        self.camera = PerspectiveCamera(width, height, fov, c2w)
        self.aperture = aperture
        self.focal_distance = focal_distance

    def image_to_rays(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        if self.aperture == 0.0:
            return self.camera.image_to_rays(coords)

        pinhole_rays_d = self.camera.image_to_camera(coords)
        pinhole_rays_o = torch.zeros_like(pinhole_rays_d)

        ft = -self.focal_distance / (pinhole_rays_d[:, 2:3] + 1e-8)
        p_focus = pinhole_rays_o + ft * pinhole_rays_d

        lens_radius = self.aperture / 2.0
        N = coords.shape[0]

        lens_points_2d = (torch.rand(N, 2, device=coords.device) * 2.0 - 1.0) * lens_radius

        new_rays_o = torch.cat([lens_points_2d, torch.zeros(N, 1, device=coords.device)], dim=1)

        new_rays_d = p_focus - new_rays_o
        new_rays_d = F.normalize(new_rays_d, p=2, dim=1)

        return self.camera.camera_to_world(new_rays_o, new_rays_d)

camera_registry.add("thin", ThinLensCamera)

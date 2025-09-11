from torch import Tensor

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
        if self.aperture == 0:
            return self.p_cam.image_to_rays(coords)

        rays_d_pinhole = self.p_cam.image_to_camera(coords)

        z_comp = rays_d_pinhole[:, 2:3]
        ft = -self.focal_distance / (z_comp + 1e-8)
        p_focus = rays_d_pinhole * ft

        n_rays = len(coords)
        lens_radius = self.aperture / 2.0

        p_lens_unit = 2 * torch.rand(n_rays, 2, device=coords.device) - 1
        p_lens_scaled = p_lens_unit * lens_radius

        rays_o_camera = torch.cat([
            p_lens_scaled,
            torch.zeros_like(p_lens_scaled[:, 0:1])
        ], dim=1)

        rays_d_camera = p_focus - rays_o_camera

        norm = torch.linalg.norm(rays_d_camera, dim=1, keepdim=True)
        rays_d_camera = rays_d_camera / (norm + 1e-8)

        return self.p_cam.camera_to_world(rays_o_camera, rays_d_camera)

camera_registry.add("thin", ThinLensCamera)

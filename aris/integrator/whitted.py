import logging

import torch
from torch import Tensor
import torch.nn.functional as F

from aris.core.scene import Scene
from aris.integrator import Integrator, integrator_registry
from aris.utils.tensor_utils import dot

logger = logging.getLogger(__name__)


class WhittedIntegrator(Integrator):
    def __init__(self, max_path_length: int, cont_prob: float) -> None:
        super().__init__()
        self.max_path_length = max_path_length
        self.cont_prob = cont_prob

    def render(self, scene: Scene, rays_o: Tensor, rays_d: Tensor) -> Tensor:
        result = torch.zeros_like(rays_d)
        device = rays_d.device

        # For Whitted: keep track of the 1/p multiplications
        throughput = torch.ones_like(rays_d)

        # Note: you can also start by initializing the active_indices with all indices,
        # and move the first ray_intersect inside the loop.
        # Doing so may or may not simplify the implementaion, it's up to your preferences.

        # start by shooting the rays into the scene
        geo_out = scene.geometry.ray_intersect(rays_o, rays_d)

        # indices of rays that are being traced
        # these are indices in the inputs, for indexing result and throughput
        active_indices = torch.nonzero(geo_out.mask)[:, 0]  # (N, 1) -> (N,)

        # Whitted loop: we keep tracing the rays that hit specular surfaces,
        # until they hit a diffuse surface (so we sample an emitter),
        # or goes into the void (result is zero),
        # or is terminated by Russian-roulette (result is zero)
        for i_path in range(self.max_path_length):
            if len(active_indices) == 0:
                break

            # YOUR TASK: Distribution Ray Tracing (DRT)
            # DRT 1: Check if a ray hits an emitter; if so, add its contribution

            points = geo_out.points[active_indices]
            normals = geo_out.sh_normals[active_indices]
            brdf_indices = geo_out.brdf_i[active_indices]
            wo = -rays_d[active_indices]

            emitter_idx = scene.geometry.emitters_idx[brdf_indices]
            emitter_hit_mask = (emitter_idx != -1)

            if emitter_hit_mask.any():
                for i_emitter in range(len(scene.emitters)):
                    emitter = scene.emitters[i_emitter]
                    current_emitter_hit_mask = (emitter_idx == i_emitter)

                    if current_emitter_hit_mask.any():
                        current_emitter_hit_indices = active_indices[current_emitter_hit_mask]
                        emitted_radiance = emitter.radiance.to(device)
                        result[current_emitter_hit_indices] += emitted_radiance


            non_emitter_hit_mask = ~emitter_hit_mask
            if not non_emitter_hit_mask.any():
                break

            active_indices = active_indices[non_emitter_hit_mask]
            points = points[non_emitter_hit_mask]
            normals = normals[non_emitter_hit_mask]
            brdf_indices = brdf_indices[non_emitter_hit_mask]
            wo = wo[non_emitter_hit_mask]

            # DRT 2: Sample an emitter for all the remaining hit points
            #   First, choose an emitter for every hit point
            #   Then, query each emitter with the assigned points, and add their contribution
            #   (You can find an example in emitter_check.py)

            brdf_query = scene.sample_brdf(wo, normals, brdf_indices)
            specular_mask = brdf_query.is_specular

            diffuse_mask = ~specular_mask

            if diffuse_mask.any():
                diffuse_indices = active_indices[diffuse_mask]
                points_diffuse = points[diffuse_mask]
                normals_diffuse = normals[diffuse_mask]
                brdf_indices_diffuse = brdf_indices[diffuse_mask]
                wo_diffuse = wo[diffuse_mask]

                n_hits = len(points_diffuse)
                n_emitters = len(scene.emitters)
                emitter_choices = torch.randint(0, n_emitters, (n_hits,), device=device)

                for i_emitter in range(n_emitters):
                    emitter = scene.emitters[i_emitter]
                    em_mask = (emitter_choices == i_emitter)
                    if not em_mask.any():
                        continue

                    em_query = emitter.sample(em_mask.sum(), device)
                    y_points = em_query.points
                    y_normals = em_query.normals
                    y_pdf_pos = em_query.pdf

                    x_points_subset = points_diffuse[em_mask]

                    wi = F.normalize(y_points - x_points_subset, p=2, dim=1)

                    em_query.targets = x_points_subset
                    em_query.d_target_point = F.normalize(y_points - x_points_subset, p=2, dim=1) # Direction is from target to source for `le`

                    le_query = emitter.le(em_query, scene.geometry)
                    le = le_query.le

                    dist_sq = dot(y_points - x_points_subset, y_points - x_points_subset)
                    cos_x = torch.clamp(dot(normals_diffuse[em_mask], wi), min=0.0)
                    cos_y = torch.clamp(dot(y_normals, -wi), min=0.0)
                    geometry_term = cos_x * cos_y / (dist_sq + 1e-8)

                    brdf_eval_query = scene.eval_brdf(wo_diffuse[em_mask], normals_diffuse[em_mask], wi, brdf_indices_diffuse[em_mask])
                    f_r = brdf_eval_query.values

                    p_choose_emitter = 1 / n_emitters
                    p_y = p_choose_emitter * y_pdf_pos.view(-1, 1)

                    radiance = f_r * le * geometry_term / (p_y + 1e-8)

                    result[diffuse_indices[em_mask]] += throughput[diffuse_indices[em_mask]] * radiance

            # END OF DRT

            # Whitted 1: change (DRT 2) code above to only points that hit a diffuse surface
            # Whitted 2: continue trace points that hit a specular surface
            pass

            if i_path > 3:
                # Do Russian-roulette after at least 3 bounces
                pass

            # Whitted 3: remove this break
            break

        return result


integrator_registry.add("whitted", WhittedIntegrator)

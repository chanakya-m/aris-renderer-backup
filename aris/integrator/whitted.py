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

        throughput = torch.ones_like(rays_d)

        active_indices = torch.arange(rays_o.shape[0], device=device)

        for i_path in range(self.max_path_length):
            if len(active_indices) == 0:
                break

            geo_out = scene.geometry.ray_intersect(rays_o, rays_d)

            miss_mask = ~geo_out.mask
            active_indices = active_indices[~miss_mask]
            rays_o = rays_o[~miss_mask]
            rays_d = rays_d[~miss_mask]
            throughput = throughput[~miss_mask]
            geo_out.points = geo_out.points[~miss_mask]
            geo_out.sh_normals = geo_out.sh_normals[~miss_mask]
            geo_out.brdf_i = geo_out.brdf_i[~miss_mask]

            if len(active_indices) == 0:
                break

            points = geo_out.points
            normals = geo_out.sh_normals
            brdf_indices = geo_out.brdf_i
            wo = -rays_d

            emitter_idx = scene.geometry.emitters_idx[brdf_indices]
            emitter_hit_mask = (emitter_idx != -1)

            if emitter_hit_mask.any():
                emitter_hit_indices = active_indices[emitter_hit_mask]

                emitter_hit_throughput = throughput[emitter_hit_mask]

                for i_emitter in range(len(scene.emitters)):
                    emitter = scene.emitters[i_emitter]
                    current_emitter_mask = (emitter_idx[emitter_hit_mask] == i_emitter)

                    if current_emitter_mask.any():
                        emitted_radiance = emitter.radiance.to(device) * emitter_hit_throughput[current_emitter_mask]
                        result[emitter_hit_indices[current_emitter_mask]] += emitted_radiance

            continue_mask = ~emitter_hit_mask
            active_indices = active_indices[continue_mask]
            points = points[continue_mask]
            normals = normals[continue_mask]
            brdf_indices = brdf_indices[continue_mask]
            wo = wo[continue_mask]
            throughput = throughput[continue_mask]

            if len(active_indices) == 0:
                break

            brdf_sample = scene.sample_brdf(wo, normals, brdf_indices)
            specular_mask = brdf_sample.is_specular
            diffuse_mask = ~specular_mask

            if diffuse_mask.any():
                diffuse_indices = active_indices[diffuse_mask]
                points_diffuse = points[diffuse_mask]
                normals_diffuse = normals[diffuse_mask]
                brdf_indices_diffuse = brdf_indices[diffuse_mask]
                wo_diffuse = wo[diffuse_mask]
                throughput_diffuse = throughput[diffuse_mask]

                n_hits = len(points_diffuse)
                n_emitters = len(scene.emitters)
                emitter_choices = torch.randint(0, n_emitters, (n_hits,), device=device)

                for i_emitter in range(n_emitters):
                    emitter = scene.emitters[i_emitter]
                    em_mask = (emitter_choices == i_emitter)
                    if not em_mask.any():
                        continue

                    em_query = emitter.sample(em_mask.sum(), device)
                    y_points, y_normals, y_pdf_pos = em_query.points, em_query.normals, em_query.pdf
                    x_points_subset = points_diffuse[em_mask]

                    wi = F.normalize(y_points - x_points_subset, p=2, dim=1)
                    em_query.targets = x_points_subset
                    em_query.d_target_point = wi

                    le_query = emitter.le(em_query, scene.geometry)

                    dist_sq = dot(y_points - x_points_subset, y_points - x_points_subset)
                    cos_x = torch.clamp(dot(normals_diffuse[em_mask], wi), min=0.0)
                    cos_y = torch.clamp(dot(y_normals, -wi), min=0.0)
                    geometry_term = cos_x * cos_y / (dist_sq + 1e-8)

                    brdf_query = scene.eval_brdf(wo_diffuse[em_mask], normals_diffuse[em_mask], wi, brdf_indices_diffuse[em_mask])
                    f_r = brdf_query.values

                    p_y = (1 / n_emitters) * y_pdf_pos.view(-1, 1)
                    radiance = f_r * le_query.le * geometry_term / (p_y + 1e-8)

                    result[diffuse_indices[em_mask]] += radiance * throughput_diffuse[em_mask]

            active_indices = active_indices[specular_mask]
            points = points[specular_mask]
            throughput = throughput[specular_mask]
            normals = normals[specular_mask]


            wi = brdf_sample.wi[specular_mask]
            brdf_val = brdf_sample.values[specular_mask]

            if len(active_indices) == 0:
                break

            throughput *= brdf_val

            if i_path > 3:
                rr_prob = self.cont_prob
                rr_sample = torch.rand(len(active_indices), 1, device=device)
                rr_mask = (rr_sample < rr_prob).squeeze(-1)

                active_indices = active_indices[rr_mask]
                points = points[rr_mask]
                wi = wi[rr_mask]
                throughput = throughput[rr_mask]

                throughput /= rr_prob
                normals = normals[rr_mask]

                if len(active_indices) == 0:
                    break

            offset_normals = normals
            flip_mask = dot(wi, offset_normals).squeeze(-1) < 0
            offset_normals[flip_mask] *= -1

            rays_o = points + offset_normals * 1e-3
            rays_d = wi

        return result


integrator_registry.add("whitted", WhittedIntegrator)

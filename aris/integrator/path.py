import logging

import torch
from torch import Tensor

from aris.core.scene import Scene
from aris.integrator import Integrator, integrator_registry
from aris.utils.tensor_utils import dot

logger = logging.getLogger(__name__)


class PathIntegrator(Integrator):
    def __init__(self, max_path_length: int, cont_prob: float, enable_emitter_sample: bool, mis: bool) -> None:
        """A path tracer implementation

        enable_emitter_sample: by default this integrator is brute-force. If set, it performs next event estimation.
        mis: requires enable_emitter_sample. When set, enables importance sampling.
        """
        super().__init__()
        self.max_path_length = max_path_length
        self.cont_prob = cont_prob
        self.enable_emitter_sample = enable_emitter_sample
        self.mis = mis

    def render(self, scene: Scene, rays_o: Tensor, rays_d: Tensor) -> Tensor:
        result = torch.zeros_like(rays_d)
        # YOUR TASK: implement the path tracer
        device = rays_d.device
        throughput = torch.ones_like(rays_d)
        active_indices = torch.arange(rays_o.shape[0], device=device)

        for i_path in range(self.max_path_length):
            if len(active_indices) == 0:
                break

            geo_out = scene.geometry.ray_intersect(rays_o, rays_d)

            hit_mask = geo_out.mask
            active_indices = active_indices[hit_mask]
            rays_o = rays_o[hit_mask]
            rays_d = rays_d[hit_mask]
            throughput = throughput[hit_mask]

            if len(active_indices) == 0:
                break

            points = geo_out.points[hit_mask]
            normals = geo_out.sh_normals[hit_mask]
            brdf_indices = geo_out.brdf_i[hit_mask]
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

            wi = brdf_sample.wi
            brdf_val = brdf_sample.values

            throughput *= brdf_val

            if i_path > 3:
                rr_prob = self.cont_prob
                rr_sample = torch.rand(len(active_indices), 1, device=device)
                rr_mask = (rr_sample < rr_prob).squeeze(-1)

                active_indices = active_indices[rr_mask]
                points = points[rr_mask]
                wi = wi[rr_mask]
                throughput = throughput[rr_mask]
                normals = normals[rr_mask]

                throughput /= rr_prob

                if len(active_indices) == 0:
                    break

            offset_normals = normals
            flip_mask = dot(wi, offset_normals).squeeze(-1) < 0
            offset_normals[flip_mask] *= -1

            rays_o = points + offset_normals * 1e-3
            rays_d = wi

        return result

integrator_registry.add("path", PathIntegrator)

import logging

import torch
from torch import Tensor

from aris.core.scene import Scene
from aris.integrator import Integrator, integrator_registry

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
        active_indices = torch.arange(len(rays_o), device=device)

        for i_path in range(self.max_path_length):
            if len(active_indices) == 0:
                break

            # Step 1: Trace only the currently active rays
            active_rays_o = rays_o[active_indices]
            active_rays_d = rays_d[active_indices]
            geo_out = scene.geometry.ray_intersect(active_rays_o, active_rays_d)

            # Find which of the active rays actually hit something and terminate misses.
            hit_mask = geo_out.mask
            active_indices = active_indices[hit_mask]
            if len(active_indices) == 0:
                break

            # Step 2: Process the hits from this bounce
            points = geo_out.points[hit_mask]
            normals = geo_out.sh_normals[hit_mask]
            brdf_indices = geo_out.brdf_i[hit_mask]
            wo = -active_rays_d[hit_mask]

            # --- Cleanly separate emitter hits from surface hits ---
            emitter_idx = scene.geometry.emitters_idx[brdf_indices]
            emitter_hit_mask = (emitter_idx != -1)
            surface_hit_mask = ~emitter_hit_mask

            # Step 3: Handle rays that directly hit an emitter (add Le and TERMINATE)
            if emitter_hit_mask.any():
                emitter_hit_global_indices = active_indices[emitter_hit_mask]
                for i_emitter in range(len(scene.emitters)):
                    emitter = scene.emitters[i_emitter]
                    current_emitter_mask = (emitter_idx[emitter_hit_mask] == i_emitter)
                    if current_emitter_mask.any():
                        indices_to_update = emitter_hit_global_indices[current_emitter_mask]
                        emitted_radiance = emitter.radiance.to(device)
                        result[indices_to_update] += throughput[indices_to_update] * emitted_radiance

            # Step 4: Handle rays that hit a non-emitting surface (bounce and CONTINUE)
            if not surface_hit_mask.any():
                break # All remaining rays hit emitters, so the paths all end here.

            # Filter all data to only the rays that hit a regular surface and will continue.
            active_indices = active_indices[surface_hit_mask]
            points_survivors = points[surface_hit_mask]
            normals_survivors = normals[surface_hit_mask]
            brdf_indices_survivors = brdf_indices[surface_hit_mask]
            wo_survivors = wo[surface_hit_mask]

            # Sample the BRDF for these continuing surfaces
            brdf_sample = scene.sample_brdf(wo_survivors, normals_survivors, brdf_indices_survivors)

            # Update throughput for the continuing rays
            throughput[active_indices] *= brdf_sample.values

            # Russian Roulette for path termination
            if i_path > 3:
                rr_prob = self.cont_prob
                rr_samples = torch.rand(len(active_indices), device=device)
                rr_keep_mask = (rr_samples < rr_prob)

                active_indices = active_indices[rr_keep_mask]
                if len(active_indices) == 0:
                    break

                # Boost throughput for surviving rays
                throughput[active_indices] /= rr_prob

                # We need to filter the BRDF sample results as well for the next step
                brdf_sample.wi = brdf_sample.wi[rr_keep_mask]

            # Step 5: Prepare for the next bounce
            offset = 1e-3
            # We need to filter the points and normals again for the RR survivors
            final_points = points_survivors
            final_normals = normals_survivors
            if i_path > 3:
                final_points = final_points[rr_keep_mask]
                final_normals = final_normals[rr_keep_mask]

            next_rays_o = final_points + final_normals * offset
            next_rays_d = brdf_sample.wi

            # Update the global ray tensors at the surviving indices.
            rays_o[active_indices] = next_rays_o
            rays_d[active_indices] = next_rays_d

        return result

integrator_registry.add("path", PathIntegrator)

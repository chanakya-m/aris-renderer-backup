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

        # "Global indices" that always refer to the original, full-sized tensors.
        active_indices = torch.arange(len(rays_o), device=device)
        # The rays for the current bounce.
        current_rays_o = rays_o
        current_rays_d = rays_d

        for i_path in range(self.max_path_length):
            if len(active_indices) == 0:
                break

            # Step 1: Trace the currently active rays
            geo_out = scene.geometry.ray_intersect(current_rays_o, current_rays_d)

            # Find which rays from the current batch actually hit something.
            hit_mask = geo_out.mask

            # Update the global indices, terminating rays that missed.
            active_indices = active_indices[hit_mask]
            if len(active_indices) == 0:
                break

            # Step 2: Process the hits from this bounce
            points = geo_out.points[hit_mask]
            normals = geo_out.sh_normals[hit_mask]
            brdf_indices = geo_out.brdf_i[hit_mask]
            wo = -current_rays_d[hit_mask]

            # Handle rays that directly hit an emitter
            emitter_idx = scene.geometry.emitters_idx[brdf_indices]
            emitter_hit_mask = (emitter_idx != -1)

            if emitter_hit_mask.any():
                emitter_hit_global_indices = active_indices[emitter_hit_mask]
                for i_emitter in range(len(scene.emitters)):
                    emitter = scene.emitters[i_emitter]
                    current_emitter_mask = (emitter_idx[emitter_hit_mask] == i_emitter)
                    if current_emitter_mask.any():
                        indices_to_update = emitter_hit_global_indices[current_emitter_mask]
                        emitted_radiance = emitter.radiance.to(device)
                        result[indices_to_update] += throughput[indices_to_update] * emitted_radiance

                # Terminate rays that hit an emitter by filtering them out for the next bounce
                non_emitter_mask = ~emitter_hit_mask
                active_indices = active_indices[non_emitter_mask]
                if len(active_indices) == 0: break
                points, normals, brdf_indices, wo = points[non_emitter_mask], normals[non_emitter_mask], brdf_indices[non_emitter_mask], wo[non_emitter_mask]

            # Step 3: Split between diffuse and specular surfaces
            brdf_sample = scene.sample_brdf(wo, normals, brdf_indices)
            diffuse_mask = ~brdf_sample.is_specular
            specular_mask = brdf_sample.is_specular

            # Handle diffuse hits (DRT and terminate)
            if diffuse_mask.any():
                diffuse_global_indices = active_indices[diffuse_mask]
                points_d, normals_d, brdf_indices_d, wo_d = points[diffuse_mask], normals[diffuse_mask], brdf_indices[diffuse_mask], wo[diffuse_mask]

                n_hits, n_emitters = len(points_d), len(scene.emitters)
                if n_emitters > 0:
                    emitter_choices = torch.randint(0, n_emitters, (n_hits,), device=device)
                    for i_emitter in range(n_emitters):
                        emitter = scene.emitters[i_emitter]
                        em_mask = (emitter_choices == i_emitter)
                        if not em_mask.any(): continue

                        em_query = emitter.sample(em_mask.sum(), device)
                        y_points, y_normals, y_pdf_pos = em_query.points, em_query.normals, em_query.pdf
                        x_points_subset = points_d[em_mask]
                        wi = F.normalize(y_points - x_points_subset, p=2, dim=1)
                        em_query.targets, em_query.d_target_point = x_points_subset, F.normalize(y_points - x_points_subset, p=2, dim=1)
                        le_query = emitter.le(em_query, scene.geometry)

                        le, dist_sq = le_query.le, dot(y_points - x_points_subset, y_points - x_points_subset)
                        cos_x = torch.clamp(dot(normals_d[em_mask], wi), min=0.0)
                        cos_y = torch.clamp(dot(y_normals, -wi), min=0.0)
                        geometry_term = cos_x * cos_y / (dist_sq + 1e-8)
                        brdf_query = scene.eval_brdf(wo_d[em_mask], normals_d[em_mask], wi, brdf_indices_d[em_mask])
                        f_r = brdf_query.values
                        p_choose_emitter = 1.0 / n_emitters
                        p_y = p_choose_emitter * y_pdf_pos.view(-1, 1)
                        radiance = f_r * le * geometry_term / (p_y + 1e-8)

                        indices_to_update = diffuse_global_indices[em_mask]
                        result[indices_to_update] += throughput[indices_to_update] * radiance

            # Step 4: Prepare next bounce for specular rays
            if not specular_mask.any():
                break

            # Filter the global indices to only keep the specular survivors
            active_indices = active_indices[specular_mask]

            # Update throughput *before* Russian Roulette, as per instructor's note
            throughput[active_indices] *= brdf_sample.values[specular_mask]

            # Apply Russian Roulette
            if i_path > 3:
                rr_prob = self.cont_prob
                rr_samples = torch.rand(len(active_indices), device=device)
                rr_keep_mask = (rr_samples < rr_prob)

                active_indices = active_indices[rr_keep_mask]
                if len(active_indices) == 0:
                    break

                # Boost throughput ONLY for the survivors, as per instructor's note
                throughput[active_indices] /= rr_prob

            # Create the rays for the next bounce. We filter the data using the final
            # set of masks that determined which rays continue.
            offset = 1e-3
            final_specular_mask = brdf_sample.is_specular
            if i_path > 3:
                # Need to update this mask to account for RR survivors
                temp_mask = torch.zeros_like(final_specular_mask)
                temp_mask[specular_mask] = rr_keep_mask
                final_specular_mask = temp_mask

            # These are the rays for the next iteration of the loop
            current_rays_o = points[final_specular_mask] + normals[final_specular_mask] * offset
            current_rays_d = brdf_sample.wi[final_specular_mask]

        return result

integrator_registry.add("whitted", WhittedIntegrator)

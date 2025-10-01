from typing import Optional

import torch
import torch.nn.functional as F
from aris.utils.tensor_utils import dot

from aris.emitter import Emitter, EmitterQuery, emitter_registry
from aris.geometry import Geometry


class AreaLight(Emitter):
    def __init__(self, radiance: list[int], geometry: Geometry, i_primitive: int, i_emitter: int) -> None:
        """Area light implementation

        radiance: energy of this light source
        geometry: the scene geometry
        i_primitive: the index of the geometry primitive (e.g. mesh) this emitter is attached to
        i_emitter: the index of this emitter in all emitters
        """

        super().__init__(geometry, i_primitive, i_emitter)

        assert len(radiance) == 3, "radiance should be RGB values"

        self.radiance = torch.tensor(radiance, dtype=torch.float32).view(1, 3)


    def sample(self, n_samples: int, device: str) -> EmitterQuery:
        """Sample points from the area light"""
        # YOUR TASK: sample a point from the mesh of this emitter
        # return an EmitterQuery with points, normals, and pdf set correctly
        # Hint: see Geometry.uniform_sample
        points, normals = self.geometry.uniform_sample(self.i_primitive, n_samples, device)
        query = EmitterQuery(points=points, normals=normals)
        self.pos_pdf(query)
        return query

    def pos_pdf(self, query: EmitterQuery) -> EmitterQuery:
        """Position pdf of points from the area light"""
        # YOUR TASK: given an EmitterQuery with points already set,
        # compute the PDF of sampling these points, and set query.pdf
        pdf_value = self.geometry.uniform_sample_pos_pdf(self.i_primitive)
        n_points = query.points.shape[0]
        query.pdf = torch.full((n_points,), pdf_value)
        return query

    def le(self, query: EmitterQuery, geometry: Optional[Geometry]) -> EmitterQuery:
        """Compute the Le term
        If geometry is not None, check if the points and targets are mutually visible
        """
        # YOUR TASK: given an EmitterQuery with points, normals, targets, and d_target_point set,
        # compute the Le term from points to targets, and set query.le
        # also, set query.mask to indicate which targets are lid
        query.le = torch.zeros_like(query.points)
        query.mask = torch.zeros(len(query.points), dtype=torch.bool)

        cos_theta = dot(query.normals, -query.d_target_point)
        front_face_mask = (cos_theta > 0).squeeze()

        if not front_face_mask.any():
            return query

        final_mask = front_face_mask

        if geometry is not None:
            # only tracing rays for front-facing points
            points_ff = query.points[front_face_mask]
            normals_ff = query.normals[front_face_mask]
            targets_ff = query.targets[front_face_mask]

            offset = 1e-4
            shadow_ray_o = points_ff + normals_ff * offset

            # The direction vector is the UNNORMALIZED vector to the target.
            # This makes the target lie at t=1 for the ray equation.
            shadow_ray_d = targets_ff - shadow_ray_o

            shadow_geom = geometry.ray_intersect(shadow_ray_o, shadow_ray_d)

            # Assume all front-facing points are visible unless we find a valid occluder
            is_occluded = torch.zeros(len(points_ff), dtype=torch.bool)

            shadow_hit_mask = shadow_geom.mask
            if shadow_hit_mask.any():
                # For rays that hit something, we calculate their parametric distance 't'
                hit_points = shadow_geom.points[shadow_hit_mask]
                ray_origins = shadow_geom.rays_o[shadow_hit_mask]
                ray_directions = shadow_geom.rays_d[shadow_hit_mask]

                vec_to_hit = hit_points - ray_origins

                # Calculate t = ((H - P_offset) • d) / (d • d)
                numerator = dot(vec_to_hit, ray_directions).squeeze()
                denominator = dot(ray_directions, ray_directions).squeeze()

                # Adding an epsilon to the denominator prevents division by zero
                t = numerator / (denominator + 1e-8)

                # A true occlusion is a hit between the origin and the target.
                # The target is at t=1, so we check for hits with 0 < t < 1.
                # We use a small epsilon on both ends for floating point safety.
                true_occlusion = (t > 1e-4) & (t < 0.9999)

                # Update the occlusion status for the points that had a shadow ray hit
                is_occluded[shadow_hit_mask] = true_occlusion

            # The final visibility mask includes all front-facing points that were NOT occluded
            visibility_mask = torch.zeros_like(front_face_mask)
            visibility_mask[front_face_mask] = ~is_occluded
            final_mask = visibility_mask

        query.mask = final_mask

        if query.mask.any():
            query.le[query.mask] = self.radiance

        return query

emitter_registry.add("area", AreaLight)

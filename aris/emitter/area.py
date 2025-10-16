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
            points_ff = query.points[front_face_mask]
            normals_ff = query.normals[front_face_mask]
            targets_ff = query.targets[front_face_mask]

            offset = 1e-4
            shadow_ray_o = points_ff + normals_ff * offset

            shadow_ray_d = targets_ff - shadow_ray_o

            shadow_geo = geometry.ray_intersect(shadow_ray_o, shadow_ray_d)

            is_occluded = torch.zeros(len(points_ff), dtype=torch.bool)

            shadow_hit_mask = shadow_geo.mask
            if shadow_hit_mask.any():
                hit_points = shadow_geo.points[shadow_hit_mask]
                ray_os = shadow_geo.rays_o[shadow_hit_mask]
                ray_ds = shadow_geo.rays_d[shadow_hit_mask]

                vec_to_hit = hit_points - ray_os

                # t = ((H - P) • d) / (d • d)
                numerator = dot(vec_to_hit, ray_ds).squeeze()
                denominator = dot(ray_ds, ray_ds).squeeze()

                t = numerator / (denominator + 1e-8)

                epsilon = 1e-4
                true_occlusion = (t > (0.0 + epsilon)) & (t < (1.0 - epsilon))

                is_occluded[shadow_hit_mask] = true_occlusion

            visibility_mask = torch.zeros_like(front_face_mask)
            visibility_mask[front_face_mask] = ~is_occluded
            final_mask = visibility_mask

        query.mask = final_mask

        if query.mask.any():
            query.le[query.mask] = self.radiance

        return query

emitter_registry.add("area", AreaLight)

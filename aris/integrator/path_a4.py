# Reference implementation for CMSC740, Fall 24
# Do not share this file with others or upload to public repositories

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aris.core.scene import Scene
from aris.emitter import EmitterQuery
from aris.geometry import GeometryOutput
from aris.integrator import Integrator, integrator_registry
from aris.utils.networks import NeradMlp
from aris.utils.tensor_utils import dot

logger = logging.getLogger(__name__)


class PathIntegrator(Integrator, nn.Module):
    def __init__(self, max_path_length: int, cont_prob: float, enable_emitter_sample: bool, mis: bool, emitter_bounce: bool = False) -> None:
        super().__init__()
        self.max_path_length = max_path_length
        self.cont_prob = cont_prob
        self.mis = mis
        self.enable_emitter_sample = enable_emitter_sample
        self.emitter_bounce = emitter_bounce
        self.network = None

        if not enable_emitter_sample:
            assert not mis, "MIS cannot be enabled when emitter_sample is disabled"

    def init_nerad(self):
        self.network = NeradMlp()

    def render(self, scene: Scene, rays_o: Tensor, rays_d: Tensor) -> Tensor:
        result = torch.zeros_like(rays_d)
        throughput = torch.ones_like(rays_d)
        is_specular = torch.zeros(0)
        prev_pdf = torch.zeros(0)

        geo_out = scene.geometry.ray_intersect(rays_o, rays_d)

        # indices of rays that are being traced
        active_indices = torch.nonzero(geo_out.mask)[:, 0]  # (N, 1) -> (N,)

        wo = -rays_d[active_indices]
        for i_path in range(self.max_path_length):
            if len(active_indices) == 0:
                break

            # get data of hit points
            mask = geo_out.mask
            normals = geo_out.sh_normals[mask]
            points = geo_out.points[mask]
            brdf_i = geo_out.brdf_i[mask]
            hit_emitter = torch.zeros_like(normals[:, 0], dtype=torch.bool)

            # hit emitter
            self.emitter_hit(i_path, scene, geo_out, points, normals, mask, brdf_i, is_specular,
                             active_indices, throughput, prev_pdf, result, hit_emitter)

            # stop tracing rays that hit emitters
            if not self.emitter_bounce:
                no_hit_emitter = ~hit_emitter
                active_indices = active_indices[no_hit_emitter]
                wo = wo[no_hit_emitter]
                normals = normals[no_hit_emitter]
                points = points[no_hit_emitter]
                brdf_i = brdf_i[no_hit_emitter]

            # sample BRDF
            brdf_output = scene.sample_brdf(wo, normals, brdf_i)
            is_specular = brdf_output.is_specular

            # sample emitter
            if self.enable_emitter_sample:
                self.sample_emitter(scene, wo, points, normals, brdf_i, ~is_specular, active_indices, throughput, result)

            # update throughput for reflected rays
            throughput[active_indices] *= brdf_output.values

            # next bounce
            wi = brdf_output.wi
            geo_out = scene.geometry.ray_intersect(points + wi * 0.0001, wi)
            hit = geo_out.mask

            # Russian roulette
            if i_path > 3:
                cont_prob = self.cont_prob
                cont_mask = torch.rand(len(active_indices), device=hit.device) < cont_prob
                throughput[active_indices] /= cont_prob
                next_mask = hit & cont_mask
            else:
                next_mask = hit

            # keep information of traced rays
            geo_out.mask = next_mask
            active_indices = active_indices[next_mask]
            wo = -wi[next_mask]
            is_specular = is_specular[next_mask]
            prev_pdf = brdf_output.pdf[next_mask]

        return result

    def render_nerad(self, scene: Scene, rays_o: Tensor, rays_d: Tensor) -> tuple[Tensor, Tensor]:
        # results: LHS, RHS and E
        RHS = torch.zeros_like(rays_d)
        LHS = torch.zeros_like(rays_d)
        E = torch.zeros_like(rays_d)

        # In nerad code, throughput isn't needed,
        # and a constant array is kept for calling functions
        throughput = torch.ones_like(rays_d)

        geo_out = scene.geometry.ray_intersect(rays_o, rays_d)

        # indices of rays that are being traced
        active_indices = torch.nonzero(geo_out.mask)[:, 0]  # (N, 1) -> (N,)
        if len(active_indices) == 0:
            return LHS, RHS

        # get data of hit points
        wo = -rays_d[active_indices]
        mask = geo_out.mask
        normals = geo_out.sh_normals[mask]
        points = geo_out.points[mask]
        brdf_i = geo_out.brdf_i[mask]
        hit_emitter = torch.zeros_like(normals[:, 0], dtype=torch.bool)

        # compute LHS
        LHS[mask] = self.network(points, wo, normals, scene.get_albedo_brdf(brdf_i).detach())

        # hit emitter, store results in E
        self.emitter_hit(0, scene, geo_out, points, normals, mask, brdf_i, None,
                         active_indices, throughput, None, E, hit_emitter)

        # YOUR TASK: implement correct RHS
        RHS = self.render(scene, rays_o, rays_d)  # remove this placeholder!

        # YOUR TASK: replace second return value by E+RHS when RHS is done
        return E + LHS, RHS

    def emitter_hit(
        self,
        ### inputs - read
        i_path: int,
        scene: Scene,
        geo_out: GeometryOutput,
        points: Tensor,
        normals: Tensor,
        mask: Tensor,
        brdf_i: Tensor,
        # is the ray from a specular hit?
        # required if i_path > 0
        is_specular: Optional[Tensor],
        active_indices: Tensor,
        throughput: Tensor,
        # BRDF sampling pdf of the ray
        # required if i_path > 0 and mis enabled
        prev_pdf: Optional[Tensor],
        ### outputs - modified in-place
        result: Tensor,
        hit_emitter: Tensor,
    ):
        for i_primitive in range(len(scene.geometry)):
            i_emitter = scene.geometry.emitters_idx[i_primitive]
            if i_emitter < 0:
                continue
            emitter = scene.emitters[i_emitter]
            em_mask = brdf_i == i_primitive
            if em_mask.sum() == 0:
                continue

            hit_emitter |= em_mask
            mis_weight = torch.ones_like(points[em_mask, 0:1])

            em_query = EmitterQuery(points[em_mask], normals[em_mask])
            em_query.targets = geo_out.rays_o[mask][em_mask]
            em_query.d_target_point = F.normalize(geo_out.rays_d[mask][em_mask], dim=1)
            emitter.pos_pdf(em_query)
            emitter.le(em_query, None)

            if i_path > 0:
                is_diffuse = ~is_specular[em_mask]
                if self.enable_emitter_sample:
                    mis_weight[is_diffuse] = 0

            if self.mis and i_path > 0:
                dist = (em_query.targets - em_query.points)
                dist = torch.sum(dist * dist, dim=1, keepdim=True)

                emitter_pdf = em_query.pdf.view(-1, 1) / dot(em_query.normals, em_query.d_target_point).abs() * dist

                em_prev_pdf = prev_pdf[em_mask][is_diffuse].view(-1, 1)
                mis_weight[is_diffuse] = em_prev_pdf / (emitter_pdf[is_diffuse] + em_prev_pdf)
                mis_weight[~torch.isfinite(mis_weight)] = 0

            result[active_indices[em_mask]] += mis_weight * throughput[active_indices[em_mask]] * em_query.le

    def sample_emitter(
        self,
        # input - read
        scene: Scene,
        wo: Tensor,
        points: Tensor,
        normals: Tensor,
        brdf_i: Tensor,
        is_diffuse: Tensor,
        active_indices: Tensor,
        throughput: Tensor,
        # output - modified in-place
        result: Tensor,
    ):
        device = points.device
        emitter_choices = torch.from_numpy(np.random.choice(len(scene.emitters), [len(points)])).to(device)

        for i_emitter in range(len(scene.emitters)):
            emitter = scene.emitters[i_emitter]
            em_mask = (emitter_choices == i_emitter) & is_diffuse
            if em_mask.sum() == 0:
                continue

            em_points = points[em_mask]
            em_wo = wo[em_mask]
            em_normals = normals[em_mask]
            em_brdf_i = brdf_i[em_mask]

            em_query = emitter.sample(len(em_points), device)
            em_query.targets = em_points
            em_query.d_target_point = F.normalize(em_query.points - em_query.targets, dim=1)
            emitter.le(em_query, scene.geometry)

            dxy = em_query.d_target_point
            em_brdf = scene.eval_brdf(em_wo, em_normals, dxy, em_brdf_i)

            dist = (em_query.targets - em_query.points)
            dist = torch.sum(dist * dist, dim=1, keepdim=True)

            cos_ny_dxy = dot(em_query.normals, dxy).abs()
            g = dot(em_normals, dxy).abs() * cos_ny_dxy / dist

            if self.mis:
                converted_pdf = em_query.pdf.view(-1, 1) / cos_ny_dxy * dist
                mis_weight = converted_pdf / (converted_pdf + em_brdf.pdf.view(-1, 1))
                mis_weight[~torch.isfinite(mis_weight)] = 0
            else:
                mis_weight = 1

            result[active_indices[em_mask]] += (
                mis_weight * throughput[active_indices[em_mask]] * g * em_query.le *
                em_brdf.values * len(scene.emitters) / em_query.pdf.view(-1, 1)
            )


integrator_registry.add("path_a4", PathIntegrator)

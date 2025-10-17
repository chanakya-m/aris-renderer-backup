# brdf/dielectric.py (Corrected using standard vector math)

import torch
from torch import Tensor
import torch.nn.functional as F

from aris.brdf import Brdf, BrdfQuery, brdf_registry


def fresnel(cos_theta_i: Tensor, cos_theta_t: Tensor, eta_in: Tensor, eta_out: Tensor) -> Tensor:
    eps = 1e-8
    r_s_num = eta_in * cos_theta_i - eta_out * cos_theta_t
    r_s_den = eta_in * cos_theta_i + eta_out * cos_theta_t
    r_s = torch.pow(r_s_num / (r_s_den + eps), 2)
    r_p_num = eta_out * cos_theta_i - eta_in * cos_theta_t
    r_p_den = eta_out * cos_theta_i + eta_in * cos_theta_t
    r_p = torch.pow(r_p_num / (r_p_den + eps), 2)
    return (r_s + r_p) / 2.0

class DielectricBrdf(Brdf):
    def __init__(self, ext_ior: float, int_ior: float) -> None:
        super().__init__()
        self.ext_ior = ext_ior
        self.int_ior = int_ior

    def sample(self, query: BrdfQuery) -> BrdfQuery:
        wo = query.wo
        device = wo.device

        # Step 1: Determine if entering or leaving and set IORs.
        # In local space, normal is (0,0,1). If wo.z > 0, we are outside looking in.
        entering = wo[:, 2:3] > 0.0
        eta_in = torch.where(entering, self.ext_ior, self.int_ior)
        eta_out = torch.where(entering, self.int_ior, self.ext_ior)
        eta = eta_in / eta_out
        cos_theta_i = torch.abs(wo[:, 2:3])

        # Step 2: Calculate reflection vector (always possible)
        wi_reflect = wo * torch.tensor([-1., -1., 1.], device=device)

        # Step 3: Check for Total Internal Reflection (TIR)
        sin_theta_t_sq = eta * eta * (1 - cos_theta_i * cos_theta_i)
        is_tir = (sin_theta_t_sq > 1.0).squeeze(-1)

        # Step 4: Calculate refraction vector using the robust formula
        # This is only physically valid if no TIR, but we compute it and select later.
        # CRITICAL FIX 1: Clamp the input to sqrt to prevent NaN values.
        cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin_theta_t_sq, min=0.0))

        # CRITICAL FIX 2: Use the robust vector formula for refraction.
        # The normal 'n' in local space is always (0,0,1).
        normal = torch.tensor([0., 0., 1.], device=device).view(1, 3)
        # The sign of wo.z determines if we are entering or leaving.
        sign = torch.sign(wo[:, 2:3])

        wi_refract = -wo * eta.view(-1, 1) + normal * sign * (eta * cos_theta_i - cos_theta_t).view(-1, 1)

        # Step 5: Calculate reflection probability using Fresnel equations
        reflect_prob = fresnel(cos_theta_i, cos_theta_t, eta_in, eta_out)

        # If TIR occurs, reflection probability is 100%.
        reflect_prob = torch.where(is_tir.unsqueeze(-1), torch.tensor(1.0, device=device), reflect_prob)

        # Step 6: Probabilistically choose between reflection and refraction
        random_sample = torch.rand(wo.shape[0], 1, device=device)
        should_reflect = (random_sample < reflect_prob).squeeze(-1)

        wi = torch.where(should_reflect.unsqueeze(-1), wi_reflect, wi_refract)

        # PDF for a discrete sample is 1.0. Values is also 1.0.
        return query.output(
            wi,
            torch.ones_like(wo),
            torch.ones_like(wo[:, 0]),
            torch.ones_like(wi[:, 0], dtype=torch.bool),
        )

    def eval(self, query: BrdfQuery) -> BrdfQuery:
        # Discrete BRDFs evaluate to zero
        wo = query.wo
        query.values = torch.zeros_like(wo)
        query.pdf = torch.zeros_like(wo[:, 0])
        query.is_specular = torch.ones_like(wo[:, 0], dtype=torch.bool)
        return query

brdf_registry.add("dielectric", DielectricBrdf)

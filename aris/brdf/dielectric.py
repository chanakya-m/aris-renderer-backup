import torch
from torch import Tensor
import torch.nn.functional as F

from aris.brdf import Brdf, BrdfQuery, brdf_registry


def fresnel(cos_theta_i: Tensor, cos_theta_t: Tensor, eta_in: Tensor, eta_out: Tensor) -> Tensor:
    # s-polarized
    r_s_num = eta_in * cos_theta_i - eta_out * cos_theta_t
    r_s_den = eta_in * cos_theta_i + eta_out * cos_theta_t
    r_s = torch.pow(r_s_num / r_s_den, 2)

    # p-polarized
    r_p_num = eta_out * cos_theta_i - eta_in * cos_theta_t
    r_p_den = eta_out * cos_theta_i + eta_in * cos_theta_t
    r_p = torch.pow(r_p_num / r_p_den, 2)

    # Total reflectance for unpolarized light
    return (r_s + r_p) / 2.0

class DielectricBrdf(Brdf):
    def __init__(self, ext_ior: float, int_ior: float) -> None:
        super().__init__()
        self.ext_ior = ext_ior
        self.int_ior = int_ior

    def sample(self, query: BrdfQuery) -> BrdfQuery:
        wo = query.wo
        device = wo.device

        # YOUR TASK: sample wi according to Fresnel
        wi = torch.zeros_like(wo)

        entering = wo[:, 2:3] > 0

        eta_in = torch.where(entering, self.ext_ior, self.int_ior)
        eta_out = torch.where(entering, self.int_ior, self.ext_ior)

        cos_theta_i = torch.abs(wo[:, 2:3]) # z component

        wi_reflect = wo * torch.tensor([-1., -1., 1.], device=device)

        eta = eta_in / eta_out
        sin_theta_t_sq = eta * eta * (1 - cos_theta_i * cos_theta_i)

        is_tir = (sin_theta_t_sq > 1.0).squeeze(-1)

        cos_theta_t = torch.sqrt(1 - sin_theta_t_sq)
        wo_perp = wo * torch.tensor([1., 1., 0.], device=device)

        wi_refract = -F.normalize(wo_perp, p=2, dim=1) * eta.view(-1, 1) * torch.sqrt(1 - cos_theta_t * cos_theta_t) \
                     - torch.tensor([0., 0., 1.], device=device) * cos_theta_t * torch.sign(wo[:, 2:3])

        reflect_prob = fresnel(cos_theta_i, eta_in, eta_out)

        reflect_prob = torch.where(is_tir.unsqueeze(-1), torch.tensor(1.0, device=device), reflect_prob)

        random_sample = torch.rand(wo.shape[0], 1, device=device)
        should_reflect = (random_sample < reflect_prob).squeeze(-1)

        wi = torch.where(should_reflect.unsqueeze(-1), wi_reflect, wi_refract)

        return query.output(
            wi,
            torch.ones_like(wo),
            torch.zeros_like(wo[:, 0]),
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

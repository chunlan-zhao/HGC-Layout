"""Output distributions for planar coordinates and orientation.

Eq. (3): the XY coordinate is a bivariate Gaussian mixture with M components.
Orientation is a von Mises distribution with mean direction and concentration.
Coordinates are handled in normalized scene units in [0, 1] and converted to
metres by the caller, which keeps the mixture parameters scale free.
"""
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMixtureHead(nn.Module):
    """Predicts mixture weights, means and Cholesky factors of M components."""

    def __init__(self, d_model: int, n_components: int = 5, min_scale: float = 1e-3):
        super().__init__()
        self.m = n_components
        self.min_scale = min_scale
        self.logits = nn.Linear(d_model, n_components)
        self.means = nn.Linear(d_model, n_components * 2)
        self.scales = nn.Linear(d_model, n_components * 3)   # log s_x, log s_y, correlation

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = h.size(0)
        pi = F.softmax(self.logits(h), dim=-1)
        mu = torch.sigmoid(self.means(h)).view(b, self.m, 2)
        raw = self.scales(h).view(b, self.m, 3)
        sx = F.softplus(raw[..., 0]) + self.min_scale
        sy = F.softplus(raw[..., 1]) + self.min_scale
        rho = torch.tanh(raw[..., 2]) * 0.95
        return {"pi": pi, "mu": mu, "sx": sx, "sy": sy, "rho": rho}

    @staticmethod
    def log_prob(params: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Log density of the mixture at `target`, shape (B,)."""
        mu, sx, sy, rho = params["mu"], params["sx"], params["sy"], params["rho"]
        d = target.unsqueeze(1) - mu
        zx, zy = d[..., 0] / sx, d[..., 1] / sy
        one_minus = 1 - rho ** 2
        quad = (zx ** 2 - 2 * rho * zx * zy + zy ** 2) / one_minus
        log_norm = torch.log(2 * math.pi * sx * sy * torch.sqrt(one_minus))
        comp = -0.5 * quad - log_norm
        return torch.logsumexp(torch.log(params["pi"] + 1e-12) + comp, dim=-1)

    @staticmethod
    def mode(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean of the highest-weight component, shape (B, 2)."""
        idx = params["pi"].argmax(dim=-1)
        return params["mu"][torch.arange(params["mu"].size(0)), idx]

    @staticmethod
    def sample(params: Dict[str, torch.Tensor], generator: torch.Generator = None) -> torch.Tensor:
        """Draw one point per row from the mixture."""
        pi, mu, sx, sy, rho = (params[k] for k in ("pi", "mu", "sx", "sy", "rho"))
        idx = torch.multinomial(pi, 1, generator=generator).squeeze(-1)
        rows = torch.arange(mu.size(0))
        m, s1, s2, r = mu[rows, idx], sx[rows, idx], sy[rows, idx], rho[rows, idx]
        eps = torch.randn(m.shape, generator=generator)
        x = m[:, 0] + s1 * eps[:, 0]
        y = m[:, 1] + s2 * (r * eps[:, 0] + torch.sqrt(1 - r ** 2) * eps[:, 1])
        return torch.stack([x, y], dim=-1)


class VonMisesHead(nn.Module):
    """Predicts the mean direction and concentration of the orientation."""

    def __init__(self, d_model: int, min_kappa: float = 1e-2):
        super().__init__()
        self.min_kappa = min_kappa
        self.direction = nn.Linear(d_model, 2)
        self.concentration = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        vec = self.direction(h)
        mu = torch.atan2(vec[:, 1], vec[:, 0]) % (2 * math.pi)
        kappa = F.softplus(self.concentration(h)).squeeze(-1) + self.min_kappa
        return {"mu": mu, "kappa": kappa}

    @staticmethod
    def log_prob(params: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Log density, using a stable series approximation of log I0(kappa)."""
        kappa = params["kappa"]
        return kappa * torch.cos(target - params["mu"]) - math.log(2 * math.pi) - _log_i0(kappa)

    @staticmethod
    def mode(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return params["mu"]


def _log_i0(kappa: torch.Tensor) -> torch.Tensor:
    """log of the modified Bessel function I0, small- and large-argument forms."""
    small = torch.log(torch.i0(torch.clamp(kappa, max=80.0)))
    large = kappa - 0.5 * torch.log(2 * math.pi * torch.clamp(kappa, min=1e-6))
    return torch.where(kappa < 80.0, small, large)

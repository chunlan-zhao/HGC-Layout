"""Graph attention encoder with retained attention coefficients.

A multi-head graph attention network in the standard formulation: additive
attention logits over incoming edges, softmax per destination node, and a
concatenation of the heads at every layer except the last, which averages
them. Each layer returns its attention coefficients alpha_ij so that the
placement decisions can be inspected after generation.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """One multi-head graph attention layer over an explicit edge index."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, concat: bool = True,
                 dropout: float = 0.0, negative_slope: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        h = self.lin(x).view(n, self.heads, self.out_dim)
        src, dst = edge_index[0], edge_index[1]
        logits = ((h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1))
        logits = F.leaky_relu(logits, self.negative_slope)
        alpha = _edge_softmax(logits, dst, n)
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros(n, self.heads, self.out_dim, dtype=x.dtype, device=x.device)
        out.index_add_(0, dst, msg)
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias, alpha


def _edge_softmax(logits: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over the incoming edges of every destination node."""
    heads = logits.size(1)
    max_per_node = torch.zeros(n, heads, dtype=logits.dtype, device=logits.device)
    max_per_node = max_per_node.scatter_reduce(
        0, dst.unsqueeze(-1).expand_as(logits), logits, reduce="amax", include_self=False)
    exp = torch.exp(logits - max_per_node[dst])
    denom = torch.zeros(n, heads, dtype=logits.dtype, device=logits.device)
    denom.index_add_(0, dst, exp)
    return exp / (denom[dst] + 1e-16)


class GATEncoder(nn.Module):
    """Stacked GAT layers mapping multi-modal node features to embeddings.

    Input features are first projected to the common dimension d, matching the
    three-block concatenation of the node feature vector.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 256, heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.0, projection_dim: int = 256):
        super().__init__()
        self.project = nn.Linear(in_dim, projection_dim)
        dims_in = projection_dim
        layers: List[GATLayer] = []
        for layer in range(n_layers):
            last = layer == n_layers - 1
            layers.append(GATLayer(dims_in, out_dim if last else hidden_dim, heads=heads,
                                   concat=not last, dropout=dropout))
            dims_in = hidden_dim * heads
        self.layers = nn.ModuleList(layers)
        self.out_dim = out_dim
        self._attention: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attention: bool = False):
        self._attention = []
        h = self.project(x)
        for i, layer in enumerate(self.layers):
            h, alpha = layer(h, edge_index)
            self._attention.append(alpha.detach())
            if i < len(self.layers) - 1:
                h = F.elu(h)
        return (h, list(self._attention)) if return_attention else h

    @property
    def attention(self) -> List[torch.Tensor]:
        """Attention coefficients of the most recent forward pass, per layer."""
        return list(self._attention)


class ProjectionHead(nn.Module):
    """Two-layer head used only during contrastive pre-training."""

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

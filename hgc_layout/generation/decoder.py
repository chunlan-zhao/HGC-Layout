"""Transformer decoder that cross-attends to the partial scene graph.

Each placement query attends causally over the tokens of entities already
placed in the group and cross-attends to the node embeddings of the partial
graph. The cross-attention weights of every layer are retained so that the
neighbours that influenced a placement can be inspected afterwards.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hgc_layout.generation.heads import GaussianMixtureHead, VonMisesHead


class DecoderLayer(nn.Module):
    """Self-attention over placed tokens, cross-attention to graph memory, FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_dim, d_model))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(tgt)
        sa, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        tgt = tgt + self.drop(sa)
        h = self.norm2(tgt)
        ca, weights = self.cross_attn(h, memory, memory, need_weights=True,
                                      average_attn_weights=True)
        tgt = tgt + self.drop(ca)
        tgt = tgt + self.drop(self.ff(self.norm3(tgt)))
        return tgt, weights


class GraphConditionedDecoder(nn.Module):
    """Maps entity queries and graph memory to coordinate and orientation parameters."""

    def __init__(self, embed_dim: int = 256, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.10,
                 n_components: int = 5, n_attributes: int = 8):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim + n_attributes, d_model)
        self.memory_proj = nn.Linear(embed_dim, d_model)
        self.context_proj = nn.Linear(embed_dim, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ff_dim, dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.coord_head = GaussianMixtureHead(d_model, n_components)
        self.orient_head = VonMisesHead(d_model + 2)
        self._cross_attention: List[torch.Tensor] = []

    def forward(self, query: torch.Tensor, memory: torch.Tensor, context: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """query: (B, T, embed+attr); memory: (B, N, embed); context: (B, embed)."""
        h = self.query_proj(query)
        mem = self.memory_proj(memory) + self.context_proj(context).unsqueeze(1)
        self._cross_attention = []
        for layer in self.layers:
            h, weights = layer(h, mem, causal_mask)
            self._cross_attention.append(weights.detach())
        h = self.norm(h)
        flat = h.reshape(-1, h.size(-1))
        coord = self.coord_head(flat)
        return {"hidden": h, "coord": coord}

    def orientation(self, hidden: torch.Tensor, position: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Orientation parameters conditioned on the placed coordinate."""
        flat = hidden.reshape(-1, hidden.size(-1))
        return self.orient_head(torch.cat([flat, position], dim=-1))

    @property
    def cross_attention(self) -> List[torch.Tensor]:
        """Cross-attention weights of the most recent forward pass, per layer."""
        return list(self._cross_attention)


def causal_mask(size: int, device: torch.device = None) -> torch.Tensor:
    """Upper-triangular additive mask preventing attention to later tokens."""
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

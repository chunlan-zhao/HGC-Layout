"""Graph contrastive pre-training with the NT-Xent objective.

Two independently corrupted views of the scene graph give two embeddings per
node; the pair is positive and the other nodes of the mini-batch are
negatives. The loss is the normalized temperature-scaled cross-entropy of
Eq. (1), computed cross-view over a mini-batch of B nodes.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hgc_layout.graph.augment import augment_view
from hgc_layout.graph.construction import GeoGraph
from hgc_layout.graph.gat import GATEncoder, ProjectionHead

log = logging.getLogger(__name__)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Eq. (1): cross-view NT-Xent over a mini-batch of node embeddings."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = z1 @ z2.t() / temperature
    target = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim, target)


def _to_tensor(graph: GeoGraph, device: torch.device):
    x = torch.as_tensor(graph.features, dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(graph.edge_index(), dtype=torch.long, device=device)
    return x, edge_index


def pretrain_encoder(graphs: List[GeoGraph], encoder: GATEncoder, epochs: int = 200,
                     batch_size: int = 10, temperature: float = 0.5, lr: float = 1e-3,
                     weight_decay: float = 1e-5, feature_dropout: float = 0.20,
                     edge_removal: float = 0.20, edge_addition: float = 0.05,
                     seed: int = 0, device: Optional[torch.device] = None,
                     log_every: int = 20) -> Dict[str, List[float]]:
    """Pre-train `encoder` on the training graphs; return the loss history."""
    device = device or torch.device("cpu")
    encoder.to(device).train()
    head = ProjectionHead(encoder.out_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                           lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    history: List[float] = []
    for epoch in range(epochs):
        epoch_losses = []
        for graph in graphs:
            v1 = augment_view(graph, rng, feature_dropout, edge_removal, edge_addition)
            v2 = augment_view(graph, rng, feature_dropout, edge_removal, edge_addition)
            x1, e1 = _to_tensor(v1, device)
            x2, e2 = _to_tensor(v2, device)
            h1, h2 = head(encoder(x1, e1)), head(encoder(x2, e2))
            n = min(batch_size, h1.size(0))
            idx = torch.as_tensor(rng.choice(h1.size(0), size=n, replace=False), device=device)
            loss = nt_xent_loss(h1[idx], h2[idx], temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        history.append(float(np.mean(epoch_losses)))
        if log_every and (epoch + 1) % log_every == 0:
            log.info("contrastive epoch %d/%d loss %.4f", epoch + 1, epochs, history[-1])
    encoder.eval()
    return {"loss": history}


def pretrain_reconstruction(graphs: List[GeoGraph], encoder: GATEncoder, epochs: int = 200,
                            lr: float = 1e-3, weight_decay: float = 1e-5, seed: int = 0,
                            device: Optional[torch.device] = None,
                            log_every: int = 20) -> Dict[str, List[float]]:
    """Ablation variant: train the encoder to reconstruct node features instead.

    No multi-view augmentation and no contrastive objective, so the encoder
    only has to preserve input information rather than become invariant to
    missing or noisy structure.
    """
    device = device or torch.device("cpu")
    encoder.to(device).train()
    decoder = torch.nn.Linear(encoder.out_dim, graphs[0].features.shape[1]).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=lr, weight_decay=weight_decay)
    torch.manual_seed(seed)
    history: List[float] = []
    for epoch in range(epochs):
        losses = []
        for graph in graphs:
            x, edge_index = _to_tensor(graph, device)
            recon = decoder(encoder(x, edge_index))
            loss = torch.nn.functional.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))
        if log_every and (epoch + 1) % log_every == 0:
            log.info("reconstruction epoch %d/%d loss %.4f", epoch + 1, epochs, history[-1])
    encoder.eval()
    return {"loss": history}


@torch.no_grad()
def embed_graph(graph: GeoGraph, encoder: GATEncoder, device: Optional[torch.device] = None,
                return_attention: bool = False):
    """Node embeddings of one graph, optionally with per-layer attention."""
    device = device or torch.device("cpu")
    encoder.eval()
    x, edge_index = _to_tensor(graph, device)
    out = encoder(x, edge_index, return_attention=return_attention)
    if return_attention:
        h, attention = out
        return h, attention, edge_index
    return out

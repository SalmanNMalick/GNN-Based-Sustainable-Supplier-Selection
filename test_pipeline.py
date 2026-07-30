"""GNN layers and the SupplierGNN model.

Implemented with dense adjacency message passing in pure PyTorch so the
project runs anywhere PyTorch runs (no PyTorch Geometric requirement).
For very large graphs (>50k suppliers), swap ``DenseSAGELayer`` /
``DenseGATLayer`` for ``torch_geometric.nn.SAGEConv`` / ``GATConv`` with a
sparse ``edge_index`` — the forward signatures map one-to-one.

Author: Salman Nawaz Malik
License: MIT
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CRITERIA, ModelConfig


class DenseSAGELayer(nn.Module):
    """GraphSAGE (mean aggregator) over a row-normalised dense adjacency.

    h_i' = W_self · h_i + W_nbr · (A_norm h)_i
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nbr = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.lin_self(x) + self.lin_nbr(adj @ x)


class DenseGATLayer(nn.Module):
    """Multi-head graph attention over a dense adjacency mask.

    Attention scores follow Velickovic et al. (2018); the adjacency is used
    as a mask (entries == 0 receive -inf before softmax) and heads are
    concatenated.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.d_head = out_dim // heads
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, self.d_head))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.d_head))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.proj(x).view(n, self.heads, self.d_head)         # (N, H, d)
        e_src = (h * self.attn_src).sum(-1)                        # (N, H)
        e_dst = (h * self.attn_dst).sum(-1)                        # (N, H)
        # e[i, j, h] = leakyrelu(e_src[i, h] + e_dst[j, h])
        e = F.leaky_relu(e_src.unsqueeze(1) + e_dst.unsqueeze(0), 0.2)  # (N, N, H)
        mask = (adj > 0).unsqueeze(-1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=1)                            # over neighbours j
        alpha = torch.nan_to_num(alpha, nan=0.0)
        out = torch.einsum("ijh,jhd->ihd", alpha, h)               # (N, H, d)
        return out.reshape(n, self.heads * self.d_head)


class SupplierGNN(nn.Module):
    """Encoder + multi-criteria scoring heads.

    Forward returns:
      embeddings:       (N, hidden_dim) node embeddings
      criterion_scores: (N, n_criteria) sigmoid scores in [0, 1], ordered as
                        ``config.CRITERIA`` (all benefit-oriented).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 n_criteria: int = len(CRITERIA), cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig(hidden_dim=hidden_dim)
        layer = (lambda i, o: DenseGATLayer(i, o, cfg.gat_heads)) \
            if cfg.layer_type == "gat" else DenseSAGELayer

        dims = [in_dim] + [cfg.hidden_dim] * cfg.n_layers
        self.layers = nn.ModuleList(layer(dims[i], dims[i + 1])
                                    for i in range(cfg.n_layers))
        self.dropout = nn.Dropout(cfg.dropout)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                          nn.ReLU(),
                          nn.Linear(cfg.hidden_dim // 2, 1))
            for _ in range(n_criteria)
        )

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        z = self.encode(x, adj)
        scores = torch.cat([torch.sigmoid(head(z)) for head in self.heads], dim=1)
        return z, scores


class DGIDiscriminator(nn.Module):
    """Deep Graph Infomax discriminator: scores (embedding, summary) pairs."""

    def __init__(self, dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        s = summary.expand_as(z)
        return self.bilinear(z, s).squeeze(-1)

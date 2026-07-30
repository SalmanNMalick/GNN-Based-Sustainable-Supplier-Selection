"""Dataset handling and supplier-graph construction.

Column contract (case-sensitive) for the input DataFrame / CSV
---------------------------------------------------------------
supplier_id        str   unique identifier
region             str   e.g. "South Asia", "EU"
category           str   product/service category
unit_cost          float currency per unit               (cost, minimise)
logistics_cost     float currency per unit               (cost, minimise)
lead_time_days     float average lead time               (time, minimise)
on_time_rate       float 0-1 on-time delivery            (time, maximise)
acceptance_rate    float 0-1 lot acceptance              (quality, maximise)
defect_ppm         float defects per million             (quality, minimise)
iso9001            int   0/1 quality certification       (quality)
fill_rate          float 0-1 order fill rate             (reliability, maximise)
financial_score    float 0-100 financial health          (reliability, maximise)
geo_risk           float 0-1 geographic/political risk   (reliability, minimise)
co2e_per_unit      float kg CO2e per unit                (net-zero, minimise)
renewable_share    float 0-1 renewable energy share      (net-zero, maximise)
iso14001           int   0/1 environmental cert          (net-zero)
sbti               int   0/1 science-based targets set   (net-zero)
scope3_disclosed   int   0/1 discloses scope-3           (net-zero)

Author: Salman Nawaz Malik
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .config import CRITERIA
from .utils import minmax_normalise

REQUIRED_COLUMNS: List[str] = [
    "supplier_id", "region", "category",
    "unit_cost", "logistics_cost", "lead_time_days", "on_time_rate",
    "acceptance_rate", "defect_ppm", "iso9001",
    "fill_rate", "financial_score", "geo_risk",
    "co2e_per_unit", "renewable_share", "iso14001", "sbti", "scope3_disclosed",
]

# Which raw columns feed each criterion index, and their direction within it.
CRITERION_FEATURES: Dict[str, List[tuple]] = {
    "cost":        [("unit_cost", -1), ("logistics_cost", -1)],
    "time":        [("lead_time_days", -1), ("on_time_rate", +1)],
    "quality":     [("acceptance_rate", +1), ("defect_ppm", -1), ("iso9001", +1)],
    "reliability": [("fill_rate", +1), ("financial_score", +1), ("geo_risk", -1)],
    "net_zero":    [("co2e_per_unit", -1), ("renewable_share", +1),
                    ("iso14001", +1), ("sbti", +1), ("scope3_disclosed", +1)],
}


@dataclass
class SupplierGraph:
    """Dense-graph container consumed by the GNN.

    Attributes
    ----------
    x:            (N, F) float tensor of normalised node features.
    adj:          (N, N) float tensor, row-normalised weighted adjacency
                  with self-loops.
    criterion_y:  (N, 5) engineered per-criterion target indices in [0, 1],
                  ordered as ``config.CRITERIA`` (all as benefit: higher=better).
    supplier_ids: list of supplier identifiers aligned to node order.
    feature_names: list of feature column names aligned to ``x`` columns.
    """

    x: torch.Tensor
    adj: torch.Tensor
    criterion_y: torch.Tensor
    supplier_ids: List[str]
    feature_names: List[str]


def load_suppliers(path: str) -> pd.DataFrame:
    """Load and validate a supplier CSV."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input file missing required columns: {missing}")
    if df["supplier_id"].duplicated().any():
        raise ValueError("duplicate supplier_id values found")
    return df.reset_index(drop=True)


def engineered_criterion_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw columns into one benefit-oriented index per criterion.

    Each raw feature is min-max normalised, flipped if it is a cost-type
    feature, then averaged within its criterion. Result columns are in
    [0, 1] where higher is always better.
    """
    out = {}
    for criterion, feats in CRITERION_FEATURES.items():
        cols = []
        for col, direction in feats:
            v = minmax_normalise(df[col].to_numpy(dtype=float))
            if direction < 0:
                v = 1.0 - v
            cols.append(v)
        out[criterion] = np.mean(np.column_stack(cols), axis=1)
    return pd.DataFrame(out, index=df.index)


class SupplierGraphBuilder:
    """Build a weighted supplier graph.

    Edges combine:
      * categorical affinity — same region and/or same category;
      * k-NN similarity in normalised numeric feature space.

    Parameters
    ----------
    knn_k: number of nearest neighbours per node (feature-space edges).
    w_region, w_category, w_knn: edge-weight contributions.
    """

    def __init__(self, knn_k: int = 8, w_region: float = 0.3,
                 w_category: float = 0.3, w_knn: float = 0.4):
        self.knn_k = knn_k
        self.w_region = w_region
        self.w_category = w_category
        self.w_knn = w_knn

    def build(self, df: pd.DataFrame) -> SupplierGraph:
        df = df.reset_index(drop=True)
        n = len(df)
        numeric_cols = [c for c in REQUIRED_COLUMNS
                        if c not in ("supplier_id", "region", "category")]

        x_num = np.column_stack(
            [minmax_normalise(df[c].to_numpy(dtype=float)) for c in numeric_cols]
        )

        # One-hot region & category appended to node features.
        region_oh = pd.get_dummies(df["region"], prefix="region")
        cat_oh = pd.get_dummies(df["category"], prefix="cat")
        x = np.column_stack([x_num, region_oh.to_numpy(float), cat_oh.to_numpy(float)])
        feature_names = numeric_cols + list(region_oh.columns) + list(cat_oh.columns)

        adj = np.zeros((n, n), dtype=float)

        # Categorical affinity edges.
        region = df["region"].to_numpy()
        category = df["category"].to_numpy()
        same_region = (region[:, None] == region[None, :])
        same_cat = (category[:, None] == category[None, :])
        adj += self.w_region * same_region + self.w_category * same_cat

        # k-NN edges in numeric feature space (cosine similarity).
        norms = np.linalg.norm(x_num, axis=1, keepdims=True) + 1e-12
        sim = (x_num / norms) @ (x_num / norms).T
        np.fill_diagonal(sim, -np.inf)
        k = min(self.knn_k, n - 1)
        for i in range(n):
            nbrs = np.argpartition(-sim[i], k)[:k]
            for j in nbrs:
                w = self.w_knn * max(sim[i, j], 0.0)
                adj[i, j] += w
                adj[j, i] += w  # symmetrise

        np.fill_diagonal(adj, 0.0)
        adj = adj + np.eye(n)                       # self-loops
        adj = adj / adj.sum(axis=1, keepdims=True)  # row-normalise

        crit = engineered_criterion_indices(df)[list(CRITERIA)].to_numpy(float)

        return SupplierGraph(
            x=torch.tensor(x, dtype=torch.float32),
            adj=torch.tensor(adj, dtype=torch.float32),
            criterion_y=torch.tensor(crit, dtype=torch.float32),
            supplier_ids=df["supplier_id"].astype(str).tolist(),
            feature_names=feature_names,
        )

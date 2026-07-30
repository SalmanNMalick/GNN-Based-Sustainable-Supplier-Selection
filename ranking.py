"""Configuration objects for the supplier-selection pipeline.

Author: Salman Nawaz Malik
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# Canonical criteria evaluated by the framework.
# direction: +1 = benefit (higher is better), -1 = cost (lower is better).
CRITERIA: Dict[str, int] = {
    "cost": -1,
    "time": -1,
    "quality": +1,
    "reliability": +1,
    "net_zero": +1,
}


@dataclass
class CriteriaConfig:
    """User-facing decision configuration.

    Parameters
    ----------
    weights:
        Subjective importance per criterion. They are re-normalised to sum
        to 1 and blended with objective entropy weights (see ``ranking.py``).
    subjective_blend:
        Fraction of the final weight taken from user weights (the remainder
        comes from entropy weights). 0.5 = equal blend.
    carbon_intensity_ceiling:
        Hard net-zero constraint: suppliers whose ``co2e_per_unit`` exceeds
        this value (kg CO2e / unit) are excluded before ranking. ``None``
        disables the constraint.
    min_renewable_share:
        Optional hard floor on renewable-energy share (0-1). ``None`` disables.
    require_sbti:
        If True, only suppliers with science-based targets are eligible.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "cost": 0.25,
            "time": 0.15,
            "quality": 0.20,
            "reliability": 0.15,
            "net_zero": 0.25,
        }
    )
    subjective_blend: float = 0.5
    carbon_intensity_ceiling: Optional[float] = None
    min_renewable_share: Optional[float] = None
    require_sbti: bool = False

    def normalised_weights(self) -> Dict[str, float]:
        missing = set(CRITERIA) - set(self.weights)
        if missing:
            raise ValueError(f"weights missing criteria: {sorted(missing)}")
        total = sum(self.weights[c] for c in CRITERIA)
        if total <= 0:
            raise ValueError("criterion weights must sum to a positive value")
        return {c: self.weights[c] / total for c in CRITERIA}


@dataclass
class ModelConfig:
    """GNN hyper-parameters."""

    hidden_dim: int = 64
    n_layers: int = 2
    layer_type: str = "sage"      # "sage" | "gat"
    gat_heads: int = 4
    dropout: float = 0.2


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 5e-4
    dgi_loss_weight: float = 0.5   # contrastive (structure) vs. criterion heads
    seed: int = 42
    verbose_every: int = 50

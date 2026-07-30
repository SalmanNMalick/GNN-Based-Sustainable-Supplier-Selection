"""Explainable multi-criteria decision layer.

Pipeline: hard net-zero constraints → entropy weighting blended with user
weights → TOPSIS closeness coefficient → final ranking.

All criterion scores arriving here are benefit-oriented in [0, 1]
(higher = better), including cost and time, which were flipped during
feature engineering.

Author: Salman Nawaz Malik
License: MIT
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import CRITERIA, CriteriaConfig


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Objective weights via Shannon entropy of each criterion column.

    Criteria whose values differentiate suppliers strongly (low entropy)
    receive higher weight.
    """
    m = matrix + 1e-12
    p = m / m.sum(axis=0, keepdims=True)
    n = matrix.shape[0]
    entropy = -(p * np.log(p)).sum(axis=0) / np.log(n)
    diversity = 1.0 - entropy
    if diversity.sum() <= 0:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    return diversity / diversity.sum()


def topsis(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """TOPSIS closeness coefficients for a benefit-oriented decision matrix."""
    norm = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-12)
    v = norm * weights
    ideal, anti = v.max(axis=0), v.min(axis=0)
    d_pos = np.linalg.norm(v - ideal, axis=1)
    d_neg = np.linalg.norm(v - anti, axis=1)
    return d_neg / (d_pos + d_neg + 1e-12)


class MCDMRanker:
    def __init__(self, config: CriteriaConfig | None = None):
        self.config = config or CriteriaConfig()

    # ------------------------------------------------------------------ #
    def _apply_constraints(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean eligibility mask and record exclusion reasons."""
        cfg = self.config
        reasons = pd.Series("", index=df.index, dtype=object)
        ok = pd.Series(True, index=df.index)

        if cfg.carbon_intensity_ceiling is not None:
            bad = df["co2e_per_unit"] > cfg.carbon_intensity_ceiling
            reasons[bad] += f"CO2e/unit > {cfg.carbon_intensity_ceiling}; "
            ok &= ~bad
        if cfg.min_renewable_share is not None:
            bad = df["renewable_share"] < cfg.min_renewable_share
            reasons[bad] += f"renewable share < {cfg.min_renewable_share}; "
            ok &= ~bad
        if cfg.require_sbti:
            bad = df["sbti"] != 1
            reasons[bad] += "no science-based targets; "
            ok &= ~bad

        self._exclusion_reasons = reasons.str.strip("; ")
        return ok

    # ------------------------------------------------------------------ #
    def rank(self, df: pd.DataFrame, criterion_scores: np.ndarray) -> pd.DataFrame:
        """Rank suppliers.

        Parameters
        ----------
        df:
            Original supplier DataFrame (row order must match node order).
        criterion_scores:
            (N, 5) GNN-refined benefit scores ordered as ``config.CRITERIA``.

        Returns
        -------
        DataFrame sorted by rank with per-criterion scores, TOPSIS closeness,
        eligibility flag and exclusion reason.
        """
        names = list(CRITERIA)
        scores = pd.DataFrame(criterion_scores, columns=names, index=df.index)

        eligible = self._apply_constraints(df)

        user_w = np.array([self.config.normalised_weights()[c] for c in names])
        if eligible.sum() >= 2:
            obj_w = entropy_weights(scores.loc[eligible, names].to_numpy())
        else:
            obj_w = np.full(len(names), 1.0 / len(names))
        b = self.config.subjective_blend
        final_w = b * user_w + (1 - b) * obj_w
        final_w = final_w / final_w.sum()
        self.final_weights: Dict[str, float] = dict(zip(names, final_w))

        closeness = np.full(len(df), np.nan)
        if eligible.sum() >= 2:
            closeness[eligible.to_numpy()] = topsis(
                scores.loc[eligible, names].to_numpy(), final_w
            )
        elif eligible.sum() == 1:
            closeness[eligible.to_numpy()] = 1.0

        out = pd.concat(
            [df[["supplier_id", "region", "category"]].reset_index(drop=True),
             scores.reset_index(drop=True)],
            axis=1,
        )
        out["topsis_closeness"] = closeness
        out["eligible"] = eligible.to_numpy()
        out["exclusion_reason"] = self._exclusion_reasons.to_numpy()
        out = out.sort_values(
            by=["eligible", "topsis_closeness"], ascending=[False, False]
        ).reset_index(drop=True)
        out.insert(0, "rank", np.where(out["eligible"],
                                       out["eligible"].cumsum(), np.nan))
        return out

    # ------------------------------------------------------------------ #
    def report(self, ranking: pd.DataFrame, top_k: int = 10) -> str:
        """Human-readable summary of the decision."""
        lines = ["=" * 78,
                 "SUPPLIER SELECTION REPORT — GNN + entropy-TOPSIS",
                 "=" * 78,
                 "Effective criterion weights (user ⊕ entropy blend):"]
        for c, w in self.final_weights.items():
            lines.append(f"  {c:<12s} {w:6.3f}")
        n_excl = int((~ranking["eligible"]).sum())
        lines.append(f"\nEligible suppliers: {int(ranking['eligible'].sum())}  "
                     f"| excluded by net-zero/other constraints: {n_excl}")
        lines.append("-" * 78)
        cols = ["rank", "supplier_id", "cost", "time", "quality",
                "reliability", "net_zero", "topsis_closeness"]
        head = ranking[ranking["eligible"]].head(top_k)[cols]
        lines.append(head.to_string(index=False,
                                    float_format=lambda v: f"{v:.3f}"))
        if n_excl:
            lines.append("-" * 78)
            lines.append("Excluded suppliers:")
            for _, r in ranking[~ranking["eligible"]].iterrows():
                lines.append(f"  {r['supplier_id']}: {r['exclusion_reason']}")
        lines.append("=" * 78)
        return "\n".join(lines)

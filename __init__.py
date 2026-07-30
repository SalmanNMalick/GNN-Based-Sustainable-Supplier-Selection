"""Generate a synthetic-but-realistic supplier dataset.

Suppliers are drawn from latent archetypes (e.g. "green premium",
"low-cost high-carbon", "balanced") so criteria are realistically
correlated — cheap suppliers tend to have longer lead times and higher
carbon intensity, sustainable suppliers command a price premium, etc.

Usage:
    python examples/generate_data.py --n-suppliers 100 --out data/suppliers.csv

Author: Salman Nawaz Malik
License: MIT
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

REGIONS = ["South Asia", "East Asia", "EU", "North America", "MENA"]
CATEGORIES = ["Raw Materials", "Components", "Packaging", "Logistics", "Services"]

# archetype: (cost_mu, lead_mu, quality_mu, co2_mu, renewable_mu, prob)
ARCHETYPES = {
    "green_premium":   (1.30, 0.9, 0.92, 0.8, 0.70, 0.20),
    "low_cost_carbon": (0.75, 1.3, 0.80, 2.8, 0.15, 0.30),
    "balanced":        (1.00, 1.0, 0.88, 1.6, 0.40, 0.35),
    "unreliable":      (0.90, 1.6, 0.70, 2.2, 0.20, 0.15),
}


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    probs = np.array([ARCHETYPES[a][5] for a in names])
    rows = []
    for i in range(n):
        a = names[rng.choice(len(names), p=probs / probs.sum())]
        cost_mu, lead_mu, q_mu, co2_mu, ren_mu, _ = ARCHETYPES[a]
        quality = np.clip(rng.normal(q_mu, 0.05), 0.5, 0.999)
        rows.append({
            "supplier_id": f"SUP-{i+1:04d}",
            "region": rng.choice(REGIONS),
            "category": rng.choice(CATEGORIES),
            "unit_cost": round(max(rng.normal(100 * cost_mu, 12), 20), 2),
            "logistics_cost": round(max(rng.normal(12 * cost_mu, 3), 1), 2),
            "lead_time_days": round(max(rng.normal(21 * lead_mu, 5), 2), 1),
            "on_time_rate": round(np.clip(rng.normal(quality, 0.06), 0.4, 1.0), 3),
            "acceptance_rate": round(quality, 3),
            "defect_ppm": round(max(rng.normal((1 - quality) * 20000, 800), 10), 0),
            "iso9001": int(rng.random() < quality),
            "fill_rate": round(np.clip(rng.normal(quality, 0.05), 0.4, 1.0), 3),
            "financial_score": round(np.clip(rng.normal(65 + 20 * (quality - 0.8) * 5, 10), 10, 100), 1),
            "geo_risk": round(np.clip(rng.beta(2, 5), 0, 1), 3),
            "co2e_per_unit": round(max(rng.normal(co2_mu, 0.35), 0.1), 3),
            "renewable_share": round(np.clip(rng.normal(ren_mu, 0.12), 0, 1), 3),
            "iso14001": int(rng.random() < ren_mu + 0.15),
            "sbti": int(rng.random() < ren_mu),
            "scope3_disclosed": int(rng.random() < ren_mu + 0.10),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-suppliers", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/suppliers.csv")
    args = ap.parse_args()

    df = generate(args.n_suppliers, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} suppliers -> {args.out}")

"""End-to-end demo: load data → build graph → train GNN → rank suppliers.

Usage:
    python examples/demo.py --data data/suppliers.csv --top-k 10 \
        --carbon-ceiling 2.5 --net-zero-weight 0.25

Author: Salman Nawaz Malik
License: MIT
"""
from __future__ import annotations

import argparse
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from supplier_gnn import (
    CriteriaConfig, GNNTrainer, MCDMRanker, SupplierGNN,
    SupplierGraphBuilder, load_suppliers,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/suppliers.csv")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--layer", choices=["sage", "gat"], default="sage")
    ap.add_argument("--carbon-ceiling", type=float, default=None,
                    help="Hard cap on kg CO2e/unit (net-zero constraint)")
    ap.add_argument("--net-zero-weight", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="Optional CSV path for ranking")
    args = ap.parse_args()

    df = load_suppliers(args.data)
    print(f"Loaded {len(df)} suppliers from {args.data}")

    graph = SupplierGraphBuilder(knn_k=8).build(df)
    print(f"Graph: {graph.x.shape[0]} nodes, {graph.x.shape[1]} features")

    from supplier_gnn.config import ModelConfig
    model = SupplierGNN(in_dim=graph.x.shape[1],
                        cfg=ModelConfig(layer_type=args.layer))
    trainer = GNNTrainer(model, epochs=args.epochs)
    scores = trainer.fit_predict(graph)

    # Redistribute weights so net-zero gets the requested share.
    nz = args.net_zero_weight
    rest = (1 - nz)
    config = CriteriaConfig(
        weights={"cost": rest * 0.33, "time": rest * 0.20,
                 "quality": rest * 0.27, "reliability": rest * 0.20,
                 "net_zero": nz},
        carbon_intensity_ceiling=args.carbon_ceiling,
    )
    ranker = MCDMRanker(config)
    ranking = ranker.rank(df, scores)
    print(ranker.report(ranking, top_k=args.top_k))

    if args.out:
        ranking.to_csv(args.out, index=False)
        print(f"Full ranking written to {args.out}")


if __name__ == "__main__":
    main()

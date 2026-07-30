# GNN-Based Multi-Criteria Supplier Selection with Net-Zero Alignment

A Graph Neural Network (GNN) framework for intelligent supplier selection that jointly evaluates **cost, delivery time, quality, reliability, and net-zero (carbon/sustainability) performance** by modelling the buyer–supplier ecosystem as a graph.

**Author:** Salman Nawaz Malik

---

## Why a GNN for Supplier Selection?

Classical supplier selection methods (AHP, TOPSIS, linear weighted scoring) treat each supplier as an **independent row in a table**. In reality, suppliers are embedded in a network:

- Suppliers share sub-suppliers, logistics corridors, and geographic risk zones.
- A supplier's *effective* carbon footprint depends on its upstream tier-2/tier-3 network.
- Disruption risk propagates through shared dependencies.

A GNN captures these **relational effects** by passing messages between connected nodes, so each supplier's learned embedding reflects not only its own attributes but also the attributes and risks of its neighbourhood. The framework then combines the learned GNN scores with a transparent multi-criteria decision layer (entropy-weighted TOPSIS) so the final ranking is both *learned* and *explainable*.

## Architecture

```
 ┌─────────────────────┐
 │  Supplier dataset    │  cost, lead time, quality, defect rate,
 │  (CSV / generated)   │  CO₂e intensity, renewable %, SBTi status ...
 └─────────┬───────────┘
           │
 ┌─────────▼───────────┐
 │  Graph builder       │  nodes = suppliers (+ buyer, categories)
 │  (data.py)           │  edges = shared region / category / logistics,
 │                      │  k-NN similarity in feature space
 └─────────┬───────────┘
           │
 ┌─────────▼───────────┐
 │  GNN encoder         │  GraphSAGE / GAT layers (pure PyTorch,
 │  (models.py)         │  no PyG dependency required)
 └─────────┬───────────┘
           │
 ┌─────────▼───────────┐
 │  Multi-head scorer   │  one head per criterion:
 │  (models.py)         │  cost · time · quality · reliability · net-zero
 └─────────┬───────────┘
           │
 ┌─────────▼───────────┐
 │  Decision layer      │  entropy weights + TOPSIS over GNN-refined
 │  (ranking.py)        │  criterion scores → final supplier ranking
 └─────────────────────┘
```

## Evaluation Criteria

| Criterion | Direction | Example raw features |
|---|---|---|
| **Cost** | minimise | unit price, logistics cost, payment terms |
| **Time** | minimise | lead time (days), on-time delivery % |
| **Quality** | maximise | acceptance rate, defect PPM, ISO 9001 |
| **Reliability / Risk** | maximise | fill rate, financial health, geographic risk |
| **Net-Zero** | maximise | CO₂e per unit, renewable energy %, ISO 14001, science-based targets (SBTi), scope-3 disclosure |

The **net-zero criterion** is a first-class citizen: the decision layer supports a *sustainability floor* (hard constraint — e.g. exclude suppliers above a carbon-intensity threshold) as well as a tunable weight so procurement teams can trade off price against decarbonisation targets.

## Installation

```bash
git clone https://github.com/<your-username>/gnn-supplier-selection.git
cd gnn-supplier-selection
pip install -r requirements.txt
```

Requires Python ≥ 3.9. The GNN layers are implemented in **pure PyTorch** (dense adjacency message passing), so **PyTorch Geometric is NOT required** — this keeps installation friction-free on Windows and CPU-only machines.

## Quick Start

```bash
# 1. Generate a synthetic supplier dataset (100 suppliers, reproducible)
python examples/generate_data.py --n-suppliers 100 --out data/suppliers.csv

# 2. Train the GNN and rank suppliers
python examples/demo.py --data data/suppliers.csv --top-k 10
```

Or in Python:

```python
from supplier_gnn import (
    SupplierGraphBuilder, SupplierGNN, GNNTrainer, MCDMRanker, CriteriaConfig
)
import pandas as pd

df = pd.read_csv("data/suppliers.csv")

# Build the supplier graph
builder = SupplierGraphBuilder(knn_k=8)
graph = builder.build(df)

# Train a self-supervised GNN encoder + multi-criteria heads
model = SupplierGNN(in_dim=graph.x.shape[1], hidden_dim=64, n_criteria=5)
trainer = GNNTrainer(model, epochs=200, lr=1e-3)
criterion_scores = trainer.fit_predict(graph)

# Explainable multi-criteria ranking (entropy-weighted TOPSIS)
config = CriteriaConfig(
    weights={"cost": 0.25, "time": 0.15, "quality": 0.20,
             "reliability": 0.15, "net_zero": 0.25},
    carbon_intensity_ceiling=2.5,   # hard net-zero floor (kg CO2e/unit)
)
ranker = MCDMRanker(config)
ranking = ranker.rank(df, criterion_scores)
print(ranking.head(10))
```

## Repository Layout

```
gnn-supplier-selection/
├── README.md
├── LICENSE                      # MIT — © Salman Nawaz Malik
├── requirements.txt
├── pyproject.toml
├── supplier_gnn/
│   ├── __init__.py
│   ├── config.py                # criteria weights, thresholds, hyper-params
│   ├── data.py                  # dataset loading + graph construction
│   ├── models.py                # GraphSAGE / GAT layers, SupplierGNN
│   ├── train.py                 # self-supervised trainer (DGI-style + criterion heads)
│   ├── ranking.py               # entropy weights, TOPSIS, net-zero constraints
│   └── utils.py                 # normalisation, seeding, report helpers
├── examples/
│   ├── generate_data.py         # synthetic-but-realistic supplier data generator
│   └── demo.py                  # end-to-end pipeline with printed report
├── tests/
│   └── test_pipeline.py         # smoke tests for graph build, model, ranking
└── data/                        # (generated) supplier CSVs
```

## Methodology Notes

1. **Graph construction** — suppliers are connected if they (a) share a region or product category, or (b) are among each other's *k* nearest neighbours in normalised feature space. Edge weights blend categorical affinity with feature similarity.
2. **Self-supervised training** — labelled "best supplier" data rarely exists, so the encoder is trained with a Deep-Graph-Infomax-style contrastive objective (real graph vs. feature-shuffled corrupted graph), plus per-criterion regression heads supervised against the engineered criterion indices. This lets the GNN *refine* raw criterion scores using network context (e.g. penalising a low-carbon supplier whose neighbourhood is high-risk).
3. **Decision layer** — GNN-refined criterion scores go through entropy weighting (objective) blended with user weights (subjective), then TOPSIS closeness coefficients produce the final ranking. Hard net-zero constraints are applied before ranking.
4. **Explainability** — the output report shows each supplier's per-criterion score, TOPSIS closeness, applied weights, and whether any sustainability constraint excluded it.

## Extending

- Swap the synthetic generator for your ERP/procurement extract — see the column contract at the top of `supplier_gnn/data.py`.
- Add criteria by extending `CriteriaConfig.CRITERIA` and the feature map in `data.py`.
- If you install PyTorch Geometric, `models.py` documents how to swap the dense layers for `SAGEConv`/`GATConv` for very large graphs.

## Citation

If you use this framework in academic work:

```bibtex
@software{malik2026gnnsupplier,
  author  = {Malik, Salman Nawaz},
  title   = {GNN-Based Multi-Criteria Supplier Selection with Net-Zero Alignment},
  year    = {2026},
  url     = {https://github.com/<your-username>/gnn-supplier-selection}
}
```

## License

MIT License — © 2026 Salman Nawaz Malik. See [LICENSE](LICENSE).

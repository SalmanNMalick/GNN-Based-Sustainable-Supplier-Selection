"""GNN-based multi-criteria supplier selection with net-zero alignment.

Author: Salman Nawaz Malik
License: MIT
"""
from .config import CRITERIA, CriteriaConfig, ModelConfig, TrainConfig
from .data import SupplierGraph, SupplierGraphBuilder, load_suppliers
from .models import SupplierGNN
from .ranking import MCDMRanker
from .train import GNNTrainer

__version__ = "1.0.0"
__author__ = "Salman Nawaz Malik"

__all__ = [
    "CRITERIA", "CriteriaConfig", "ModelConfig", "TrainConfig",
    "SupplierGraph", "SupplierGraphBuilder", "load_suppliers",
    "SupplierGNN", "GNNTrainer", "MCDMRanker",
]

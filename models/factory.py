"""Factory for creating models from configuration."""
import torch

from models.model import UnifiedModel
from models.backbones.base import BackboneBase
from models.backbones.mpnn import MPNNBackbone
from models.backbones.gcn import GCNBackbone
from models.backbones.gat import GATBackbone
from models.backbones.gin import GINBackbone
from models.backbones.transformer import GraphTransformerBackbone, GPSBackbone
from models.backbones.schnet import SchNetBackbone
from models.backbones.dimenet import DimeNetPlusPlusBackbone
from models.heads.graph_head import GraphPredictionHead
from models.heads.node_head import NodePredictionHead


BACKBONE_MAP: dict[str, type[BackboneBase]] = {
    'mpnn': MPNNBackbone,
    'gcn': GCNBackbone,
    'gat': GATBackbone,
    'gin': GINBackbone,
    'transformer': GraphTransformerBackbone,
    'gps': GPSBackbone,
    # Equivariant models — require data.z (atomic numbers) and data.pos
    'schnet': SchNetBackbone,
    'dimenet++': DimeNetPlusPlusBackbone,
    # 'mace': MACEBackbone  — requires `pip install mace-torch`
}

HEAD_MAP = {
    'graph': GraphPredictionHead,
    'vector': GraphPredictionHead,  # vector targets also use graph-level head
    'node': NodePredictionHead,
}

# Default configurations for each backbone.
# All parameters correspond to the backbone __init__ kwargs.
_DEFAULT_BACKBONE_CFG: dict[str, dict] = {
    'mpnn':        {'node_dim': 64, 'edge_nn_dim': 128, 'mp_times': 3},
    'gcn':         {'node_dim': 64, 'num_layers': 3},
    'gat':         {'node_dim': 64, 'num_layers': 3, 'heads': 4},
    'gin':         {'node_dim': 64, 'num_layers': 3},
    'transformer': {'node_dim': 64, 'num_layers': 3, 'heads': 4},
    'gps':         {'node_dim': 64, 'num_layers': 3, 'heads': 4},
    'schnet':      {'hidden_channels': 128, 'num_interactions': 6},
    'dimenet++':   {'hidden_channels': 128, 'num_blocks': 4},
}

_DEFAULT_HEAD_CFG: dict[str, dict] = {
    'graph':  {'node_dim': 64, 'readout': 'set2set', 'processing_steps': 3},
    'vector': {'node_dim': 64, 'readout': 'set2set', 'processing_steps': 3},
    'node':   {'node_dim': 64},
}

# Equivariant models that use data.z / data.pos instead of data.x
EQUIVARIANT_BACKBONES = {'schnet', 'dimenet++', 'mace'}


def create_model(
    dataset,
    target_type: str,
    num_targets: int,
    backbone_name: str = 'mpnn',
    backbone_cfg: dict | None = None,
    head_cfg: dict | None = None,
    device: torch.device | None = None,
) -> UnifiedModel:
    """Create a UnifiedModel from configuration.

    Args:
        dataset: PyG dataset.
        target_type: 'graph', 'vector', or 'node'.
        num_targets: Number of prediction targets.
        backbone_name: Key in BACKBONE_MAP.
        backbone_cfg: Overrides for the backbone defaults.
        head_cfg: Overrides for the head defaults.
        device: If provided, the model is moved to this device.

    Returns:
        UnifiedModel ready for training.
    """
    if backbone_name not in BACKBONE_MAP:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Available: {list(BACKBONE_MAP.keys())}"
        )
    if target_type not in HEAD_MAP:
        raise ValueError(
            f"Unknown target_type '{target_type}'. Available: {list(HEAD_MAP.keys())}"
        )

    bb_cfg = dict(_DEFAULT_BACKBONE_CFG[backbone_name])
    if backbone_cfg:
        bb_cfg.update(backbone_cfg)

    hd_cfg = dict(_DEFAULT_HEAD_CFG[target_type])
    if head_cfg:
        hd_cfg.update(head_cfg)

    # Equivariant models don't need dataset for feature dims (they use z/pos),
    # so we pass None as dataset.
    backbone_dataset = None if backbone_name in EQUIVARIANT_BACKBONES else dataset
    backbone = BACKBONE_MAP[backbone_name](backbone_dataset, **bb_cfg)
    head = HEAD_MAP[target_type](backbone.output_dim, dataset, num_targets, **hd_cfg)
    model = UnifiedModel(backbone, head)

    if device is not None:
        model = model.to(device)
    return model

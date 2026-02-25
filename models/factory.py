"""Factory for creating models from configuration."""
import torch

from models.model import UnifiedModel
from models.backbones.base import BackboneBase
from models.backbones.mpnn import MPNNBackbone
from models.backbones.gcn import GCNBackbone
from models.backbones.gat import GATBackbone
from models.backbones.gin import GINBackbone
from models.backbones.transformer import GraphTransformerBackbone
from models.heads.graph_head import GraphPredictionHead
from models.heads.node_head import NodePredictionHead


BACKBONE_MAP: dict[str, type[BackboneBase]] = {
    'mpnn': MPNNBackbone,
    'gcn': GCNBackbone,
    'gat': GATBackbone,
    'gin': GINBackbone,
    'transformer': GraphTransformerBackbone,
}

HEAD_MAP = {
    'graph': GraphPredictionHead,
    'vector': GraphPredictionHead,  # vector targets use graph-level head
    'node': NodePredictionHead,
}

# Default backbone configurations (can be overridden via config)
_DEFAULT_BACKBONE_CFG: dict[str, dict] = {
    'mpnn': {'node_dim': 64, 'edge_nn_dim': 128, 'mp_times': 3},
    'gcn':  {'node_dim': 64, 'num_layers': 3},
    'gat':  {'node_dim': 64, 'num_layers': 3, 'heads': 4},
    'gin':  {'node_dim': 64, 'num_layers': 3},
    'transformer': {'node_dim': 64, 'num_layers': 3, 'heads': 4},
}

_DEFAULT_HEAD_CFG: dict[str, dict] = {
    'graph': {'node_dim': 64, 'readout': 'set2set', 'processing_steps': 3},
    'vector': {'node_dim': 64, 'readout': 'set2set', 'processing_steps': 3},
    'node':  {'node_dim': 64},
}


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
        backbone_cfg: Keyword arguments for the backbone constructor.
            Defaults to _DEFAULT_BACKBONE_CFG[backbone_name] if not provided.
        head_cfg: Keyword arguments for the head constructor.
            Defaults to _DEFAULT_HEAD_CFG[target_type] if not provided.
        device: If provided, the model is moved to this device.

    Returns:
        UnifiedModel ready for training.

    Raises:
        ValueError: If backbone_name or target_type is unrecognized.
    """
    if backbone_name not in BACKBONE_MAP:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Available: {list(BACKBONE_MAP.keys())}")
    if target_type not in HEAD_MAP:
        raise ValueError(f"Unknown target_type '{target_type}'. Available: {list(HEAD_MAP.keys())}")

    bb_cfg = dict(_DEFAULT_BACKBONE_CFG[backbone_name])
    if backbone_cfg:
        bb_cfg.update(backbone_cfg)

    hd_cfg = dict(_DEFAULT_HEAD_CFG[target_type])
    if head_cfg:
        hd_cfg.update(head_cfg)

    backbone = BACKBONE_MAP[backbone_name](dataset, **bb_cfg)
    head = HEAD_MAP[target_type](backbone.output_dim, dataset, num_targets, **hd_cfg)
    model = UnifiedModel(backbone, head)

    if device is not None:
        model = model.to(device)
    return model

"""Factory for creating models from configuration."""
import torch

from models.model import UnifiedModel
from models.backbones.base import BackboneBase
from models.backbones.mpnn import MPNNBackbone
from models.backbones.gcn import GCNBackbone, GATBackbone, GINBackbone
from models.heads.graph_head import GraphPredictionHead
from models.heads.node_head import NodePredictionHead


BACKBONE_MAP = {
    'mpnn': MPNNBackbone,
    'gcn': GCNBackbone,
    'gat': GATBackbone,
    'gin': GINBackbone,
}

HEAD_MAP = {
    'graph': GraphPredictionHead,
    'node': NodePredictionHead,
}


class ModelFactory:
    """Factory for creating models from configuration dictionaries."""

    @staticmethod
    def create_backbone(backbone_name: str, dataset, backbone_config: dict) -> BackboneBase:
        """Create a backbone from name and configuration.

        Args:
            backbone_name: Name of the backbone ('mpnn', 'gcn', 'gat', 'gin', etc.)
            dataset: PyG dataset
            backbone_config: Configuration dictionary for the backbone

        Returns:
            BackboneBase: Instantiated backbone

        Raises:
            ValueError: If backbone_name is not recognized
        """
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONE_MAP.keys())}")

        BackboneClass = BACKBONE_MAP[backbone_name]
        return BackboneClass(dataset, **backbone_config)

    @staticmethod
    def create_head(head_name: str, backbone_dim: int, dataset, num_targets: int,
                    head_config: dict):
        """Create a head from name and configuration.

        Args:
            head_name: Name of the head ('graph' or 'node')
            backbone_dim: Output dimension of the backbone
            dataset: PyG dataset
            num_targets: Number of prediction targets
            head_config: Configuration dictionary for the head

        Returns:
            Head module (GraphPredictionHead or NodePredictionHead)

        Raises:
            ValueError: If head_name is not recognized
        """
        if head_name not in HEAD_MAP:
            raise ValueError(f"Unknown head: {head_name}. Available: {list(HEAD_MAP.keys())}")

        HeadClass = HEAD_MAP[head_name]
        return HeadClass(backbone_dim, dataset, num_targets, **head_config)

    @staticmethod
    def create_model(dataset, target_type: str, num_targets: int,
                     backbone_name: str = 'mpnn', backbone_config: dict = None,
                     head_config: dict = None, device: torch.device = None) -> UnifiedModel:
        """Create a unified model from configuration.

        Args:
            dataset: PyG dataset
            target_type: Type of targets ('graph' or 'node')
            num_targets: Number of prediction targets
            backbone_name: Name of the backbone
            backbone_config: Configuration for the backbone (defaults provided if None)
            head_config: Configuration for the head (defaults provided if None)
            device: Device to move model to (optional)

        Returns:
            UnifiedModel: Fully instantiated model

        Raises:
            ValueError: If target_type is not recognized
        """
        if target_type not in HEAD_MAP:
            raise ValueError(f"Unknown target type: {target_type}. Available: {list(HEAD_MAP.keys())}")

        # Default configurations
        if backbone_config is None:
            backbone_config = {'dim_linear': 64, 'dim_conv': 64, 'mp_times': 2}
        if head_config is None:
            head_config = {'processing_steps': 1, 'dim_linear': 64}

        # Create backbone and head
        backbone = ModelFactory.create_backbone(backbone_name, dataset, backbone_config)
        head = ModelFactory.create_head(target_type, backbone.output_dim, dataset,
                                        num_targets, head_config)

        # Create unified model
        model = UnifiedModel(backbone, head)

        # Move to device if specified
        if device is not None:
            model = model.to(device)

        return model

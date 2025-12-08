"""
Models package for multimodal pathology segmentation

Contains:
- dual_encoder: Multimodal (HE + MIF) models
- single_modality: Single modality ablation models
- losses: Loss functions
- trainer: Training utilities
"""

# Import losses
from .losses import DiceFocalLoss, MSGELossMaps

# Import trainers
from .trainer import train_model, set_seed
from .trainer_he_only import train_he_only_model
from .trainer_mif_only import train_mif_only_model

# Import dual encoder models
from .dual_encoder import DualEncoderUNet, MultiModalPathologyUNet

# Single modality models
from .single_modality import HEOnlyUNet, MIFOnlyUNet

__all__ = [
    # Losses
    'DiceFocalLoss',
    'MSGELossMaps',
    
    # Multimodal models
    'DualEncoderUNet',
    'MultiModalPathologyUNet',
    
    # Single modality models
    'HEOnlyUNet',
    'MIFOnlyUNet',
    
    # Training utilities
    'train_model',              # For dual encoder (multimodal)
    'train_he_only_model',      # For HE-only
    'train_mif_only_model',     # For MIF-only
    'set_seed'
]
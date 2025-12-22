"""
Vitamin-P: Multi-modal pathology image segmentation

A collection of models for nuclei and cell segmentation in H&E and MIF images.
"""

__version__ = "0.1.0"

# Import models
from .models import (
    VitaminPFlex,
    VitaminPDual,
    VitaminPHEBaseline,
    VitaminPMIFBaseline,
)

# Import core components
from .backbone import DINOv2Backbone
from .trainer import VitaminPTrainer
from .losses import DiceFocalLoss, HVLoss, get_segmentation_loss, get_hv_loss
from .blocks import ConvBlock, DecoderBlock, SegmentationHead, get_decoder_dims
from .utils import (
    SimplePreprocessing,
    prepare_he_input,
    prepare_mif_input,
    compute_dice,
    compute_iou,
    count_parameters,
    freeze_backbone,
    unfreeze_backbone,
)

__all__ = [
    # Models
    'VitaminPFlex',
    'VitaminPDual',
    'VitaminPHEBaseline',
    'VitaminPMIFBaseline',
    # Backbone
    'DINOv2Backbone',
    # Losses
    'DiceFocalLoss',
    'HVLoss',
    'get_segmentation_loss',
    'get_hv_loss',
    # Blocks
    'ConvBlock',
    'DecoderBlock',
    'SegmentationHead',
    'get_decoder_dims',
    # Utils
    'SimplePreprocessing',
    'prepare_he_input',
    'prepare_mif_input',
    'compute_dice',
    'compute_iou',
    'count_parameters',
    'freeze_backbone',
    'unfreeze_backbone',
    'VitaminPTrainer',
]
"""
Vitamin-P: DINOv2 U-Net models for H&E + MIF Pathology Images
"""
from .models import VitaminPDual, VitaminPSyn, VitaminPFlex, VitaminPBaselineHE, VitaminPBaselineMIF
from .trainer import VitaminPTrainer
from .losses import DiceFocalLoss, HVLoss, MSGELossMaps
from .backbone import DINOv2Backbone
from .blocks import ConvBlock
from .utils import SimplePreprocessing, compute_dice, prepare_he_input, prepare_mif_input
from .pretrained import load_model, available_models, MODEL_REGISTRY

__version__ = "0.2.0"

__all__ = [
    # Models
    'VitaminPDual',
    'VitaminPSyn',
    'VitaminPFlex',
    'VitaminPBaselineHE',
    'VitaminPBaselineMIF',
    # Training
    'VitaminPTrainer',
    # Losses
    'DiceFocalLoss',
    'HVLoss',
    'MSGELossMaps',
    # Building blocks
    'DINOv2Backbone',
    'ConvBlock',
    # Utils
    'SimplePreprocessing',
    'compute_dice',
    'prepare_he_input',
    'prepare_mif_input',
    # Pretrained loader  ← new
    'load_model',
    'available_models',
    'MODEL_REGISTRY',
]
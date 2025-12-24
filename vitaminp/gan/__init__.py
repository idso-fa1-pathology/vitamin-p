"""
GAN module for H&E to MIF synthesis
"""

from .models import Pix2PixGenerator, PatchGANDiscriminator
from .trainer import Pix2PixTrainer
from .losses import GANLoss
from .utils import GANPreprocessing

__version__ = "0.1.0"

__all__ = [
    'Pix2PixGenerator',
    'PatchGANDiscriminator',
    'Pix2PixTrainer',
    'GANLoss',
    'GANPreprocessing',
]
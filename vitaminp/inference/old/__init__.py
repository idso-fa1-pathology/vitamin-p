# -*- coding: utf-8 -*-
# VitaminP Inference Module
# Main API for WSI inference with VitaminP models

from vitaminp.inference.predictor import WSIPredictor
from vitaminp.inference.wsi_handler import WSIHandler
from vitaminp.inference.utils import load_wsi_metadata

__all__ = [
    "WSIPredictor",
    "WSIHandler", 
    "load_wsi_metadata",
]

__version__ = "0.1.0"
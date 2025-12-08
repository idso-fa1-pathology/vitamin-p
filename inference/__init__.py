# -*- coding: utf-8 -*-
"""
WSI Inference Module for Vitamin-P
Handles whole slide image inference with patch-based processing
Uses existing postprocessing module for HV-based instance segmentation
"""

from .wsi_inference import WSIInference
from .overlap_cleaner import OverlapCleaner
from .utils import (
    get_cell_position,
    get_cell_position_margin,
    get_edge_patch,
    convert_to_global_coordinates
)

__all__ = [
    "WSIInference",
    "OverlapCleaner",
    "get_cell_position",
    "get_cell_position_margin",
    "get_edge_patch",
    "convert_to_global_coordinates"
]

__version__ = "0.1.0"
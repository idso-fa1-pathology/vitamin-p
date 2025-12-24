#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tile-based processing for large images."""

import torch
import numpy as np
from tqdm import tqdm


class TileProcessor:
    """Efficient tile-based inference for large histology images
    
    Strategy:
    - Binary masks: Use MAXIMUM blending (preserves all detections)
    - HV maps: Use weighted averaging (smooth gradients)
    - Threshold AFTER stitching to avoid dilution
    """
    
    def __init__(self, model, device, tile_size=512, overlap=64):
        """
        Args:
            model: Trained model
            device: torch device
            tile_size: Size of tiles (default 512 to match training)
            overlap: Overlap between tiles in pixels (default 64)
        """
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
    def extract_tiles(self, image):
        """Extract overlapping tiles from large image
        
        Args:
            image: numpy array (H, W, 3)
            
        Returns:
            tiles: List of tile arrays
            positions: List of (y1, x1, y2, x2) positions
            grid_shape: (n_tiles_h, n_tiles_w)
        """
        h, w = image.shape[:2]
        tiles = []
        positions = []
        
        # Calculate number of tiles needed
        n_tiles_h = int(np.ceil((h - self.overlap) / self.stride))
        n_tiles_w = int(np.ceil((w - self.overlap) / self.stride))
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile position
                y1 = i * self.stride
                x1 = j * self.stride
                y2 = min(y1 + self.tile_size, h)
                x2 = min(x1 + self.tile_size, w)
                
                # Adjust if we hit the edge
                if y2 - y1 < self.tile_size:
                    y1 = max(0, y2 - self.tile_size)
                if x2 - x1 < self.tile_size:
                    x1 = max(0, x2 - self.tile_size)
                
                tile = image[y1:y2, x1:x2]
                
                # Pad if necessary (edge cases)
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=tile.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                tiles.append(tile)
                positions.append((y1, x1, y2, x2))
        
        return tiles, positions, (n_tiles_h, n_tiles_w)
    
    def create_tile_weights(self):
        """Create smooth weight map for tile blending
        
        Returns:
            numpy array: Weight map (tile_size, tile_size)
        """
        weights = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
        
        # Create smooth falloff at edges
        fade_width = self.overlap // 2
        
        # Top edge
        for i in range(fade_width):
            weights[i, :] *= (i + 1) / fade_width
        # Bottom edge
        for i in range(fade_width):
            weights[-(i+1), :] *= (i + 1) / fade_width
        # Left edge
        for j in range(fade_width):
            weights[:, j] *= (j + 1) / fade_width
        # Right edge
        for j in range(fade_width):
            weights[:, -(j+1)] *= (j + 1) / fade_width
        
        return weights
    
    def blend_overlaps(self, full_map, tile_pred, y1, x1, y2, x2, weight_map, tile_weights):
        """Blend tile predictions using weighted averaging
        
        Args:
            full_map: Full image map to update
            tile_pred: Tile prediction
            y1, x1, y2, x2: Tile position
            weight_map: Accumulated weights
            tile_weights: Weight map for this tile
        """
        tile_h = y2 - y1
        tile_w = x2 - x1
        
        # Add weighted prediction
        full_map[y1:y2, x1:x2] += tile_pred[:tile_h, :tile_w] * tile_weights[:tile_h, :tile_w]
        weight_map[y1:y2, x1:x2] += tile_weights[:tile_h, :tile_w]
    
    def stitch_predictions(self, tiles_preds, positions, image_shape, branch_outputs):
        """Stitch tile predictions back into full image
        
        Args:
            tiles_preds: List of prediction dicts from each tile
            positions: List of tile positions
            image_shape: Original image shape (H, W, C)
            branch_outputs: Dict with keys like 'seg', 'hv' for this branch
            
        Returns:
            dict: Stitched predictions {seg, hv}
        """
        h, w = image_shape[:2]
        tile_weights = self.create_tile_weights()
        
        # Initialize output maps
        seg_max = np.zeros((h, w), dtype=np.float32)
        hv = np.zeros((h, w, 2), dtype=np.float32)
        weight_map_hv = np.zeros((h, w), dtype=np.float32)
        
        # Blend all tiles
        for pred, (y1, x1, y2, x2) in zip(tiles_preds, positions):
            tile_h = y2 - y1
            tile_w = x2 - x1
            
            # Segmentation: take MAXIMUM (preserves detections)
            seg_max[y1:y2, x1:x2] = np.maximum(
                seg_max[y1:y2, x1:x2],
                pred['seg'][:tile_h, :tile_w]
            )
            
            # HV maps: weighted average (smooth gradients)
            self.blend_overlaps(hv[:, :, 0], pred['hv'][0],
                              y1, x1, y2, x2, weight_map_hv, tile_weights)
            self.blend_overlaps(hv[:, :, 1], pred['hv'][1],
                              y1, x1, y2, x2, weight_map_hv, tile_weights)
        
        # Normalize HV maps by accumulated weights
        hv = np.divide(hv, weight_map_hv[:, :, None], where=weight_map_hv[:, :, None] > 0)
        
        return {
            'seg': seg_max,
            'hv': hv
        }
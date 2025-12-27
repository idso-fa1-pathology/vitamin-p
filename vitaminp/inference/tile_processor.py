#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tile-based processing for large images with per-tile post-processing."""

import torch
import numpy as np
from tqdm import tqdm


class TileProcessor:
    """Efficient tile-based inference for large histology images
    
    NEW Strategy:
    - Process each tile immediately after inference (like CellViT)
    - Extract instances from each tile locally
    - Convert coordinates from local to global
    - Collect all instances across tiles
    - Clean overlaps only at tile boundaries
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
        self.wsi_reader = None 
        
    def extract_tiles_streaming(self, wsi_reader, filter_tissue=False, tissue_threshold=0.1):
        """Extract tiles on-demand from WSI reader without loading full image (NEW)
        
        Args:
            wsi_reader: WSIReader object with .read_region() method
            filter_tissue: Whether to filter tiles by tissue content
            tissue_threshold: Minimum tissue percentage (0-1) for a tile to be processed
            
        Returns:
            positions: List of (y1, x1, y2, x2) positions
            grid_shape: (n_tiles_h, n_tiles_w)
            tile_mask: Boolean array indicating which tiles have tissue (None if not filtering)
        """
        h, w = wsi_reader.height, wsi_reader.width
        self.wsi_reader = wsi_reader  # Store for later tile reading
        
        positions = []
        tile_mask = []
        
        # Calculate number of tiles needed
        n_tiles_h = int(np.ceil((h - self.overlap) / self.stride))
        n_tiles_w = int(np.ceil((w - self.overlap) / self.stride))
        
        print(f"   Scanning {n_tiles_h}x{n_tiles_w} tile grid...")
        
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
                
                # Tissue filtering
                if filter_tissue:
                    # Read tile on-demand for tissue detection
                    tile = wsi_reader.read_region((x1, y1), (x2-x1, y2-y1))
                    tissue_pct = self._calculate_tissue_percentage(tile)
                    has_tissue = tissue_pct >= tissue_threshold
                    tile_mask.append(has_tissue)
                else:
                    tile_mask.append(True)
                
                positions.append((y1, x1, y2, x2))
        
        return positions, (n_tiles_h, n_tiles_w), (tile_mask if filter_tissue else None)
        
    def read_tile(self, position):
        """Read a single tile from WSI reader (NEW)
        
        Args:
            position: (y1, x1, y2, x2) tile position
            
        Returns:
            numpy array (tile_size, tile_size, 3), padded if needed
        """
        if self.wsi_reader is None:
            raise ValueError("No WSI reader available. Call extract_tiles_streaming first.")
        
        y1, x1, y2, x2 = position
        
        # Read region
        tile = self.wsi_reader.read_region((x1, y1), (x2-x1, y2-y1))
        
        # Pad if needed (edge tiles)
        if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size, tile.shape[2]), dtype=tile.dtype)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
        
        return tile
    
    def extract_tiles(self, image, filter_tissue=False, tissue_threshold=0.1):
        """Extract overlapping tiles from loaded image (LEGACY - for compatibility)
        
        This method loads the full image. For streaming, use extract_tiles_streaming().
        """
        h, w = image.shape[:2]
        tiles = []
        positions = []
        tile_mask = []
        
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
                
                # Tissue filtering
                if filter_tissue:
                    tissue_pct = self._calculate_tissue_percentage(tile)
                    has_tissue = tissue_pct >= tissue_threshold
                    tile_mask.append(has_tissue)
                    
                    if not has_tissue:
                        tiles.append(None)
                        positions.append((y1, x1, y2, x2))
                        continue
                else:
                    tile_mask.append(True)
                
                # Pad edge tiles
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    padded = np.zeros((self.tile_size, self.tile_size, tile.shape[2]), dtype=tile.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                tiles.append(tile)
                positions.append((y1, x1, y2, x2))
        
        return tiles, positions, (n_tiles_h, n_tiles_w), (tile_mask if filter_tissue else None)


    def _calculate_tissue_percentage(self, tile):
        """Calculate percentage of tissue in a tile using Otsu thresholding
        
        Args:
            tile: RGB tile image (H, W, 3) or MIF image (H, W, 2)
            
        Returns:
            float: Tissue percentage (0-1)
        """
        import cv2
        
        # **NEW: Handle different number of channels**
        if tile.ndim != 3:
            raise ValueError(f"Expected 3D tile (H, W, C), got shape {tile.shape}")
        
        n_channels = tile.shape[2]
        
        if n_channels == 3:
            # RGB/H&E image - convert to grayscale
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        elif n_channels == 2:
            # MIF image - use first channel (nuclear) or max projection
            # Option 1: Use nuclear channel only
            gray = tile[:, :, 0]
            
            # Option 2: Use max projection of both channels (uncomment if preferred)
            # gray = np.max(tile, axis=2)
            
            # Ensure uint8 for Otsu
            if gray.dtype == np.float32 or gray.dtype == np.float64:
                gray = (gray * 255).astype(np.uint8)
            elif gray.dtype == np.uint16:
                gray = (gray / 256).astype(np.uint8)
        elif n_channels == 1:
            # Single channel - use directly
            gray = tile[:, :, 0]
        else:
            raise ValueError(f"Unsupported number of channels: {n_channels}")
        
        # Ensure gray is uint8
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is white (common in histology)
        if binary.mean() > 127:
            binary = 255 - binary
        
        # Calculate tissue percentage
        tissue_pct = np.sum(binary > 0) / binary.size
        
        return tissue_pct

    def process_tile_instances(
        self, 
        tile_pred, 
        position, 
        magnification=40, 
        detection_threshold=0.5,
        use_gpu=True
    ):
        """Process a single tile to extract instances (NEW - CellViT style)
        
        This is the KEY function that processes each tile immediately!
        
        Args:
            tile_pred: Dict with 'seg' and 'hv' predictions for this tile
            position: (y1, x1, y2, x2) global position of this tile
            magnification: Magnification level (20 or 40)
            detection_threshold: Binary threshold
            use_gpu: Use GPU for post-processing
            
        Returns:
            List of cell dictionaries with GLOBAL coordinates
        """
        from vitaminp.postprocessing.hv_postprocess import process_model_outputs
        
        y1, x1, y2, x2 = position
        
        # Extract instances from THIS TILE ONLY (small 512x512)
        inst_map, inst_info, num_instances = process_model_outputs(
            seg_pred=tile_pred['seg'],
            h_map=tile_pred['hv'][0],
            v_map=tile_pred['hv'][1],
            magnification=magnification,
            binary_threshold=detection_threshold,
            use_gpu=use_gpu  # Now we can use GPU efficiently!
        )
        
        # Convert local coordinates to global coordinates
        cells_global = []
        for inst_id, cell_data in inst_info.items():
            # Adjust coordinates from tile-local to WSI-global
            cell_global = {
                'bbox': (cell_data['bbox'] + np.array([[y1, x1], [y1, x1]])).tolist(),
                'centroid': (cell_data['centroid'] + np.array([x1, y1])).tolist(),
                'contour': (cell_data['contour'] + np.array([x1, y1])).tolist(),
                'type_prob': cell_data.get('type_prob'),
                'type': cell_data.get('type'),
                'patch_coordinates': position,  # Store which tile this came from
                'edge_position': self._is_edge_cell(cell_data['bbox'], self.tile_size, margin=self.overlap//2),
            }
            cells_global.append(cell_global)
        
        return cells_global
    
    def _is_edge_cell(self, bbox, patch_size, margin=32):
        """Check if a cell is near the edge of a tile
        
        Args:
            bbox: Bounding box [[rmin, cmin], [rmax, cmax]]
            patch_size: Size of the patch
            margin: Margin to consider as edge
            
        Returns:
            bool: True if cell is near edge
        """
        bbox = np.array(bbox)
        return (np.max(bbox) > patch_size - margin or np.min(bbox) < margin)
    
    # ============================================================================
    # LEGACY METHODS (kept for backward compatibility with old stitching approach)
    # These are no longer used in the new per-tile processing approach
    # ============================================================================
    
    def create_tile_weights(self):
        """Create smooth weight map for tile blending (LEGACY)"""
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
        """Blend tile predictions using weighted averaging (LEGACY)"""
        tile_h = y2 - y1
        tile_w = x2 - x1
        
        # Add weighted prediction
        full_map[y1:y2, x1:x2] += tile_pred[:tile_h, :tile_w] * tile_weights[:tile_h, :tile_w]
        weight_map[y1:y2, x1:x2] += tile_weights[:tile_h, :tile_w]
    
    def stitch_predictions(self, tiles_preds, positions, image_shape, branch_outputs, tile_mask=None):
        """Stitch tile predictions back into full image (LEGACY - for visualization only)
        
        NOTE: This is now only used for creating visualization masks.
        Instance extraction happens per-tile in the new approach.
        """
        h, w = image_shape[:2]
        tile_weights = self.create_tile_weights()
        
        # Initialize output maps
        seg_max = np.zeros((h, w), dtype=np.float32)
        hv = np.zeros((h, w, 2), dtype=np.float32)
        weight_map_hv = np.zeros((h, w), dtype=np.float32)
        
        # Blend all tiles
        for idx, (pred, (y1, x1, y2, x2)) in enumerate(zip(tiles_preds, positions)):
            # Skip filtered tiles
            if pred is None:
                continue
            
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
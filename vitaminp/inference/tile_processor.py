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
        

    def extract_tiles_streaming(self, wsi_reader, filter_tissue=False, tissue_threshold=0.1, tissue_dilation=1, scale_factor=1.0):
        """Extract tiles on-demand from WSI reader without loading full image (NEW)
        
        Args:
            wsi_reader: WSIReader object with .read_region() method
            filter_tissue: Whether to filter tiles by tissue content
            tissue_threshold: Minimum tissue percentage (0-1) for a tile to be processed
            tissue_dilation: Number of tiles to dilate tissue regions (0=no dilation, 1=dilate by 1 tile)
            scale_factor: Scale factor for resolution matching (tiles extracted from upscaled space)
            
        Returns:
            positions: List of (y1, x1, y2, x2) positions IN ORIGINAL WSI SPACE
            grid_shape: (n_tiles_h, n_tiles_w)
            tile_mask: Boolean array indicating which tiles have tissue (None if not filtering)
        """
        import cv2
        
        # Original WSI dimensions
        h_original = wsi_reader.height
        w_original = wsi_reader.width
        
        # Virtual upscaled dimensions
        h_upscaled = int(h_original * scale_factor)
        w_upscaled = int(w_original * scale_factor)
        
        self.wsi_reader = wsi_reader
        self.scale_factor = scale_factor  # Store for tile reading
        
        positions = []
        tile_mask = []
        
        # Calculate number of tiles needed IN UPSCALED SPACE
        n_tiles_h = int(np.ceil((h_upscaled - self.overlap) / self.stride))
        n_tiles_w = int(np.ceil((w_upscaled - self.overlap) / self.stride))
        
        print(f"   Virtual upscaled size: {h_upscaled}x{w_upscaled} (from {h_original}x{w_original})")
        print(f"   Scanning {n_tiles_h}x{n_tiles_w} tile grid...")
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile position IN UPSCALED SPACE
                y1_up = i * self.stride
                x1_up = j * self.stride
                y2_up = min(y1_up + self.tile_size, h_upscaled)
                x2_up = min(x1_up + self.tile_size, w_upscaled)
                
                # Adjust if we hit the edge
                if y2_up - y1_up < self.tile_size:
                    y1_up = max(0, y2_up - self.tile_size)
                if x2_up - x1_up < self.tile_size:
                    x1_up = max(0, x2_up - self.tile_size)
                
                # Convert to ORIGINAL WSI SPACE for storage
                y1_orig = int(y1_up / scale_factor)
                x1_orig = int(x1_up / scale_factor)
                y2_orig = int(y2_up / scale_factor)
                x2_orig = int(x2_up / scale_factor)
                
                # Tissue filtering (read from original space, upscale, then check)
                if filter_tissue:
                    # Read small region from original WSI
                    tile_small = wsi_reader.read_region((x1_orig, y1_orig), (x2_orig-x1_orig, y2_orig-y1_orig))
                    # Upscale to 512x512
                    tile_upscaled = cv2.resize(tile_small, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)
                    tissue_pct = self._calculate_tissue_percentage(tile_upscaled)
                    has_tissue = tissue_pct >= tissue_threshold
                    tile_mask.append(has_tissue)
                else:
                    tile_mask.append(True)
                
                # Store ORIGINAL space coordinates
                positions.append((y1_orig, x1_orig, y2_orig, x2_orig))
        
        # Apply morphological dilation to tissue mask
        if filter_tissue and tissue_dilation > 0:
            tile_mask_2d = np.array(tile_mask, dtype=np.uint8).reshape(n_tiles_h, n_tiles_w)
            kernel_size = 2 * tissue_dilation + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            tile_mask_dilated = cv2.dilate(tile_mask_2d, kernel, iterations=1)
            tile_mask = tile_mask_dilated.flatten().tolist()
            
            n_original = sum(np.array(tile_mask_2d).flatten())
            n_dilated = sum(tile_mask)
            print(f"   Tissue dilation: {n_original} → {n_dilated} tiles (+{n_dilated - n_original} boundary tiles)")
        
        return positions, (n_tiles_h, n_tiles_w), (tile_mask if filter_tissue else None)

    def read_tile(self, position):
        """Read a single tile from WSI reader and upscale to tile_size
        
        Args:
            position: (y1, x1, y2, x2) tile position IN ORIGINAL WSI SPACE
            
        Returns:
            numpy array (tile_size, tile_size, 3) - upscaled to match training resolution
        """
        import cv2
        
        if self.wsi_reader is None:
            raise ValueError("No WSI reader available. Call extract_tiles_streaming first.")
        
        y1, x1, y2, x2 = position
        
        # Read region from ORIGINAL WSI
        tile = self.wsi_reader.read_region((x1, y1), (x2-x1, y2-y1))
        
        # Upscale to tile_size (512x512)
        if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
            tile = cv2.resize(tile, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)
        
        return tile

    def read_tile_mif(self, position):
        """Read a single MIF tile at the same position as H&E tile
        
        This method reads from the stored MIF image at the exact same position
        as the H&E tile to ensure perfect alignment, and applies channel processing.
        
        Args:
            position: (y1, x1, y2, x2) tile position IN ORIGINAL WSI SPACE
            
        Returns:
            numpy array (tile_size, tile_size, 2) - MIF tile with 2 channels (nuclear, membrane)
        """
        import cv2
        
        if not hasattr(self, 'mif_image') or self.mif_image is None:
            raise ValueError("No MIF image available. Ensure mif_image is set before calling read_tile_mif().")
        
        y1, x1, y2, x2 = position
        
        # Extract tile from full MIF image - shape is (H, W, 2) since it's already processed
        tile_mif = self.mif_image[y1:y2, x1:x2]
        
        # Upscale to tile_size (512x512) to match H&E tile
        if tile_mif.shape[0] != self.tile_size or tile_mif.shape[1] != self.tile_size:
            # Preserve number of channels during resize
            tile_mif = cv2.resize(
                tile_mif, 
                (self.tile_size, self.tile_size), 
                interpolation=cv2.INTER_LINEAR
            )
        
        return tile_mif

    def extract_tiles(self, image, filter_tissue=False, tissue_threshold=0.1, tissue_dilation=1, scale_factor=1.0):
        """Extract overlapping tiles from loaded image with virtual upscaling support
        
        This method loads the full image. For streaming, use extract_tiles_streaming().
        
        Args:
            image: Full image array (H, W, C)
            filter_tissue: Whether to filter tiles by tissue content
            tissue_threshold: Minimum tissue percentage (0-1)
            tissue_dilation: Number of tiles to dilate tissue regions
            scale_factor: Scale factor for resolution matching (virtual upscaling)
        
        Returns:
            tiles: List of tile arrays (or None for filtered tiles)
            positions: List of (y1, x1, y2, x2) positions IN ORIGINAL IMAGE SPACE
            grid_shape: (n_tiles_h, n_tiles_w)
            tile_mask: Boolean array indicating which tiles have tissue (None if not filtering)
        """
        import cv2
        
        # Original image dimensions
        h_original, w_original = image.shape[:2]
        
        # Virtual upscaled dimensions
        h_upscaled = int(h_original * scale_factor)
        w_upscaled = int(w_original * scale_factor)
        
        if scale_factor != 1.0:
            print(f"   Virtual upscaled size: {h_upscaled}x{w_upscaled} (from {h_original}x{w_original})")
        
        tiles = []
        positions = []
        tile_mask = []
        
        # Calculate number of tiles needed IN UPSCALED SPACE
        n_tiles_h = int(np.ceil((h_upscaled - self.overlap) / self.stride))
        n_tiles_w = int(np.ceil((w_upscaled - self.overlap) / self.stride))
        
        print(f"   Scanning {n_tiles_h}x{n_tiles_w} tile grid...")
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile position IN UPSCALED SPACE
                y1_up = i * self.stride
                x1_up = j * self.stride
                y2_up = min(y1_up + self.tile_size, h_upscaled)
                x2_up = min(x1_up + self.tile_size, w_upscaled)
                
                # Adjust if we hit the edge
                if y2_up - y1_up < self.tile_size:
                    y1_up = max(0, y2_up - self.tile_size)
                if x2_up - x1_up < self.tile_size:
                    x1_up = max(0, x2_up - self.tile_size)
                
                # Convert to ORIGINAL IMAGE SPACE for extraction
                y1_orig = int(y1_up / scale_factor)
                x1_orig = int(x1_up / scale_factor)
                y2_orig = int(y2_up / scale_factor)
                x2_orig = int(x2_up / scale_factor)
                
                # Extract tile from ORIGINAL image
                tile = image[y1_orig:y2_orig, x1_orig:x2_orig]
                
                # Upscale tile to tile_size (512x512) to match training resolution
                if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                    tile = cv2.resize(tile, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)
                
                # Tissue filtering (on upscaled tile)
                if filter_tissue:
                    tissue_pct = self._calculate_tissue_percentage(tile)
                    has_tissue = tissue_pct >= tissue_threshold
                    tile_mask.append(has_tissue)
                    
                    if not has_tissue:
                        tiles.append(None)
                        positions.append((y1_orig, x1_orig, y2_orig, x2_orig))
                        continue
                else:
                    tile_mask.append(True)
                
                tiles.append(tile)
                positions.append((y1_orig, x1_orig, y2_orig, x2_orig))
        
        # Apply morphological dilation to tissue mask
        if filter_tissue and tissue_dilation > 0:
            tile_mask_2d = np.array(tile_mask, dtype=np.uint8).reshape(n_tiles_h, n_tiles_w)
            kernel_size = 2 * tissue_dilation + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            tile_mask_dilated = cv2.dilate(tile_mask_2d, kernel, iterations=1)
            
            # Update tile_mask and tiles list
            tile_mask_new = tile_mask_dilated.flatten().tolist()
            
            # Add None tiles that were dilated in
            tiles_updated = []
            for idx, (old_mask, new_mask) in enumerate(zip(tile_mask, tile_mask_new)):
                if old_mask:
                    tiles_updated.append(tiles[sum(tile_mask[:idx])])
                elif new_mask:  # Newly included by dilation
                    tiles_updated.append(None)  # Will be loaded later
                
            tiles = tiles_updated
            tile_mask = tile_mask_new
            
            n_original = sum(np.array(tile_mask_2d).flatten())
            n_dilated = sum(tile_mask)
            print(f"   Tissue dilation: {n_original} → {n_dilated} tiles (+{n_dilated - n_original} boundary tiles)")
        
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
        mpp=0.25,
        detection_threshold=0.5,
        min_area_um=None,
        use_gpu=True,
        scale_factor=1.0
    ):
        """Process a single tile to extract instances (CellViT style)
        
        Args:
            tile_pred: Dict with 'seg' and 'hv' predictions for this tile
            position: (y1, x1, y2, x2) global position of this tile
            magnification: Magnification level (20 or 40)
            mpp: Microns per pixel
            detection_threshold: Binary threshold
            min_area_um: Minimum area in μm²
            use_gpu: Use GPU for post-processing
            scale_factor: Scale factor applied to tile before inference
            
        Returns:
            List of cell dictionaries with GLOBAL coordinates
        """
        from vitaminp.postprocessing.hv_postprocess import process_model_outputs
        
        y1, x1, y2, x2 = position
        
        # Extract instances from THIS TILE ONLY
        inst_map, inst_info, num_instances = process_model_outputs(
            seg_pred=tile_pred['seg'],
            h_map=tile_pred['hv'][0],
            v_map=tile_pred['hv'][1],
            magnification=magnification,
            mpp=mpp,
            binary_threshold=detection_threshold,
            min_area_um=min_area_um,
            use_gpu=use_gpu
        )
        
        # Convert local coordinates to global coordinates
        cells_global = []
        for inst_id, cell_data in inst_info.items():
            # 🔥 STEP 3: Downscale coordinates back to original space
            if scale_factor != 1.0:
                bbox_local = cell_data['bbox'] / scale_factor
                centroid_local = cell_data['centroid'] / scale_factor
                contour_local = cell_data['contour'] / scale_factor
            else:
                bbox_local = cell_data['bbox']
                centroid_local = cell_data['centroid']
                contour_local = cell_data['contour']
            
            # Then adjust from tile-local to WSI-global
            cell_global = {
                'bbox': (bbox_local + np.array([[y1, x1], [y1, x1]])).astype(np.float32).tolist(),
                'centroid': (centroid_local + np.array([x1, y1])).astype(np.float32).tolist(),
                'contour': (contour_local + np.array([x1, y1])).astype(np.int32).tolist(),  # Convert to int32 for OpenCV
                'type_prob': cell_data.get('type_prob'),
                'type': cell_data.get('type'),
                'area_pixels': cell_data.get('area_pixels'),
                'area_um': cell_data.get('area_um'),
                'patch_coordinates': position,
                'edge_position': self._is_edge_cell(bbox_local, self.tile_size, margin=self.overlap//2),
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
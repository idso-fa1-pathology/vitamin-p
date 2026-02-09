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
        # Apply morphological dilation to tissue mask
        if filter_tissue and tissue_dilation > 0:
            tile_mask_2d = np.array(tile_mask, dtype=np.uint8).reshape(n_tiles_h, n_tiles_w)
            kernel_size = 2 * tissue_dilation + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            tile_mask_dilated = cv2.dilate(tile_mask_2d, kernel, iterations=1)
            
            # Update tile_mask and tiles list
            tile_mask_new = tile_mask_dilated.flatten().tolist()
            
            # ========== FIX: Load tiles that were added by dilation ==========
            tiles_updated = []
            
            for flat_idx, (old_mask, new_mask) in enumerate(zip(tile_mask, tile_mask_new)):
                if old_mask:
                    # This tile was already in the original list
                    # FIX: Use flat_idx directly because 'tiles' list matches the grid 1:1
                    tiles_updated.append(tiles[flat_idx])
                
                elif new_mask:
                    # This tile was added by dilation - extract it from image!
                    i = flat_idx // n_tiles_w
                    j = flat_idx % n_tiles_w
                    
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
                    
                    # Upscale tile to tile_size (512x512)
                    if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                        tile = cv2.resize(tile, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)
                    
                    tiles_updated.append(tile)
                
                else:
                    # FIX: You MUST append None for background tiles to keep the list aligned!
                    tiles_updated.append(None)
            # ==================================================================
            
            tiles = tiles_updated
            tile_mask = tile_mask_new
            
            n_original = sum(np.array(tile_mask_2d).flatten())
            n_dilated = sum(tile_mask)
            print(f"   Tissue dilation: {n_original} → {n_dilated} tiles (+{n_dilated - n_original} boundary tiles)")
                
        return tiles, positions, (n_tiles_h, n_tiles_w), (tile_mask if filter_tissue else None)

    def _calculate_tissue_percentage(self, tile):
            """Robust Tissue Detection (Modality-Aware).
            
            Logic:
            1. Detect Modality based on background brightness.
            2. H&E (Bright BG): Blur + Threshold < 210 (No Auto-Contrast).
            3. MIF (Dark BG): Auto-Contrast + Threshold > 40 (With Noise Floor).
            """
            import cv2
            
            # 1. Channel Selection
            if tile.ndim == 3:
                if tile.shape[2] == 3: 
                    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
                else: 
                    gray = tile[:, :, 0]
            else:
                gray = tile
                
            # 2. Modality Detection (Bright vs Dark Background)
            # H&E background is white (~200-255). MIF background is black (~0-20).
            mean_val = np.mean(gray)
            is_brightfield = mean_val > 100 
            
            if is_brightfield:
                # === H&E / Brightfield Path ===
                # DO NOT Auto-contrast! It turns faint white noise into black artifacts.
                
                # Ensure uint8
                if gray.dtype != np.uint8: gray = gray.astype(np.uint8)
                
                # Blur to remove pixel noise
                blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                
                # Simple Threshold: Tissue is darker than 210
                # Background (211-255) becomes 0. Tissue (0-210) becomes 1.
                _, binary = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY_INV)
                
            else:
                # === MIF / Fluorescence Path ===
                # Signal is faint. We MUST Auto-contrast, but safely.
                
                gray_f = gray.astype(np.float32)
                g_min, g_max = gray_f.min(), gray_f.max()
                dynamic_range = g_max - g_min
                
                # Noise Floor: If range is tiny, it's just sensor noise. Don't stretch.
                # 16-bit images often have ~50-100 noise range. 8-bit ~10.
                noise_floor = 100 if tile.dtype == np.uint16 else 15
                
                if dynamic_range > noise_floor:
                    # Stretch valid faint signal to full 0-255
                    gray_norm = ((gray_f - g_min) / dynamic_range) * 255.0
                    gray_uint8 = gray_norm.astype(np.uint8)
                else:
                    # Range too small? Treat as empty black tile.
                    gray_uint8 = np.zeros_like(gray, dtype=np.uint8)

                # Blur
                blurred = cv2.GaussianBlur(gray_uint8, (7, 7), 0)
                
                # Threshold: Tissue is brighter than 40
                _, binary = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
                
            return np.sum(binary > 0) / binary.size
    # =========================================================================
    # MODIFIED: process_tile_instances — now supports constrained watershed
    # =========================================================================
    def process_tile_instances(
        self, 
        tile_pred, 
        position, 
        magnification=40,
        mpp=0.25,
        detection_threshold=0.5,
        min_area_um=None,
        use_gpu=True,
        scale_factor=1.0,
        grid_position=None,          # (tile_row, tile_col, n_tiles_h, n_tiles_w)
        # ── NEW: constrained watershed inputs ──────────────────────────────
        nuclei_inst_map=None,        # inst_map from the nuclei branch (same tile)
        cell_threshold=0.5,          # threshold for binarising the cell seg map
        # ────────────────────────────────────────────────────────────────────
    ):
        """Process a single tile to extract instances (CellViT style)
        
        When nuclei_inst_map is provided AND tile_pred contains a cell seg map,
        the method switches to nuclei-constrained watershed instead of the
        standard HoVer-Net instance extraction.  The output format is identical
        either way, so nothing downstream changes.

        Args:
            tile_pred: Dict with 'seg' and 'hv' predictions for this tile.
                       For the constrained path, 'seg' is the raw cell probability map.
            position: (y1, x1, y2, x2) global position of this tile
            magnification: Magnification level (20 or 40)
            mpp: Microns per pixel
            detection_threshold: Binary threshold (used only in the standard HoVer-Net path)
            min_area_um: Minimum area in μm²
            use_gpu: Use GPU for post-processing (standard path only)
            scale_factor: Scale factor applied to tile before inference
            grid_position: Tuple of (tile_row, tile_col, n_tiles_h, n_tiles_w)
            nuclei_inst_map: (H, W) instance-labeled nuclei map from the nuclei branch.
                             Pass this to activate constrained watershed for cell branches.
            cell_threshold: Probability threshold for the cell seg map when using
                            constrained watershed (default 0.5).
            
        Returns:
            List of cell dictionaries with GLOBAL coordinates
        """
        from vitaminp.postprocessing.hv_postprocess import process_model_outputs

        y1, x1, y2, x2 = position

        # ─────────────────────────────────────────────────────────────────────
        # BRANCH A: Nuclei-constrained watershed (cell branches only)
        # Activated when caller passes nuclei_inst_map from the nuclei branch.
        # ─────────────────────────────────────────────────────────────────────
        if nuclei_inst_map is not None:
            from vitaminp.inference.constrained_watershed import (
                apply_nuclei_constrained_watershed,
                extract_instances_from_labels,
            )

            # tile_pred['seg'] is the raw cell probability map (H, W) in [0, 1]
            cell_seg_map = tile_pred['seg']

            # Run constrained watershed: nuclei seeds → cell boundaries
            constrained_labels = apply_nuclei_constrained_watershed(
                nuclei_labels=nuclei_inst_map,
                cell_seg_map=cell_seg_map,
                cell_threshold=cell_threshold,
            )

            # Convert min_area_um → pixels (same logic as standard path)
            min_area_px = 0
            if min_area_um is not None and mpp > 0:
                min_area_px = int(min_area_um / (mpp ** 2))

            # Extract per-instance info in the same dict format as process_model_outputs
            inst_info = extract_instances_from_labels(
                label_map=constrained_labels,
                min_area=min_area_px,
            )

        # ─────────────────────────────────────────────────────────────────────
        # BRANCH B: Standard HoVer-Net watershed (original path, unchanged)
        # ─────────────────────────────────────────────────────────────────────
        else:
            _inst_map, inst_info, _num = process_model_outputs(
                seg_pred=tile_pred['seg'],
                h_map=tile_pred['hv'][0],
                v_map=tile_pred['hv'][1],
                magnification=magnification,
                mpp=mpp,
                binary_threshold=detection_threshold,
                min_area_um=min_area_um,
                use_gpu=use_gpu,
            )

        # ─────────────────────────────────────────────────────────────────────
        # SHARED: coordinate conversion + boundary detection
        # Identical for both paths — the inst_info dict has the same keys.
        # ─────────────────────────────────────────────────────────────────────
        cells_global = []
        for inst_id, cell_data in inst_info.items():
            # Downscale coordinates back to original WSI space if needed
            if scale_factor != 1.0:
                bbox_local     = np.array(cell_data['bbox'])     / scale_factor
                centroid_local = np.array(cell_data['centroid']) / scale_factor
                contour_local  = np.array(cell_data['contour'])  / scale_factor
            else:
                bbox_local     = np.array(cell_data['bbox'])
                centroid_local = np.array(cell_data['centroid'])
                contour_local  = np.array(cell_data['contour'])

            # ── Directional boundary detection ──────────────────────────────
            lx_min = np.min(contour_local[:, 0])
            lx_max = np.max(contour_local[:, 0])
            ly_min = np.min(contour_local[:, 1])
            ly_max = np.max(contour_local[:, 1])

            touches_top    = (ly_min <= 2)
            touches_bottom = (ly_max >= self.tile_size - 3)
            touches_left   = (lx_min <= 2)
            touches_right  = (lx_max >= self.tile_size - 3)

            grid_info = None
            if grid_position is not None:
                tile_row, tile_col, n_tiles_h, n_tiles_w = grid_position
                grid_info = {
                    'tile_row':   tile_row,
                    'tile_col':   tile_col,
                    'n_tiles_h':  n_tiles_h,
                    'n_tiles_w':  n_tiles_w,
                }
            # ─────────────────────────────────────────────────────────────────

            # Shift from tile-local → WSI-global
            cell_global = {
                'bbox':      (bbox_local + np.array([[y1, x1], [y1, x1]])).astype(np.float32).tolist(),
                'centroid':  (centroid_local + np.array([x1, y1])).astype(np.float32).tolist(),
                'contour':   (contour_local  + np.array([x1, y1])).astype(np.int32).tolist(),
                'type_prob': cell_data.get('type_prob'),
                'type':      cell_data.get('type'),
                'area_pixels': cell_data.get('area_pixels'),
                'area_um':     cell_data.get('area_um'),
                'patch_coordinates': position,
                # Directional boundary flags
                'touches_top':    touches_top,
                'touches_bottom': touches_bottom,
                'touches_left':   touches_left,
                'touches_right':  touches_right,
                'grid_info':      grid_info,
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
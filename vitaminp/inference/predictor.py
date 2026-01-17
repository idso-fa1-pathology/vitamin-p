#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WSI Predictor - High-level interface for whole slide inference with per-tile processing."""

import time
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

from vitaminp import SimplePreprocessing, prepare_he_input
from .wsi_handler import MultiFormatImageLoader
from .tile_processor import TileProcessor
from .overlap_cleaner import OverlapCleaner
from .utils import ResultExporter, setup_logger

MODEL_TRAINING_MPP = 0.263  # Geometric mean of training MPPs (0.2125, 0.3250)

class WSIPredictor:
    """High-level predictor for whole slide image inference
    
    NEW ARCHITECTURE (CellViT-style):
    1. Load WSI (streaming - no full image loading!)
    2. Tile extraction positions
    3. FOR EACH TILE:
       a. Load tile on-demand
       b. Run model inference
       c. IMMEDIATELY extract instances (512x512 - FAST!)
       d. Convert coordinates to global
    4. Collect all instances from all tiles
    5. Clean overlaps (only at tile boundaries)
    6. Export results
    
    Example:
        >>> model = VitaminPFlex(model_size='large').to('cuda')
        >>> predictor = WSIPredictor(model=model, device='cuda')
        >>> results = predictor.predict(
        ...     wsi_path='slide.svs',
        ...     branch='he_nuclei',
        ...     save_geojson=True
        ... )
    """
    
    SUPPORTED_BRANCHES = {
        'he_nuclei': {'seg_key': 'he_nuclei_seg', 'hv_key': 'he_nuclei_hv'},
        'he_cell': {'seg_key': 'he_cell_seg', 'hv_key': 'he_cell_hv'},
        'mif_nuclei': {'seg_key': 'mif_nuclei_seg', 'hv_key': 'mif_nuclei_hv'},
        'mif_cell': {'seg_key': 'mif_cell_seg', 'hv_key': 'mif_cell_hv'},
    }
    
    def __init__(
        self,
        model,
        checkpoint_path=None,
        device='cuda',
        patch_size=512,
        overlap=64,
        target_mpp=0.25,
        magnification=40,
        mixed_precision=False,
        logger=None,
        mif_channel_config=None,
        tissue_dilation=1,
    ):
        """Initialize WSI Predictor
        
        Args:
            model: Loaded model instance (VitaminPFlex or VitaminPDual)
            checkpoint_path: Path to checkpoint (optional, for reference)
            device: Device for inference ('cuda' or 'cpu')
            patch_size: Tile size (must match training)
            overlap: Overlap between tiles in pixels
            target_mpp: Target microns per pixel
            magnification: Magnification level (20 or 40)
            mixed_precision: Use FP16 for inference
            logger: Logger instance (creates one if None)
            mif_channel_config: MIF channel configuration
            tissue_dilation: Number of tiles to dilate tissue regions
        """
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.target_mpp = target_mpp
        self.magnification = magnification
        self.mixed_precision = mixed_precision
        self.tissue_dilation = tissue_dilation
        
        # 🔥 FIXED: Detect if this is a dual model using correct attribute names
        self.is_dual_model = (
            hasattr(model, 'he_backbone') and 
            hasattr(model, 'mif_backbone') and 
            hasattr(model, 'shared_backbone')
        )
        
        # Setup logger
        if logger is None:
            self.logger = setup_logger('WSIPredictor')
        else:
            self.logger = logger
        
        # Initialize components
        self.wsi_handler = MultiFormatImageLoader()
        self.tile_processor = TileProcessor(
            model=self.model,
            device=self.device,
            tile_size=self.patch_size,
            overlap=self.overlap
        )
        self.preprocessor = SimplePreprocessing()
        
        self.model.eval()
        
        self.logger.info(f"WSIPredictor initialized:")
        self.logger.info(f"  Device: {device}")
        self.logger.info(f"  Model type: {'VitaminPDual (dual-modality)' if self.is_dual_model else 'VitaminPFlex (single-modality)'}")
        self.logger.info(f"  Patch size: {patch_size}")
        self.logger.info(f"  Overlap: {overlap}")
        self.logger.info(f"  Magnification: {magnification}")
        self.mif_channel_config = mif_channel_config
        if mif_channel_config is not None:
            self.logger.info(f"  MIF channels: {mif_channel_config.get_description()}")

    def predict(
        self,
        wsi_path,
        wsi_path_mif=None,  # 🔥 NEW: Path to co-registered MIF image
        output_dir='results',
        branch='he_nuclei',
        branches=None,
        wsi_properties=None,
        filter_tissue=False,
        tissue_threshold=0.1,
        clean_overlaps=True,
        iou_threshold=0.5,
        save_masks=True,
        save_json=True,
        save_geojson=True,
        save_csv=False,
        save_heatmap=False,
        save_visualization=True,
        detection_threshold=0.5,
        min_area_um=3.0,
        mpp_override=None,
    ):
        """Run inference on WSI
        
        Args:
            wsi_path: Path to H&E WSI
            wsi_path_mif: Path to co-registered MIF WSI (required for dual models)
            output_dir: Output directory
            branch: Single branch to process (ignored if branches is provided)
            branches: List of branches to process (e.g., ['he_nuclei', 'mif_nuclei'])
            wsi_properties: WSI metadata (optional)
            filter_tissue: Filter tiles by tissue content
            tissue_threshold: Minimum tissue percentage (0-1)
            clean_overlaps: Remove overlapping instances
            iou_threshold: IoU threshold for overlap removal
            save_masks: Save binary masks
            save_json: Save JSON results
            save_geojson: Save GeoJSON results
            save_csv: Save CSV results
            save_heatmap: Save heatmap visualization
            save_visualization: Save visualization with contours
            detection_threshold: Binary threshold for instance extraction (0.5-0.8)
            min_area_um: Minimum cell area in μm² (default 3.0 for nuclei, None to disable)
            mpp_override: Override auto-detected MPP (for files with bad/missing metadata)
        
        Returns:
            dict: Results with predictions, instances, timing
        """
        start_time = time.time()
        
        # 🔥 NEW: Validate dual model usage
        if self.is_dual_model and wsi_path_mif is None:
            raise ValueError(
                "This is a dual model (VitaminPDual) which requires both H&E and MIF inputs. "
                "Please provide wsi_path_mif parameter."
            )
        
        if not self.is_dual_model and wsi_path_mif is not None:
            self.logger.warning(
                "wsi_path_mif provided but model is single-modality (VitaminPFlex). "
                "MIF image will be ignored."
            )
            wsi_path_mif = None
        
        # Determine branches to process
        if branches is not None:
            branch_list = branches if isinstance(branches, list) else [branches]
        else:
            branch_list = [branch]
        
        # Validate branches
        for b in branch_list:
            if b not in self.SUPPORTED_BRANCHES:
                raise ValueError(f"Unsupported branch: {b}. Choose from {list(self.SUPPORTED_BRANCHES.keys())}")
        
        # 🔥 NEW: Validate dual model usage
        if self.is_dual_model and wsi_path_mif is None:
            raise ValueError(
                "This is a dual model (VitaminPDual) which requires both H&E and MIF inputs. "
                "Please provide wsi_path_mif parameter."
            )

        if not self.is_dual_model and wsi_path_mif is not None:
            self.logger.warning(
                "wsi_path_mif provided but model is single-modality (VitaminPFlex). "
                "MIF image will be ignored."
            )
            wsi_path_mif = None
        # Process single or multiple branches
        if len(branch_list) == 1:
            return self._process_single_branch(
                wsi_path=wsi_path,
                wsi_path_mif=wsi_path_mif,  # 🔥 NEW: Pass MIF path
                branch=branch_list[0],
                output_dir=output_dir,
                clean_overlaps=clean_overlaps,
                iou_threshold=iou_threshold,
                save_masks=save_masks,
                save_json=save_json,
                save_geojson=save_geojson,
                save_csv=save_csv,
                save_visualization=save_visualization,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                detection_threshold=detection_threshold,
                min_area_um=min_area_um,
                mpp_override=mpp_override,
            )
        else:
            # Multiple branches
            all_results = {}
            for b in branch_list:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Processing branch: {b}")
                self.logger.info(f"{'='*60}")
                
                branch_output_dir = Path(output_dir) / b
                results = self._process_single_branch(
                    wsi_path=wsi_path,
                    wsi_path_mif=wsi_path_mif,  # 🔥 NEW: Pass MIF path
                    branch=b,
                    output_dir=str(branch_output_dir),
                    clean_overlaps=clean_overlaps,
                    iou_threshold=iou_threshold,
                    save_masks=save_masks,
                    save_json=save_json,
                    save_geojson=save_geojson,
                    save_csv=save_csv,
                    save_visualization=save_visualization,
                    filter_tissue=filter_tissue,
                    tissue_threshold=tissue_threshold,
                    detection_threshold=detection_threshold,
                    min_area_um=min_area_um,
                    mpp_override=mpp_override,
                )
                all_results[b] = results
            
            return all_results

    def _process_single_branch(
        self,
        wsi_path,
        wsi_path_mif,  # 🔥 NEW: MIF WSI path
        branch,
        output_dir,
        clean_overlaps,
        iou_threshold,
        save_masks,
        save_json,
        save_geojson,
        save_csv,
        save_visualization,
        filter_tissue=False,
        tissue_threshold=0.1,
        detection_threshold=0.5,
        min_area_um=3.0,
        mpp_override=None,
    ):
        """Process a single branch with per-tile instance extraction (NEW ARCHITECTURE)"""
        start_time = time.time()
        
        # Detect if this is a MIF branch
        is_mif_branch = 'mif' in branch.lower()
        
        # Auto-detect MPP from WSI metadata
        mpp = mpp_override if mpp_override is not None else self.target_mpp
        detected_mag = None
        
        # For H&E branches or single-modality models, detect MPP from H&E file
        if not is_mif_branch and mpp_override is None:
            try:
                # Try to detect from file metadata
                temp_reader = self.wsi_handler.get_wsi_reader(wsi_path)
                if hasattr(temp_reader, 'mpp') and temp_reader.mpp is not None:
                    mpp = temp_reader.mpp
                    self.logger.info(f"   ✓ Auto-detected MPP: {mpp:.4f} μm/px from file metadata")
                else:
                    self.logger.info(f"   ⚠ No MPP in metadata, using default: {mpp:.4f} μm/px")
                
                if hasattr(temp_reader, 'magnification') and temp_reader.magnification is not None:
                    detected_mag = temp_reader.magnification
                    self.logger.info(f"   ✓ Auto-detected magnification: {detected_mag}x from file metadata")
                
                temp_reader.close()
            except Exception as e:
                self.logger.info(f"   ⚠ Could not read metadata, using defaults (MPP={mpp:.4f})")
        else:
            if mpp_override is not None:
                self.logger.info(f"   Manual MPP override: {mpp:.4f} μm/px")
            else:
                self.logger.info(f"   Using default MPP: {mpp:.4f} μm/px")
        
        # Calculate scale factor
        scale_factor = mpp / MODEL_TRAINING_MPP
        self.logger.info(f"🔍 Resolution matching:")
        self.logger.info(f"   WSI MPP: {mpp:.4f} μm/px")
        self.logger.info(f"   Model training MPP: {MODEL_TRAINING_MPP:.4f} μm/px")
        self.logger.info(f"   Scale factor: {scale_factor:.2f}x")
        
        # Log filtering settings
        if min_area_um is not None:
            min_area_pixels = min_area_um / (mpp ** 2)
            self.logger.info(f"   Min area filter: {min_area_um:.1f} μm² = {min_area_pixels:.0f} pixels²")
        else:
            self.logger.info(f"   No area filtering")
        
        # Use detected magnification if available
        magnification_to_use = detected_mag if detected_mag is not None else self.magnification

        # ========================================================================
        # 🔥 NEW: Dual model branch - load BOTH H&E and MIF
        # ========================================================================
        if self.is_dual_model:
            self.logger.info(f"📁 Opening dual WSI pair:")
            self.logger.info(f"   H&E: {wsi_path}")
            self.logger.info(f"   MIF: {wsi_path_mif}")
            
            # Open H&E reader (streaming)
            wsi_reader_he = self.wsi_handler.get_wsi_reader(wsi_path)
            self.logger.info(f"   ✓ H&E Size: {wsi_reader_he.width}x{wsi_reader_he.height} pixels")
            
            # Load MIF image (full load for now - TODO: add MIF streaming later)
            image_mif = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
            image_mif = np.transpose(image_mif, (1, 2, 0))
            self.logger.info(f"   ✓ MIF Size: {image_mif.shape[0]}x{image_mif.shape[1]} pixels, {image_mif.shape[2]} channels")
            
            # Verify alignment
            if wsi_reader_he.height != image_mif.shape[0] or wsi_reader_he.width != image_mif.shape[1]:
                raise ValueError(
                    f"H&E and MIF images are not aligned! "
                    f"H&E: {wsi_reader_he.height}x{wsi_reader_he.width}, "
                    f"MIF: {image_mif.shape[0]}x{image_mif.shape[1]}"
                )
            
            # Extract tile positions (based on H&E)
            self.logger.info(f"📐 Extracting tile positions...")
            positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles_streaming(
                wsi_reader_he,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation,
                scale_factor=scale_factor
            )
            
            # Store MIF image in tile processor for aligned extraction
            self.tile_processor.mif_image = image_mif
            
            wsi_reader = wsi_reader_he
            image_shape = (wsi_reader_he.height, wsi_reader_he.width, 3)
            image = None
            tiles = None
            
        # ========================================================================
        # Single-modality models (existing logic)
        # ========================================================================
        elif is_mif_branch:
            # MIF branch with single-modality model
            self.logger.info(f"📁 Opening MIF WSI: {wsi_path_mif if wsi_path_mif else wsi_path}")
            image = self.wsi_handler.load_mif_image(
                wsi_path_mif if wsi_path_mif else wsi_path, 
                self.mif_channel_config
            )
            image = np.transpose(image, (1, 2, 0))
            self.logger.info(f"   ✓ MIF Size: {image.shape[0]}x{image.shape[1]} pixels, {image.shape[2]} channels")
            
            # Use legacy tile extraction for MIF
            self.logger.info(f"📐 Extracting tiles...")
            tiles, positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles(
                image,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation
            )
            wsi_reader = None
            image_shape = image.shape
            
        else:
            # H&E branch with single-modality model
            self.logger.info(f"📁 Opening H&E WSI: {wsi_path}")
            wsi_reader = self.wsi_handler.get_wsi_reader(wsi_path)
            self.logger.info(f"   ✓ Size: {wsi_reader.width}x{wsi_reader.height} pixels")
            
            # Extract tile positions only (streaming)
            self.logger.info(f"📐 Extracting tile positions...")
            positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles_streaming(
                wsi_reader,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation,
                scale_factor=scale_factor
            )
            tiles = None
            image_shape = (wsi_reader.height, wsi_reader.width, 3)
            image = None

        # Count tissue tiles
        if tile_mask is not None:
            n_tissue_tiles = sum(tile_mask)
            tissue_pct = n_tissue_tiles / len(positions) * 100
            self.logger.info(f"   ✓ Created {len(positions)} tiles ({n_h}x{n_w} grid)")
            self.logger.info(f"   ✓ Tissue tiles: {n_tissue_tiles}/{len(positions)} ({tissue_pct:.1f}%)")
        else:
            self.logger.info(f"   ✓ Created {len(positions)} tiles ({n_h}x{n_w} grid)")
        
        # ========================================================================
        # Process tiles and extract instances
        # ========================================================================
        self.logger.info(f"🧠 Running predictions and extracting instances on {branch}...")
        
        all_cells = []
        tiles_preds_for_viz = []
        
        if tiles is not None:
            # MIF path - tiles already loaded (single-modality)
            for idx, tile in enumerate(tqdm(positions, desc="Processing tiles")):
                if tile_mask is not None and not tile_mask[idx]:
                    tiles_preds_for_viz.append(None)
                    continue
                
                tile = tiles[idx]
                position = positions[idx]
                
                # Run inference (single input)
                pred = self._predict_tile(
                    tile_he=tile, 
                    tile_mif=None,
                    branch=branch, 
                    is_mif=is_mif_branch
                )
                
                if save_masks or save_visualization:
                    tiles_preds_for_viz.append(None)
                
                # Extract instances
                cells_in_tile = self.tile_processor.process_tile_instances(
                    tile_pred=pred,
                    position=position,
                    magnification=magnification_to_use,
                    mpp=mpp,
                    detection_threshold=detection_threshold,
                    min_area_um=min_area_um,
                    use_gpu=True,
                    scale_factor=scale_factor
                )
                all_cells.extend(cells_in_tile)
                
        else:
            # Streaming path - load tiles on-demand
            for idx, position in enumerate(tqdm(positions, desc="Processing tiles")):
                if tile_mask is not None and not tile_mask[idx]:
                    tiles_preds_for_viz.append(None)
                    continue
                
                # 🔥 NEW: For dual models, read BOTH H&E and MIF tiles
                if self.is_dual_model:
                    # Read H&E tile from streaming reader
                    tile_he = self.tile_processor.read_tile(position)
                    
                    # Read corresponding MIF tile at same position
                    tile_mif = self.tile_processor.read_tile_mif(position)
                    
                    # Resize both tiles if needed
                    if scale_factor != 1.0:
                        new_size = int(self.patch_size * scale_factor)
                        tile_he = cv2.resize(tile_he, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
                        tile_mif = cv2.resize(tile_mif, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
                    
                    # Run dual inference
                    pred = self._predict_tile(
                        tile_he=tile_he,
                        tile_mif=tile_mif,
                        branch=branch,
                        is_mif=is_mif_branch
                    )
                else:
                    # Single-modality H&E streaming
                    tile_he = self.tile_processor.read_tile(position)
                    
                    if scale_factor != 1.0:
                        new_size = int(self.patch_size * scale_factor)
                        tile_he = cv2.resize(tile_he, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
                    
                    pred = self._predict_tile(
                        tile_he=tile_he,
                        tile_mif=None,
                        branch=branch,
                        is_mif=is_mif_branch
                    )
                
                if save_masks or save_visualization:
                    tiles_preds_for_viz.append(None)
                
                # Extract instances
                cells_in_tile = self.tile_processor.process_tile_instances(
                    tile_pred=pred,
                    position=position,
                    magnification=magnification_to_use,
                    mpp=mpp,
                    detection_threshold=detection_threshold,
                    min_area_um=min_area_um,
                    use_gpu=True,
                    scale_factor=scale_factor
                )
                all_cells.extend(cells_in_tile)

        self.logger.info(f"   ✓ Extracted {len(all_cells)} instances from tiles (before cleaning)")
        
        # Convert cells list to inst_info dict format
        inst_info = {}
        for idx, cell in enumerate(all_cells, start=1):
            inst_info[idx] = {
                'bbox': np.array(cell['bbox']),
                'centroid': np.array(cell['centroid']),
                'contour': np.array(cell['contour']),
                'type_prob': cell.get('type_prob'),
                'type': cell.get('type'),
            }
        
        num_instances = len(inst_info)
        
        # Clean overlaps
        if clean_overlaps and len(inst_info) > 0:
            self.logger.info(f"🧹 Cleaning overlapping instances at tile boundaries...")
            edge_cells = [cell for cell in all_cells if cell.get('edge_position', False)]
            self.logger.info(f"   Found {len(edge_cells)} edge cells to check for overlaps")
            
            inst_info = self._clean_overlaps(inst_info, iou_threshold)
            num_instances = len(inst_info)
            self.logger.info(f"   ✓ After cleaning: {num_instances} instances")
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger.info(f"💾 Saving results to {output_dir}...")
        
        object_type = 'nuclei' if 'nuclei' in branch else 'cell'
        
        if save_json or save_geojson:
            ResultExporter.export_all_formats(
                inst_info_dict=inst_info,
                save_dir=output_dir,
                image_path=wsi_path,
                object_type=object_type
            )
        
        if save_visualization:
            # Load appropriate image for visualization
            if image is None:
                self.logger.info(f"   Loading full image for visualization...")
                
                # 🔥 NEW: Load MIF image for MIF branches, H&E for H&E branches
                if is_mif_branch:
                    # Load MIF image
                    if self.is_dual_model and wsi_path_mif is not None:
                        # For dual models, load from MIF path
                        image_mif_viz = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
                        image = np.transpose(image_mif_viz, (1, 2, 0))  # (2, H, W) -> (H, W, 2)
                    else:
                        # For single-modality MIF, it's already loaded or use wsi_path
                        if wsi_path_mif:
                            image_mif_viz = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
                        else:
                            image_mif_viz = self.wsi_handler.load_mif_image(wsi_path, self.mif_channel_config)
                        image = np.transpose(image_mif_viz, (1, 2, 0))  # (2, H, W) -> (H, W, 2)
                else:
                    # Load H&E image
                    wsi_reader_viz = self.wsi_handler.get_wsi_reader(wsi_path)
                    image = wsi_reader_viz.read_region((0, 0), (wsi_reader_viz.width, wsi_reader_viz.height))
                    wsi_reader_viz.close()
            
            self._save_visualization(image, inst_info, output_dir, object_type)
        
        processing_time = time.time() - start_time
        
        results = {
            'branch': branch,
            'num_detections': num_instances,
            'processing_time': processing_time,
            'output_dir': str(output_dir),
            'instances': inst_info,
        }
        
        self.logger.info(f"✅ Complete! {num_instances} detections in {processing_time:.2f}s")
        
        return results

    def _predict_tile(self, tile_he, tile_mif=None, branch='he_nuclei', is_mif=False):
        """Run inference on a single tile (supports both single and dual models)
        
        Args:
            tile_he: H&E tile image (H, W, 3) - RGB
            tile_mif: MIF tile image (H, W, 2) - 2-channel (nuclear, membrane)
            branch: Branch name
            is_mif: Whether this is MIF data (affects preprocessing for single-modality)
        
        Returns:
            dict: Predictions with 'seg' and 'hv' keys
        """
        # ========================================================================
        # 🔥 DUAL MODEL PATH
        # ========================================================================
        if self.is_dual_model:
            if tile_mif is None:
                raise ValueError("Dual model requires both tile_he and tile_mif inputs")
            
            # Prepare H&E tile (3 channels - RGB)
            if tile_he.max() > 1.0:
                tile_he = tile_he.astype(np.float32) / 255.0
            tile_he_tensor = torch.from_numpy(tile_he).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Prepare MIF tile (2 channels - already processed by load_mif_image)
            # tile_mif is already float32 in [0, 1] range and shape (H, W, 2)
            if tile_mif.max() > 1.0:
                tile_mif = tile_mif.astype(np.float32) / 255.0
            
            # Convert (H, W, 2) -> (1, 2, H, W)
            tile_mif_tensor = torch.from_numpy(tile_mif).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Apply preprocessing
            from vitaminp import prepare_he_input
            tile_he_tensor = prepare_he_input(tile_he_tensor)
            
            # 🔥 DO NOT use prepare_mif_input for dual models - it adds a 3rd channel
            # MIF backbone expects 2 channels, not 3
            # tile_mif_tensor stays as (1, 2, H, W)
            
            # Apply normalization
            tile_he_tensor = self.preprocessor.percentile_normalize(tile_he_tensor)
            tile_mif_tensor = self.preprocessor.percentile_normalize(tile_mif_tensor)
            
            # Predict with BOTH inputs
            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tile_he_tensor, tile_mif_tensor)  # 🔥 TWO INPUTS
                else:
                    outputs = self.model(tile_he_tensor, tile_mif_tensor)  # 🔥 TWO INPUTS
            
            # Extract branch outputs
            branch_config = self.SUPPORTED_BRANCHES[branch]
            seg = outputs[branch_config['seg_key']][0, 0].cpu().numpy()
            hv = outputs[branch_config['hv_key']][0].cpu().numpy()
            
            return {'seg': seg, 'hv': hv}
        
        # ========================================================================
        # SINGLE-MODALITY MODEL PATH (existing logic)
        # ========================================================================
        else:
            # For single-modality, tile_he contains either H&E or MIF data
            tile = tile_he
            
            # Prepare tile
            if tile.max() > 1.0:
                tile = tile.astype(np.float32) / 255.0
            
            # Convert to tensor: (H, W, C) -> (1, C, H, W)
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Apply appropriate preprocessing based on image type
            if is_mif:
                from vitaminp import prepare_mif_input
                tile_tensor = prepare_mif_input(tile_tensor)  # Adds 3rd channel for single-modality
            else:
                from vitaminp import prepare_he_input
                tile_tensor = prepare_he_input(tile_tensor)
            
            # Apply normalization
            tile_tensor = self.preprocessor.percentile_normalize(tile_tensor)
            
            # Predict with single input
            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tile_tensor)
                else:
                    outputs = self.model(tile_tensor)
            
            # Extract branch outputs
            branch_config = self.SUPPORTED_BRANCHES[branch]
            seg = outputs[branch_config['seg_key']][0, 0].cpu().numpy()
            hv = outputs[branch_config['hv_key']][0].cpu().numpy()
            
            return {'seg': seg, 'hv': hv}

    def _clean_overlaps(self, inst_info, iou_threshold):
        """Clean overlapping instances"""
        detections = []
        inst_id_mapping = {}
        
        for idx, (inst_id, inst_data) in enumerate(inst_info.items()):
            contour = inst_data['contour']
            
            if 'bbox' in inst_data:
                bbox = inst_data['bbox']
            else:
                x_coords = contour[:, 0]
                y_coords = contour[:, 1]
                bbox = [[x_coords.min(), y_coords.min()], [x_coords.max(), y_coords.max()]]
            
            detection = {
                'bbox': bbox,
                'centroid': inst_data['centroid'].tolist() if hasattr(inst_data['centroid'], 'tolist') else inst_data['centroid'],
                'contour': contour.tolist() if hasattr(contour, 'tolist') else contour,
                'area': inst_data.get('area', 0),
                'cell_status': 1
            }
            detections.append(detection)
            inst_id_mapping[idx] = inst_id
        
        cleaner = OverlapCleaner(
            detections=detections,
            logger=self.logger,
            iou_threshold=iou_threshold,
            max_iterations=10
        )
        cleaned_df = cleaner.clean()
        
        inst_info_cleaned = {}
        for idx in cleaned_df.index:
            inst_info_cleaned[inst_id_mapping[idx]] = inst_info[inst_id_mapping[idx]]
        
        return inst_info_cleaned
    
    def _save_masks(self, predictions, output_dir, object_type):
        """Save binary masks"""
        cv2.imwrite(
            str(output_dir / f'{object_type}_mask.png'),
            (predictions['seg'] * 255).astype(np.uint8)
        )
        
        # HV magnitude
        hv_mag = np.sqrt(predictions['hv'][:, :, 0]**2 + predictions['hv'][:, :, 1]**2)
        cv2.imwrite(
            str(output_dir / f'{object_type}_hv_magnitude.png'),
            (hv_mag * 255).astype(np.uint8)
        )
    
    def _save_visualization(self, image, inst_info, output_dir, object_type):
        """Save visualization with contours
        
        Args:
            image: Original image (H, W, 3) for RGB or (H, W, 2) for MIF
            inst_info: Instance information dictionary
            output_dir: Output directory
            object_type: 'nuclei' or 'cell'
        """
        # Handle both RGB and MIF images
        if image.shape[2] == 2:
            # MIF image - create RGB visualization
            vis_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
            vis_image[:, :, 0] = image[:, :, 0]  # Nuclear -> Red channel
            vis_image[:, :, 1] = image[:, :, 1]  # Membrane -> Green channel
            
            # Normalize to 0-255
            if vis_image.max() <= 1.0:
                vis_image = (vis_image * 255).astype(np.uint8)
            else:
                vis_image = vis_image.astype(np.uint8)
        elif image.shape[2] == 3:
            # RGB image - use directly
            vis_image = image.copy()
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Draw contours and centroids
        for inst_id, inst_data in inst_info.items():
            contour = inst_data['contour']
            
            # Ensure contour is in correct shape for cv2.drawContours
            if isinstance(contour, np.ndarray) and len(contour) >= 3:
                # Convert to shape (N, 1, 2) if needed
                if contour.ndim == 2:
                    contour = contour.reshape(-1, 1, 2).astype(np.int32)
                else:
                    contour = contour.astype(np.int32)
                
                cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
            
            # Draw centroid
            centroid = inst_data['centroid']
            if isinstance(centroid, np.ndarray):
                centroid = centroid.astype(int)
            else:
                centroid = np.array(centroid, dtype=int)
            
            cv2.circle(vis_image, tuple(centroid), 3, (255, 0, 0), -1)
        
        # Handle color conversion based on image type
        if image.shape[2] == 2:
            # MIF was already converted to RGB above, save directly
            cv2.imwrite(
                str(output_dir / f'{object_type}_boundaries.png'),
                vis_image
            )
        else:
            # RGB image - convert RGB to BGR for OpenCV
            cv2.imwrite(
                str(output_dir / f'{object_type}_boundaries.png'),
                cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            )
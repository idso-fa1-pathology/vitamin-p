#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WSI Predictor - High-level interface for whole slide inference with per-tile processing.

NOW SUPPORTS VitaminPDual with optional synthetic MIF generation!
"""

import time
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

from vitaminp import SimplePreprocessing, prepare_he_input
from vitaminp.gan import Pix2PixGenerator, GANPreprocessing
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
       b. [OPTIONAL] Generate synthetic MIF from H&E using GAN
       c. Run model inference
       d. IMMEDIATELY extract instances (512x512 - FAST!)
       e. Convert coordinates to global
    4. Collect all instances from all tiles
    5. Clean overlaps (only at tile boundaries)
    6. Export results
    
    Example:
        >>> # VitaminPFlex (MIF only)
        >>> model = VitaminPFlex(model_size='large').to('cuda')
        >>> predictor = WSIPredictor(model=model, device='cuda')
        >>> results = predictor.predict(
        ...     wsi_path='slide.ome.tiff',
        ...     branch='mif_nuclei',
        ...     save_geojson=True
        ... )
        
        >>> # VitaminPDual with real MIF
        >>> model = VitaminPDual(model_size='base').to('cuda')
        >>> predictor = WSIPredictor(model=model, device='cuda')
        >>> results = predictor.predict(
        ...     wsi_path='he_slide.svs',
        ...     wsi_path_mif='mif_slide.ome.tiff',
        ...     branches=['he_nuclei', 'he_cell'],
        ...     save_geojson=True
        ... )
        
        >>> # VitaminPDual with synthetic MIF (H&E only!)
        >>> model = VitaminPDual(model_size='base').to('cuda')
        >>> predictor = WSIPredictor(
        ...     model=model, 
        ...     device='cuda',
        ...     gan_checkpoint_path='checkpoints/pix2pix_he_to_mif_best.pth',
        ...     use_synthetic_mif=True  # ← Auto-generate synthetic MIF!
        ... )
        >>> results = predictor.predict(
        ...     wsi_path='he_slide.svs',  # ← Only H&E needed!
        ...     branches=['he_nuclei', 'he_cell'],
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
        gan_checkpoint_path=None,
        gan_use_attention=True,
        gan_use_spectral_norm=False,
        gan_n_residual_blocks=4,
        use_synthetic_mif=False,  # ← NEW parameter
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
            gan_checkpoint_path: Path to GAN checkpoint for synthetic MIF generation
            gan_use_attention: GAN architecture parameter
            gan_use_spectral_norm: GAN architecture parameter
            gan_n_residual_blocks: GAN architecture parameter
            use_synthetic_mif: If True, generate synthetic MIF from H&E (VitaminPDual only)
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
        self.use_synthetic_mif = use_synthetic_mif
        
        model_class_name = type(model).__name__
        
        # 🔥 Detect model type
        self.is_dual_model = (model_class_name == 'VitaminPDual')
        
        # Validate synthetic MIF usage
        if use_synthetic_mif and not self.is_dual_model:
            raise ValueError(
                "use_synthetic_mif=True requires VitaminPDual model. "
                f"Current model is {model_class_name}"
            )
        
        if use_synthetic_mif and gan_checkpoint_path is None:
            raise ValueError(
                "use_synthetic_mif=True requires gan_checkpoint_path to be provided."
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
        
        # ========== GAN SETUP FOR SYNTHETIC MIF ==========
        self.gan_generator = None
        self.gan_preprocessor = None
        
        if gan_checkpoint_path is not None:
            self.logger.info(f"🎨 Loading GAN generator from {gan_checkpoint_path}")
            
            # Load checkpoint first to check for saved architecture params
            checkpoint = torch.load(gan_checkpoint_path, map_location=device)
            
            # Try to extract architecture from checkpoint, fall back to provided params
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                use_attention = config.get('use_attention', gan_use_attention)
                use_spectral_norm = config.get('use_spectral_norm', gan_use_spectral_norm)
                n_residual_blocks = config.get('n_residual_blocks', gan_n_residual_blocks)
                self.logger.info(f"   ✓ Using architecture from checkpoint metadata")
            else:
                use_attention = gan_use_attention
                use_spectral_norm = gan_use_spectral_norm
                n_residual_blocks = gan_n_residual_blocks
                self.logger.info(f"   ⚠ No architecture metadata in checkpoint, using provided params")
            
            # Log architecture
            self.logger.info(f"   Architecture: attention={use_attention}, "
                            f"spectral_norm={use_spectral_norm}, "
                            f"residual_blocks={n_residual_blocks}")
            
            # Create generator with correct architecture
            self.gan_generator = Pix2PixGenerator(
                in_channels=3, 
                out_channels=2,
                use_attention=use_attention,
                use_spectral_norm=use_spectral_norm,
                n_residual_blocks=n_residual_blocks
            ).to(device)
            
            self.gan_generator.load_state_dict(checkpoint['generator_state_dict'])
            self.gan_generator.eval()
            
            self.gan_preprocessor = GANPreprocessing()
            self.logger.info(f"   ✓ GAN generator loaded successfully")

        self.model.eval()
        
        # Log model type
        if self.is_dual_model:
            if use_synthetic_mif:
                model_type = 'VitaminPDual (dual-modality with synthetic MIF generation)'
            else:
                model_type = 'VitaminPDual (dual-modality)'
        else:
            model_type = 'VitaminPFlex (single-modality)'
        
        self.logger.info(f"WSIPredictor initialized:")
        self.logger.info(f"  Device: {device}")
        self.logger.info(f"  Model type: {model_type}")
        self.logger.info(f"  Patch size: {patch_size}")
        self.logger.info(f"  Overlap: {overlap}")
        self.logger.info(f"  Magnification: {magnification}")
        if use_synthetic_mif:
            self.logger.info(f"  Synthetic MIF: Enabled")
        self.mif_channel_config = mif_channel_config
        if mif_channel_config is not None:
            self.logger.info(f"  MIF channels: {mif_channel_config.get_description()}")

    def predict(
        self,
        wsi_path,
        wsi_path_mif=None,
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
            wsi_path: Path to H&E WSI (or MIF for Flex model)
            wsi_path_mif: Path to co-registered MIF WSI (required for dual models unless use_synthetic_mif=True)
            output_dir: Output directory
            branch: Single branch to process (ignored if branches is provided)
            branches: List of branches to process (e.g., ['he_nuclei', 'he_cell'])
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
        
        # Validate dual model usage
        if self.is_dual_model and wsi_path_mif is None and not self.use_synthetic_mif:
            raise ValueError(
                "This is a dual model (VitaminPDual) which requires both H&E and MIF inputs. "
                "Please provide wsi_path_mif parameter OR set use_synthetic_mif=True when initializing WSIPredictor."
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
        
        # Process single or multiple branches
        if len(branch_list) == 1:
            return self._process_single_branch(
                wsi_path=wsi_path,
                wsi_path_mif=wsi_path_mif,
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
                    wsi_path_mif=wsi_path_mif,
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
        wsi_path_mif,
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
        """Process a single branch with per-tile instance extraction"""
        start_time = time.time()
        
        # Detect branch type
        is_mif_branch = 'mif' in branch.lower()
        
        # Use MIF predictions for H&E branches in DUAL models only
        use_mif_for_he = False
        actual_branch = branch
        
        if self.is_dual_model and not is_mif_branch and (wsi_path_mif is not None or self.use_synthetic_mif):
            use_mif_for_he = True
            actual_branch = branch.replace('he_', 'mif_')
            self.logger.info(f"🔄 Using MIF predictions for {branch} (better quality)")
        
        # Auto-detect MPP from WSI metadata
        mpp = mpp_override if mpp_override is not None else self.target_mpp
        detected_mag = None
        
        # For H&E branches or single-modality models, detect MPP from H&E file
        if not is_mif_branch and mpp_override is None:
            try:
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
        
        magnification_to_use = detected_mag if detected_mag is not None else self.magnification

        # ========================================================================
        # DUAL MODEL BRANCH
        # ========================================================================
        if self.is_dual_model:
            if self.use_synthetic_mif:
                # Synthetic MIF mode - only H&E needed
                self.logger.info(f"📁 Opening H&E WSI (will generate synthetic MIF):")
                self.logger.info(f"   H&E: {wsi_path}")
                
                wsi_reader_he = self.wsi_handler.get_wsi_reader(wsi_path)
                self.logger.info(f"   ✓ H&E Size: {wsi_reader_he.width}x{wsi_reader_he.height} pixels")
                
                # No MIF image to load - will generate per-tile
                image_mif = None
            else:
                # Real MIF mode - load both H&E and MIF
                self.logger.info(f"📁 Opening dual WSI pair:")
                self.logger.info(f"   H&E: {wsi_path}")
                self.logger.info(f"   MIF: {wsi_path_mif}")
                
                wsi_reader_he = self.wsi_handler.get_wsi_reader(wsi_path)
                self.logger.info(f"   ✓ H&E Size: {wsi_reader_he.width}x{wsi_reader_he.height} pixels")
                
                image_mif = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
                image_mif = np.transpose(image_mif, (1, 2, 0))
                self.logger.info(f"   ✓ MIF Size: {image_mif.shape[0]}x{image_mif.shape[1]} pixels, {image_mif.shape[2]} channels")
                
                if wsi_reader_he.height != image_mif.shape[0] or wsi_reader_he.width != image_mif.shape[1]:
                    raise ValueError(
                        f"H&E and MIF images are not aligned! "
                        f"H&E: {wsi_reader_he.height}x{wsi_reader_he.width}, "
                        f"MIF: {image_mif.shape[0]}x{image_mif.shape[1]}"
                    )
            
            self.logger.info(f"📐 Extracting tile positions...")
            positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles_streaming(
                wsi_reader_he,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation,
                scale_factor=scale_factor
            )
            
            if image_mif is not None:
                self.tile_processor.mif_image = image_mif
            
            wsi_reader = wsi_reader_he
            image_shape = (wsi_reader_he.height, wsi_reader_he.width, 3)
            image = None
            tiles = None
        
        # ========================================================================
        # MIF BRANCH (single-modality)
        # ========================================================================
        elif is_mif_branch:
            self.logger.info(f"📁 Opening MIF WSI: {wsi_path_mif if wsi_path_mif else wsi_path}")
            image = self.wsi_handler.load_mif_image(
                wsi_path_mif if wsi_path_mif else wsi_path, 
                self.mif_channel_config
            )
            image = np.transpose(image, (1, 2, 0))
            self.logger.info(f"   ✓ MIF Size: {image.shape[0]}x{image.shape[1]} pixels, {image.shape[2]} channels")
            
            self.logger.info(f"📐 Extracting tiles...")
            tiles, positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles(
                image,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation,
                scale_factor=scale_factor
            )
            wsi_reader = None
            image_shape = image.shape
        
        # ========================================================================
        # H&E BRANCH (single-modality)
        # ========================================================================
        else:
            self.logger.info(f"📁 Opening H&E WSI: {wsi_path}")
            wsi_reader = self.wsi_handler.get_wsi_reader(wsi_path)
            self.logger.info(f"   ✓ Size: {wsi_reader.width}x{wsi_reader.height} pixels")
            
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
            tile_idx = 0  # Counter for valid tiles
            tiles_processed = 0  # ← ADD THIS
            
            for idx in tqdm(range(len(positions)), desc="Processing tiles"):
                # Check if this position should be processed
                if tile_mask is not None and not tile_mask[idx]:
                    tiles_preds_for_viz.append(None)
                    continue
                
                # Calculate grid position
                tile_row = idx // n_w
                tile_col = idx % n_w
                grid_position = (tile_row, tile_col, n_h, n_w)
                
                # Get tile from the compacted tiles list
                tile = tiles[idx]
                
                position = positions[idx]
                
                # Skip None tiles
                if tile is None:
                    tiles_preds_for_viz.append(None)
                    continue
                
                # ========== ADD DEBUG HERE ==========
                tiles_processed += 1

                
                pred = self._predict_tile(
                    tile_he=tile, 
                    tile_mif=None,
                    branch=branch, 
                    is_mif=is_mif_branch,
                )

                if save_masks or save_visualization:
                    tiles_preds_for_viz.append(None)
                
                cells_in_tile = self.tile_processor.process_tile_instances(
                    tile_pred=pred,
                    position=position,
                    magnification=magnification_to_use,
                    mpp=mpp,
                    detection_threshold=detection_threshold,
                    min_area_um=min_area_um,
                    use_gpu=True,
                    scale_factor=scale_factor,
                    grid_position=grid_position,
                )

                all_cells.extend(cells_in_tile)
            
            # ========== ADD DEBUG AFTER LOOP ==========
            self.logger.info(f"   DEBUG: Actually processed {tiles_processed} tiles out of {len(tiles)} available")
            # =========================================
        else:
            # Streaming path - load tiles on-demand
            for idx, position in enumerate(tqdm(positions, desc="Processing tiles")):
                if tile_mask is not None and not tile_mask[idx]:
                    tiles_preds_for_viz.append(None)
                    continue
                
                # Calculate grid position
                tile_row = idx // n_w
                tile_col = idx % n_w
                grid_position = (tile_row, tile_col, n_h, n_w)
                
                # DUAL MODEL
                if self.is_dual_model:
                    tile_he = self.tile_processor.read_tile(position)
                    
                    if self.use_synthetic_mif:
                        if tile_he.max() > 1.0:
                            tile_he_normalized = tile_he.astype(np.float32) / 255.0
                        else:
                            tile_he_normalized = tile_he
                        
                        tile_mif = self._generate_synthetic_mif(tile_he_normalized)
                        tile_mif = (tile_mif * 255).astype(np.uint8)
                    else:
                        tile_mif = self.tile_processor.read_tile_mif(position)
                    
                    pred = self._predict_tile(
                        tile_he=tile_he,
                        tile_mif=tile_mif,
                        branch=actual_branch,
                        is_mif=is_mif_branch,
                    )
                
                # SINGLE-MODALITY H&E
                else:
                    tile_he = self.tile_processor.read_tile(position)
                    
                    pred = self._predict_tile(
                        tile_he=tile_he,
                        tile_mif=None,
                        branch=branch,
                        is_mif=is_mif_branch,
                    )
                
                if save_masks or save_visualization:
                    tiles_preds_for_viz.append(None)
                
                cells_in_tile = self.tile_processor.process_tile_instances(
                    tile_pred=pred,
                    position=position,
                    magnification=magnification_to_use,
                    mpp=mpp,
                    detection_threshold=detection_threshold,
                    min_area_um=min_area_um,
                    use_gpu=True,
                    scale_factor=scale_factor,
                    grid_position=grid_position,  # ← NEW
                )
                all_cells.extend(cells_in_tile)
        self.logger.info(f"   ✓ Extracted {len(all_cells)} instances from tiles (before cleaning)")
        
        # ========== DEBUG: Check overlap region detections ==========
        self.logger.info(f"   🔍 DEBUG: Tile configuration:")
        self.logger.info(f"      - Tile size: {self.patch_size}px")
        self.logger.info(f"      - Overlap: {self.overlap}px")
        self.logger.info(f"      - Grid: {n_h}x{n_w} tiles")

        # Check how many cells are near tile boundaries
        cells_near_boundaries = 0
        for cell in all_cells:
            centroid = cell['centroid']
            # Check if within overlap distance of any tile boundary
            for dim in [0, 1]:  # x and y
                if centroid[dim] % self.patch_size < self.overlap or \
                centroid[dim] % self.patch_size > (self.patch_size - self.overlap):
                    cells_near_boundaries += 1
                    break

        self.logger.info(f"   🔍 DEBUG: Cells near tile boundaries (within {self.overlap}px): {cells_near_boundaries}")
        if len(all_cells) > 0:
            self.logger.info(f"   🔍 DEBUG: Potential overlap rate: {cells_near_boundaries/len(all_cells)*100:.1f}%")
        else:
            self.logger.info(f"   🔍 DEBUG: No cells detected")
        # ==========================================================
        # Convert cells list to inst_info dict format
        # Convert cells list to inst_info dict format
        inst_info = {}
        for idx, cell in enumerate(all_cells, start=1):
            inst_info[idx] = {
                'bbox': np.array(cell['bbox']),
                'centroid': np.array(cell['centroid']),
                'contour': np.array(cell['contour']),
                'type_prob': cell.get('type_prob'),
                'type': cell.get('type'),
                # ========== NEW: Copy boundary flags and grid info ==========
                'touches_top': cell.get('touches_top', False),
                'touches_bottom': cell.get('touches_bottom', False),
                'touches_left': cell.get('touches_left', False),
                'touches_right': cell.get('touches_right', False),
                'grid_info': cell.get('grid_info'),
                # ============================================================
            }
        
        num_instances = len(inst_info)
        
        # Clean overlaps
        if clean_overlaps and len(inst_info) > 0:
            self.logger.info(f"🧹 Cleaning overlapping instances at tile boundaries...")
            
            # ========== DEBUG: Check edge_cells ==========
            edge_cells = [cell for cell in all_cells if cell.get('edge_position', False)]
            self.logger.info(f"   🔍 DEBUG: Total cells before cleaning: {len(inst_info)}")
            self.logger.info(f"   🔍 DEBUG: Edge cells found: {len(edge_cells)}")
            self.logger.info(f"   🔍 DEBUG: Will clean ALL instances (not just edge cells)")
            # ============================================
            
            before_count = len(inst_info)
            inst_info = self._clean_overlaps(inst_info, iou_threshold)
            num_instances = len(inst_info)
            
            # ========== DEBUG: Cleaning results ==========
            removed = before_count - num_instances
            self.logger.info(f"   🔍 DEBUG: Instances removed by cleaning: {removed}")
            self.logger.info(f"   🔍 DEBUG: Cleaning rate: {removed/before_count*100:.1f}%")
            # ============================================
            
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
            if image is None:
                self.logger.info(f"   Loading full image for visualization...")
                
                if is_mif_branch:
                    if self.is_dual_model and wsi_path_mif is not None:
                        image_mif_viz = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
                        image = np.transpose(image_mif_viz, (1, 2, 0))
                    else:
                        if wsi_path_mif:
                            image_mif_viz = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
                        else:
                            image_mif_viz = self.wsi_handler.load_mif_image(wsi_path, self.mif_channel_config)
                        image = np.transpose(image_mif_viz, (1, 2, 0))
                else:
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

    def _generate_synthetic_mif(self, tile_he):
        """Generate synthetic MIF from H&E tile using GAN
        
        Args:
            tile_he: H&E tile (512, 512, 3) numpy array in [0, 1] range
        
        Returns:
            Synthetic MIF tile (512, 512, 2) numpy array in [0, 1] range
        """
        # Tile is already 512×512 from read_tile()
        assert tile_he.shape[0] == 512 and tile_he.shape[1] == 512, \
            f"Expected 512×512 input, got {tile_he.shape}"
        
        # Convert to tensor (H, W, 3) -> (1, 3, H, W)
        tile_he_tensor = torch.from_numpy(tile_he).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Apply GAN preprocessing
        tile_he_norm = self.gan_preprocessor.percentile_normalize(tile_he_tensor)
        tile_he_gan_input = self.gan_preprocessor.to_gan_range(tile_he_norm)
        
        # Generate synthetic MIF (512×512 → 512×512)
        with torch.no_grad():
            fake_mif = self.gan_generator(tile_he_gan_input)
        
        # Convert back to [0, 1] range
        fake_mif_01 = self.gan_preprocessor.from_gan_range(fake_mif)
        
        # Convert to numpy: (1, 2, H, W) -> (H, W, 2)
        synthetic_mif = fake_mif_01[0].permute(1, 2, 0).cpu().numpy()
        
        return synthetic_mif

    def _predict_tile(self, tile_he, tile_mif=None, branch='he_nuclei', is_mif=False):
        """Run inference on a single tile
        
        Args:
            tile_he: H&E tile image (H, W, 3) - RGB
            tile_mif: MIF tile image (H, W, 2) - 2-channel (nuclear, membrane) [optional]
            branch: Branch name
            is_mif: Whether this is MIF data (affects preprocessing for single-modality)
        
        Returns:
            dict: Predictions with 'seg' and 'hv' keys
        """
        # ========================================================================
        # DUAL MODEL PATH (when tile_mif is provided)
        # ========================================================================
        if tile_mif is not None:
            # Prepare H&E tile
            if tile_he.max() > 1.0:
                tile_he = tile_he.astype(np.float32) / 255.0
            tile_he_tensor = torch.from_numpy(tile_he).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Prepare MIF tile
            if tile_mif.max() > 1.0:
                tile_mif = tile_mif.astype(np.float32) / 255.0
            
            tile_mif_tensor = torch.from_numpy(tile_mif).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Apply preprocessing
            from vitaminp import prepare_he_input
            tile_he_tensor = prepare_he_input(tile_he_tensor)
            
            # Apply normalization
            tile_he_tensor = self.preprocessor.percentile_normalize(tile_he_tensor)
            tile_mif_tensor = self.preprocessor.percentile_normalize(tile_mif_tensor)
            
            # Predict with BOTH inputs
            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tile_he_tensor, tile_mif_tensor)
                else:
                    outputs = self.model(tile_he_tensor, tile_mif_tensor)
            
            # Extract branch outputs
            branch_config = self.SUPPORTED_BRANCHES[branch]
            seg = outputs[branch_config['seg_key']][0, 0].cpu().numpy()
            hv = outputs[branch_config['hv_key']][0].cpu().numpy()
            
            return {'seg': seg, 'hv': hv}
        
        # ========================================================================
        # SINGLE-MODALITY MODEL PATH
        # ========================================================================
        else:
            tile = tile_he
            
            # Prepare tile
            if tile.max() > 1.0:
                tile = tile.astype(np.float32) / 255.0
            
            # Convert to tensor: (H, W, C) -> (1, C, H, W)
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Apply appropriate preprocessing based on image type
            if is_mif:
                from vitaminp import prepare_mif_input
                tile_tensor = prepare_mif_input(tile_tensor)
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
        """Clean overlapping instances using Hard Drop + Iterative strategy
        
        Phase 1 (Hard Drop): Drop cells touching boundaries if neighbor tile exists
        Phase 2 (Iterative): Clean remaining overlaps by keeping largest
        """
        from shapely.geometry import Polygon, MultiPolygon
        from shapely import strtree
        
        self.logger.info(f"   🔍 Starting with {len(inst_info)} instances")
        
        # ========== PHASE 1: HARD DROP ==========
        self.logger.info(f"   📍 Phase 1: Hard Drop (directional boundary cleaning)")
        
        surviving_cells = []
        dropped_count = 0
        
        for inst_id, inst_data in inst_info.items():
            # Check if this cell should be dropped
            grid_info = inst_data.get('grid_info')
            
            if grid_info is None:
                # No grid info - keep the cell (backward compatibility)
                surviving_cells.append((inst_id, inst_data))
                continue
            
            tile_row = grid_info['tile_row']
            tile_col = grid_info['tile_col']
            n_tiles_h = grid_info['n_tiles_h']
            n_tiles_w = grid_info['n_tiles_w']
            
            drop = False
            
            # Check Top: if touches top AND there's a tile above, drop
            if inst_data.get('touches_top', False) and tile_row > 0:
                drop = True
            
            # Check Bottom: if touches bottom AND there's a tile below, drop
            if inst_data.get('touches_bottom', False) and tile_row < n_tiles_h - 1:
                drop = True
            
            # Check Left: if touches left AND there's a tile to the left, drop
            if inst_data.get('touches_left', False) and tile_col > 0:
                drop = True
            
            # Check Right: if touches right AND there's a tile to the right, drop
            if inst_data.get('touches_right', False) and tile_col < n_tiles_w - 1:
                drop = True
            
            if drop:
                dropped_count += 1
            else:
                surviving_cells.append((inst_id, inst_data))
        
        self.logger.info(f"   ✓ Hard Drop removed {dropped_count} boundary cells")
        self.logger.info(f"   ✓ Survivors: {len(surviving_cells)}")
        
        if len(surviving_cells) == 0:
            return {}
        
        # ========== PHASE 2: ITERATIVE OVERLAP CLEANING ==========
        self.logger.info(f"   🔄 Phase 2: Iterative overlap removal")
        
        # Prepare cells with polygons
        current_cells = []
        for inst_id, inst_data in surviving_cells:
            contour = np.array(inst_data['contour'])
            
            try:
                poly = Polygon(contour)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)
                
                if not poly.is_empty:
                    current_cells.append({
                        'inst_id': inst_id,
                        'inst_data': inst_data,
                        'poly': poly,
                        'area': poly.area,
                    })
            except:
                # Keep cell even if polygon creation fails
                pass
        
        # Iterative overlap removal
        max_iterations = 10
        overlap_threshold = 0.01  # 1% overlap (same as your notebook)
        
        for iteration in range(max_iterations):
            if len(current_cells) == 0:
                break
            
            # Build spatial index
            geometries = [c['poly'] for c in current_cells]
            tree = strtree.STRtree(geometries)
            
            to_keep = []
            processed_indices = set()
            overlaps_found = 0
            
            for i, cell in enumerate(current_cells):
                if i in processed_indices:
                    continue
                
                # Find overlapping candidates
                candidates_idx = tree.query(cell['poly'])
                
                # Build cluster of overlapping cells
                cluster = []
                for cand_idx in candidates_idx:
                    if cand_idx in processed_indices:
                        continue
                    
                    candidate = current_cells[cand_idx]
                    
                    # Calculate overlap ratio (same logic as notebook)
                    inter_area = cell['poly'].intersection(candidate['poly']).area
                    
                    if inter_area == 0:
                        continue
                    
                    ratio_a = inter_area / cell['area']
                    ratio_b = inter_area / candidate['area']
                    
                    # If overlap > threshold OR same cell, add to cluster
                    if ratio_a > overlap_threshold or ratio_b > overlap_threshold or i == cand_idx:
                        cluster.append((cand_idx, candidate))
                
                if len(cluster) == 0:
                    # No overlaps - keep cell
                    to_keep.append(cell)
                    processed_indices.add(i)
                elif len(cluster) == 1:
                    # Only self in cluster - keep
                    to_keep.append(cell)
                    processed_indices.add(i)
                else:
                    # Multiple cells overlap - keep largest
                    overlaps_found += 1
                    cluster.sort(key=lambda x: x[1]['area'], reverse=True)
                    winner = cluster[0][1]
                    to_keep.append(winner)
                    
                    # Mark all as processed
                    for idx, _ in cluster:
                        processed_indices.add(idx)
            
            self.logger.info(f"      Iteration {iteration+1}: Found {overlaps_found} overlaps, kept {len(to_keep)}/{len(current_cells)}")
            
            if overlaps_found == 0:
                self.logger.info(f"      ✓ Converged! No more overlaps.")
                break
            
            current_cells = to_keep
        
        # Convert back to inst_info dict
        inst_info_cleaned = {}
        for cell in current_cells:
            inst_info_cleaned[cell['inst_id']] = cell['inst_data']
        
        self.logger.info(f"   ✅ Final count: {len(inst_info_cleaned)} instances")
        
        return inst_info_cleaned


    def _save_masks(self, predictions, output_dir, object_type):
        """Save binary masks"""
        cv2.imwrite(
            str(output_dir / f'{object_type}_mask.png'),
            (predictions['seg'] * 255).astype(np.uint8)
        )
        
        hv_mag = np.sqrt(predictions['hv'][:, :, 0]**2 + predictions['hv'][:, :, 1]**2)
        cv2.imwrite(
            str(output_dir / f'{object_type}_hv_magnitude.png'),
            (hv_mag * 255).astype(np.uint8)
        )
    
    def _save_visualization(self, image, inst_info, output_dir, object_type):
        """Save visualization with contours"""
        if image.shape[2] == 2:
            vis_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
            vis_image[:, :, 0] = image[:, :, 0]
            vis_image[:, :, 1] = image[:, :, 1]
            
            if vis_image.max() <= 1.0:
                vis_image = (vis_image * 255).astype(np.uint8)
            else:
                vis_image = vis_image.astype(np.uint8)
        elif image.shape[2] == 3:
            vis_image = image.copy()
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        for inst_id, inst_data in inst_info.items():
            contour = inst_data['contour']
            
            if isinstance(contour, np.ndarray) and len(contour) >= 3:
                if contour.ndim == 2:
                    contour = contour.reshape(-1, 1, 2).astype(np.int32)
                else:
                    contour = contour.astype(np.int32)
                
                cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
            
            centroid = inst_data['centroid']
            if isinstance(centroid, np.ndarray):
                centroid = centroid.astype(int)
            else:
                centroid = np.array(centroid, dtype=int)
            
            cv2.circle(vis_image, tuple(centroid), 3, (255, 0, 0), -1)
        
        if image.shape[2] == 2:
            cv2.imwrite(
                str(output_dir / f'{object_type}_boundaries.png'),
                vis_image
            )
        else:
            cv2.imwrite(
                str(output_dir / f'{object_type}_boundaries.png'),
                cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            )
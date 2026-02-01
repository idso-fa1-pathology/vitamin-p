#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WSI Predictor - High-level interface for whole slide inference with per-tile processing.

NOW SUPPORTS VitaminPDual with optional synthetic MIF generation!
NOW SUPPORTS nuclei-constrained watershed for cell branches!
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

# ── Map each cell branch to its matching nuclei branch ───────────────────────
_CELL_TO_NUCLEI_BRANCH = {
    'he_cell':  'he_nuclei',
    'mif_cell': 'mif_nuclei',
}


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
        use_synthetic_mif=False,
        use_constrained_watershed=True,   # ← NEW: on by default
        cell_threshold=0.5,               # ← NEW: threshold for cell seg map in constrained WS
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
            use_constrained_watershed: If True, use nuclei-constrained watershed for cell
                                       branches when the matching nuclei branch is also being
                                       processed.  Falls back to standard HoVer-Net when the
                                       nuclei branch is not available.  (default True)
            cell_threshold: Probability threshold applied to the cell seg map inside
                            the constrained watershed (default 0.5).
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
        self.use_constrained_watershed = use_constrained_watershed   # ← NEW
        self.cell_threshold = cell_threshold                          # ← NEW
        
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
        if use_constrained_watershed:
            self.logger.info(f"  Constrained watershed: Enabled (cell_threshold={cell_threshold})")
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
        simplify_epsilon=1.0,
        coord_precision=1,
        save_parquet=False,
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
        
        # ─── NEW: warn if constrained WS is on but nuclei branch is missing ───
        if self.use_constrained_watershed:
            for b in branch_list:
                if b in _CELL_TO_NUCLEI_BRANCH:
                    needed_nuclei = _CELL_TO_NUCLEI_BRANCH[b]
                    if needed_nuclei not in branch_list:
                        self.logger.warning(
                            f"  ⚠ Constrained watershed enabled, but '{needed_nuclei}' is not in "
                            f"branches list.  '{b}' will fall back to standard HoVer-Net watershed. "
                            f"Add '{needed_nuclei}' to branches to activate constrained watershed."
                        )
        # ──────────────────────────────────────────────────────────────────────
        
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
                simplify_epsilon=simplify_epsilon,
                coord_precision=coord_precision,
                save_parquet=save_parquet,
            )
        else:
            # ─── NEW: Multi-branch with per-tile nuclei caching ─────────────
            # Instead of processing branches sequentially (each doing a full
            # pass over all tiles), we process ALL branches together tile-by-tile.
            # This way we naturally have the nuclei inst_map available when we
            # process the cell branch on the same tile.
            return self._process_multi_branch(
                wsi_path=wsi_path,
                wsi_path_mif=wsi_path_mif,
                branch_list=branch_list,
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
                simplify_epsilon=simplify_epsilon,
                coord_precision=coord_precision,
                save_parquet=save_parquet,
            )
            # ─────────────────────────────────────────────────────────────────

    # =========================================================================
    # NEW: Multi-branch processing — runs all branches per tile together
    # =========================================================================
    def _process_multi_branch(
        self,
        wsi_path,
        wsi_path_mif,
        branch_list,
        output_dir,
        clean_overlaps,
        iou_threshold,
        save_masks,
        save_json,
        save_geojson,
        save_csv,
        save_visualization,
        filter_tissue,
        tissue_threshold,
        detection_threshold,
        min_area_um,
        mpp_override,
        simplify_epsilon,
        coord_precision,
        save_parquet,
    ):
        """Process multiple branches together, tile by tile.

        This is the key enabler for constrained watershed: by iterating tiles
        once and running every branch on each tile before moving to the next,
        we can pass the nuclei inst_map directly into the cell branch without
        any extra storage or a second pass.

        For branches that don't need constrained watershed (or when it's disabled),
        this is functionally identical to processing them sequentially — just
        slightly more memory-efficient since tile data is loaded once.
        """
        # ── shared setup (MPP, scale_factor, tile grid) ─────────────────────
        mpp = mpp_override if mpp_override is not None else self.target_mpp
        detected_mag = None

        # Auto-detect MPP
        try:
            temp_reader = self.wsi_handler.get_wsi_reader(wsi_path)
            if hasattr(temp_reader, 'mpp') and temp_reader.mpp is not None:
                mpp = temp_reader.mpp if mpp_override is None else mpp
                self.logger.info(f"   ✓ Auto-detected MPP: {mpp:.4f} μm/px")
            if hasattr(temp_reader, 'magnification') and temp_reader.magnification is not None:
                detected_mag = temp_reader.magnification
            temp_reader.close()
        except Exception:
            pass

        scale_factor = mpp / MODEL_TRAINING_MPP
        magnification_to_use = detected_mag if detected_mag is not None else self.magnification

        self.logger.info(f"🔍 Resolution matching: MPP={mpp:.4f}, scale={scale_factor:.2f}x")

        # ── open WSI & build tile grid ────────────────────────────────────
        wsi_reader_he = self.wsi_handler.get_wsi_reader(wsi_path)

        # MIF setup (dual model, real MIF)
        image_mif = None
        if self.is_dual_model and not self.use_synthetic_mif and wsi_path_mif is not None:
            image_mif = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
            image_mif = np.transpose(image_mif, (1, 2, 0))
            self.tile_processor.mif_image = image_mif

        positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles_streaming(
            wsi_reader_he,
            filter_tissue=filter_tissue,
            tissue_threshold=tissue_threshold,
            tissue_dilation=self.tissue_dilation,
            scale_factor=scale_factor,
        )

        # ── determine which cell branches can use constrained WS ──────────
        # A cell branch can use constrained WS only if:
        #   1. use_constrained_watershed is True
        #   2. its matching nuclei branch is also in branch_list
        constrained_cell_branches = set()
        if self.use_constrained_watershed:
            for b in branch_list:
                if b in _CELL_TO_NUCLEI_BRANCH:
                    nuclei_b = _CELL_TO_NUCLEI_BRANCH[b]
                    if nuclei_b in branch_list:
                        constrained_cell_branches.add(b)
                        self.logger.info(f"   🔗 {b} will use constrained watershed (seeds from {nuclei_b})")

        # ── per-branch accumulator ────────────────────────────────────────
        all_cells_per_branch = {b: [] for b in branch_list}

        # ── tile loop ─────────────────────────────────────────────────────
        self.logger.info(f"🧠 Running predictions ({len(branch_list)} branches, constrained-WS on "
                         f"{len(constrained_cell_branches)} cell branch(es))...")

        for idx, position in enumerate(tqdm(positions, desc="Processing tiles")):
            if tile_mask is not None and not tile_mask[idx]:
                continue

            tile_row = idx // n_w
            tile_col = idx % n_w
            grid_position = (tile_row, tile_col, n_h, n_w)

            # ── load tile data once ─────────────────────────────────────
            tile_he = self.tile_processor.read_tile(position)

            tile_mif = None
            if self.is_dual_model:
                if self.use_synthetic_mif:
                    tile_he_norm = tile_he.astype(np.float32) / 255.0 if tile_he.max() > 1.0 else tile_he
                    tile_mif = self._generate_synthetic_mif(tile_he_norm)
                    tile_mif = (tile_mif * 255).astype(np.uint8)
                elif image_mif is not None:
                    tile_mif = self.tile_processor.read_tile_mif(position)

            # ── run model ONCE — it returns ALL branch outputs together ──
            # We pick out what we need per branch below.
            # Determine the "actual" branch for the model call.
            # For dual models with real/synthetic MIF, he_ branches redirect to mif_ outputs.
            # We only need ONE forward pass regardless of how many branches we read.
            with torch.no_grad():
                outputs = self._forward_pass(tile_he, tile_mif)

            # ── cache: nuclei inst_maps for constrained WS ─────────────
            # keyed by nuclei branch name (e.g. 'he_nuclei', 'mif_nuclei')
            nuclei_inst_maps = {}

            # ── process branches in order: nuclei first, then cell ───────
            # Sort so nuclei branches come before cell branches
            sorted_branches = sorted(branch_list, key=lambda b: (0 if 'nuclei' in b else 1))

            for b in sorted_branches:
                # Determine which output keys to read
                actual_b = b
                if self.is_dual_model and 'he_' in b and tile_mif is not None:
                    actual_b = b.replace('he_', 'mif_')

                branch_config = self.SUPPORTED_BRANCHES[actual_b]
                seg  = outputs[branch_config['seg_key']][0, 0].cpu().numpy()
                hv   = outputs[branch_config['hv_key']][0].cpu().numpy()
                tile_pred = {'seg': seg, 'hv': hv}

                # ── decide: constrained WS or standard? ────────────────
                nuclei_inst_map_for_cell = None

                if b in constrained_cell_branches:
                    # Look up the nuclei inst_map we cached earlier this tile
                    nuclei_b = _CELL_TO_NUCLEI_BRANCH[b]
                    nuclei_inst_map_for_cell = nuclei_inst_maps.get(nuclei_b)
                    if nuclei_inst_map_for_cell is None:
                        # Shouldn't happen if branch ordering is correct, but be safe
                        self.logger.warning(f"   ⚠ Nuclei inst_map not found for {nuclei_b} on tile {idx}, "
                                            f"falling back to standard watershed for {b}")

                cells_in_tile = self.tile_processor.process_tile_instances(
                    tile_pred=tile_pred,
                    position=position,
                    magnification=magnification_to_use,
                    mpp=mpp,
                    detection_threshold=detection_threshold,
                    min_area_um=min_area_um,
                    use_gpu=True,
                    scale_factor=scale_factor,
                    grid_position=grid_position,
                    # ── constrained WS args (None → standard path) ────
                    nuclei_inst_map=nuclei_inst_map_for_cell,
                    cell_threshold=self.cell_threshold,
                )

                all_cells_per_branch[b].extend(cells_in_tile)

                # ── if this is a nuclei branch, cache its inst_map ─────
                if 'nuclei' in b:
                    # Re-run the standard extraction just to get inst_map
                    # (process_tile_instances doesn't return it, but it's cheap)
                    from vitaminp.postprocessing.hv_postprocess import process_model_outputs
                    inst_map, _, _ = process_model_outputs(
                        seg_pred=tile_pred['seg'],
                        h_map=tile_pred['hv'][0],
                        v_map=tile_pred['hv'][1],
                        magnification=magnification_to_use,
                        binary_threshold=detection_threshold,
                        min_area_um=min_area_um,
                        use_gpu=True,
                    )
                    nuclei_inst_maps[b] = inst_map

        # ── post-loop: clean overlaps + export, per branch ───────────────
        all_results = {}
        for b in branch_list:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Finalizing branch: {b}")
            self.logger.info(f"{'='*60}")

            branch_output_dir = Path(output_dir) / b
            results = self._finalize_branch(
                branch=b,
                all_cells=all_cells_per_branch[b],
                wsi_path=wsi_path,
                wsi_path_mif=wsi_path_mif,
                output_dir=str(branch_output_dir),
                clean_overlaps=clean_overlaps,
                iou_threshold=iou_threshold,
                save_masks=save_masks,
                save_json=save_json,
                save_geojson=save_geojson,
                save_csv=save_csv,
                save_visualization=save_visualization,
                simplify_epsilon=simplify_epsilon,
                coord_precision=coord_precision,
                save_parquet=save_parquet,
            )
            all_results[b] = results

        return all_results

    def _forward_pass(self, tile_he, tile_mif=None):
        """Run a single model forward pass and return the raw outputs dict.

        This centralises the tensor prep + inference so _process_multi_branch
        can call it once and then index into the outputs for each branch.
        """
        # Prepare H&E
        if tile_he.max() > 1.0:
            tile_he = tile_he.astype(np.float32) / 255.0
        tile_he_tensor = torch.from_numpy(tile_he).permute(2, 0, 1).unsqueeze(0).to(self.device)

        from vitaminp import prepare_he_input
        tile_he_tensor = prepare_he_input(tile_he_tensor)
        tile_he_tensor = self.preprocessor.percentile_normalize(tile_he_tensor)

        if tile_mif is not None:
            # Dual path
            if tile_mif.max() > 1.0:
                tile_mif = tile_mif.astype(np.float32) / 255.0
            tile_mif_tensor = torch.from_numpy(tile_mif).permute(2, 0, 1).unsqueeze(0).to(self.device)
            tile_mif_tensor = self.preprocessor.percentile_normalize(tile_mif_tensor)

            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tile_he_tensor, tile_mif_tensor)
                else:
                    outputs = self.model(tile_he_tensor, tile_mif_tensor)
        else:
            # Single-modality path
            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tile_he_tensor)
                else:
                    outputs = self.model(tile_he_tensor)

        return outputs

    # =========================================================================
    # NEW: Shared finalisation (overlap cleaning + export) extracted so both
    #      single-branch and multi-branch paths can reuse it.
    # =========================================================================
    def _finalize_branch(
        self,
        branch,
        all_cells,
        wsi_path,
        wsi_path_mif,
        output_dir,
        clean_overlaps,
        iou_threshold,
        save_masks,
        save_json,
        save_geojson,
        save_csv,
        save_visualization,
        simplify_epsilon,
        coord_precision,
        save_parquet,
    ):
        """Convert cells list → inst_info, clean overlaps, export, return results."""
        self.logger.info(f"   ✓ Extracted {len(all_cells)} instances (before cleaning)")

        # Convert cells list to inst_info dict
        inst_info = {}
        for idx, cell in enumerate(all_cells, start=1):
            inst_info[idx] = {
                'bbox':           np.array(cell['bbox']),
                'centroid':       np.array(cell['centroid']),
                'contour':        np.array(cell['contour']),
                'type_prob':      cell.get('type_prob'),
                'type':           cell.get('type'),
                'touches_top':    cell.get('touches_top', False),
                'touches_bottom': cell.get('touches_bottom', False),
                'touches_left':   cell.get('touches_left', False),
                'touches_right':  cell.get('touches_right', False),
                'grid_info':      cell.get('grid_info'),
            }

        num_instances = len(inst_info)

        # Clean overlaps
        if clean_overlaps and num_instances > 0:
            self.logger.info(f"🧹 Cleaning overlapping instances at tile boundaries...")
            before_count = num_instances
            inst_info = self._clean_overlaps(inst_info, iou_threshold)
            num_instances = len(inst_info)
            removed = before_count - num_instances
            self.logger.info(f"   ✓ Removed {removed}, remaining: {num_instances}")

        # Save results
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True, parents=True)

        object_type = 'nuclei' if 'nuclei' in branch else 'cell'

        if save_json or save_geojson or save_parquet:
            ResultExporter.export_all_formats(
                inst_info_dict=inst_info,
                save_dir=output_dir_path,
                image_path=wsi_path,
                object_type=object_type,
                simplify_epsilon=simplify_epsilon,
                coord_precision=coord_precision,
                save_parquet=save_parquet,
            )

        if save_visualization:
            is_mif_branch = 'mif' in branch.lower()
            if is_mif_branch:
                mif_path = wsi_path_mif if wsi_path_mif else wsi_path
                image_mif_viz = self.wsi_handler.load_mif_image(mif_path, self.mif_channel_config)
                image = np.transpose(image_mif_viz, (1, 2, 0))
            else:
                wsi_reader_viz = self.wsi_handler.get_wsi_reader(wsi_path)
                image = wsi_reader_viz.read_region((0, 0), (wsi_reader_viz.width, wsi_reader_viz.height))
                wsi_reader_viz.close()
            self._save_visualization(image, inst_info, output_dir_path, object_type)

        results = {
            'branch':           branch,
            'num_detections':   num_instances,
            'output_dir':       str(output_dir_path),
            'instances':        inst_info,
        }
        self.logger.info(f"✅ {branch}: {num_instances} detections")
        return results

    # =========================================================================
    # ORIGINAL: single-branch path (unchanged logic, now delegates finalisation)
    # =========================================================================
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
        simplify_epsilon=1.0,
        coord_precision=1,
        save_parquet=False,
    ):
        """Process a single branch with per-tile instance extraction.

        NOTE: When only one branch is requested, constrained watershed cannot
        activate (it needs the nuclei branch on the same tile).  A warning is
        already emitted in predict() if the user enables it without the nuclei
        branch.  This method is kept intact so single-branch usage is unchanged.
        """
        start_time = time.time()

        is_mif_branch = 'mif' in branch.lower()

        use_mif_for_he = False
        actual_branch = branch

        if self.is_dual_model and not is_mif_branch and (wsi_path_mif is not None or self.use_synthetic_mif):
            use_mif_for_he = True
            actual_branch = branch.replace('he_', 'mif_')
            self.logger.info(f"🔄 Using MIF predictions for {branch} (better quality)")

        mpp = mpp_override if mpp_override is not None else self.target_mpp
        detected_mag = None

        if not is_mif_branch and mpp_override is None:
            try:
                temp_reader = self.wsi_handler.get_wsi_reader(wsi_path)
                if hasattr(temp_reader, 'mpp') and temp_reader.mpp is not None:
                    mpp = temp_reader.mpp
                    self.logger.info(f"   ✓ Auto-detected MPP: {mpp:.4f} μm/px")
                else:
                    self.logger.info(f"   ⚠ No MPP in metadata, using default: {mpp:.4f} μm/px")
                if hasattr(temp_reader, 'magnification') and temp_reader.magnification is not None:
                    detected_mag = temp_reader.magnification
                temp_reader.close()
            except Exception:
                self.logger.info(f"   ⚠ Could not read metadata, using defaults (MPP={mpp:.4f})")
        else:
            if mpp_override is not None:
                self.logger.info(f"   Manual MPP override: {mpp:.4f} μm/px")

        scale_factor = mpp / MODEL_TRAINING_MPP
        self.logger.info(f"🔍 Resolution matching: MPP={mpp:.4f}, scale={scale_factor:.2f}x")

        magnification_to_use = detected_mag if detected_mag is not None else self.magnification

        # ── open WSI & tile grid ──────────────────────────────────────────
        if self.is_dual_model:
            wsi_reader_he = self.wsi_handler.get_wsi_reader(wsi_path)
            image_mif = None
            if not self.use_synthetic_mif and wsi_path_mif:
                image_mif = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
                image_mif = np.transpose(image_mif, (1, 2, 0))
                self.tile_processor.mif_image = image_mif

            positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles_streaming(
                wsi_reader_he,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation,
                scale_factor=scale_factor,
            )
            wsi_reader = wsi_reader_he
            tiles = None
            image = None

        elif is_mif_branch:
            image = self.wsi_handler.load_mif_image(
                wsi_path_mif if wsi_path_mif else wsi_path,
                self.mif_channel_config,
            )
            image = np.transpose(image, (1, 2, 0))
            tiles, positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles(
                image,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation,
                scale_factor=scale_factor,
            )
            wsi_reader = None

        else:
            wsi_reader = self.wsi_handler.get_wsi_reader(wsi_path)
            positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles_streaming(
                wsi_reader,
                filter_tissue=filter_tissue,
                tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation,
                scale_factor=scale_factor,
            )
            tiles = None
            image = None

        # ── tile loop ─────────────────────────────────────────────────────
        self.logger.info(f"🧠 Running predictions on {branch}...")
        all_cells = []

        if tiles is not None:
            # MIF single-modality path (tiles pre-loaded)
            for idx in tqdm(range(len(positions)), desc="Processing tiles"):
                if tile_mask is not None and not tile_mask[idx]:
                    continue

                tile = tiles[idx]
                if tile is None:
                    continue

                position = positions[idx]
                tile_row = idx // n_w
                tile_col = idx % n_w
                grid_position = (tile_row, tile_col, n_h, n_w)

                pred = self._predict_tile(tile_he=tile, tile_mif=None, branch=branch, is_mif=is_mif_branch)

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
        else:
            # Streaming path
            for idx, position in enumerate(tqdm(positions, desc="Processing tiles")):
                if tile_mask is not None and not tile_mask[idx]:
                    continue

                tile_row = idx // n_w
                tile_col = idx % n_w
                grid_position = (tile_row, tile_col, n_h, n_w)

                if self.is_dual_model:
                    tile_he = self.tile_processor.read_tile(position)
                    if self.use_synthetic_mif:
                        tile_he_norm = tile_he.astype(np.float32) / 255.0 if tile_he.max() > 1.0 else tile_he
                        tile_mif = self._generate_synthetic_mif(tile_he_norm)
                        tile_mif = (tile_mif * 255).astype(np.uint8)
                    else:
                        tile_mif = self.tile_processor.read_tile_mif(position)

                    pred = self._predict_tile(tile_he=tile_he, tile_mif=tile_mif, branch=actual_branch, is_mif=is_mif_branch)
                else:
                    tile_he = self.tile_processor.read_tile(position)
                    pred = self._predict_tile(tile_he=tile_he, tile_mif=None, branch=branch, is_mif=is_mif_branch)

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

        # ── finalise ──────────────────────────────────────────────────────
        processing_time = time.time() - start_time

        results = self._finalize_branch(
            branch=branch,
            all_cells=all_cells,
            wsi_path=wsi_path,
            wsi_path_mif=wsi_path_mif,
            output_dir=output_dir,
            clean_overlaps=clean_overlaps,
            iou_threshold=iou_threshold,
            save_masks=save_masks,
            save_json=save_json,
            save_geojson=save_geojson,
            save_csv=save_csv,
            save_visualization=save_visualization,
            simplify_epsilon=simplify_epsilon,
            coord_precision=coord_precision,
            save_parquet=save_parquet,
        )
        results['processing_time'] = processing_time
        return results

    # =========================================================================
    # Unchanged helper methods
    # =========================================================================

    def _generate_synthetic_mif(self, tile_he):
        """Generate synthetic MIF from H&E tile using GAN"""
        assert tile_he.shape[0] == 512 and tile_he.shape[1] == 512, \
            f"Expected 512×512 input, got {tile_he.shape}"

        tile_he_tensor = torch.from_numpy(tile_he).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tile_he_norm = self.gan_preprocessor.percentile_normalize(tile_he_tensor)
        tile_he_gan_input = self.gan_preprocessor.to_gan_range(tile_he_norm)

        with torch.no_grad():
            fake_mif = self.gan_generator(tile_he_gan_input)

        fake_mif_01 = self.gan_preprocessor.from_gan_range(fake_mif)
        synthetic_mif = fake_mif_01[0].permute(1, 2, 0).cpu().numpy()
        return synthetic_mif

    def _predict_tile(self, tile_he, tile_mif=None, branch='he_nuclei', is_mif=False):
        """Run inference on a single tile (single-branch path)"""
        if tile_mif is not None:
            if tile_he.max() > 1.0:
                tile_he = tile_he.astype(np.float32) / 255.0
            tile_he_tensor = torch.from_numpy(tile_he).permute(2, 0, 1).unsqueeze(0).to(self.device)

            if tile_mif.max() > 1.0:
                tile_mif = tile_mif.astype(np.float32) / 255.0
            tile_mif_tensor = torch.from_numpy(tile_mif).permute(2, 0, 1).unsqueeze(0).to(self.device)

            from vitaminp import prepare_he_input
            tile_he_tensor = prepare_he_input(tile_he_tensor)
            tile_he_tensor = self.preprocessor.percentile_normalize(tile_he_tensor)
            tile_mif_tensor = self.preprocessor.percentile_normalize(tile_mif_tensor)

            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tile_he_tensor, tile_mif_tensor)
                else:
                    outputs = self.model(tile_he_tensor, tile_mif_tensor)

            branch_config = self.SUPPORTED_BRANCHES[branch]
            seg = outputs[branch_config['seg_key']][0, 0].cpu().numpy()
            hv  = outputs[branch_config['hv_key']][0].cpu().numpy()
            return {'seg': seg, 'hv': hv}
        else:
            tile = tile_he
            if tile.max() > 1.0:
                tile = tile.astype(np.float32) / 255.0
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)

            if is_mif:
                from vitaminp import prepare_mif_input
                tile_tensor = prepare_mif_input(tile_tensor)
            else:
                from vitaminp import prepare_he_input
                tile_tensor = prepare_he_input(tile_tensor)

            tile_tensor = self.preprocessor.percentile_normalize(tile_tensor)

            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tile_tensor)
                else:
                    outputs = self.model(tile_tensor)

            branch_config = self.SUPPORTED_BRANCHES[branch]
            seg = outputs[branch_config['seg_key']][0, 0].cpu().numpy()
            hv  = outputs[branch_config['hv_key']][0].cpu().numpy()
            return {'seg': seg, 'hv': hv}

    def _clean_overlaps(self, inst_info, iou_threshold):
        """Clean overlapping instances using Hard Drop + Iterative strategy"""
        from shapely.geometry import Polygon, MultiPolygon
        from shapely import strtree

        self.logger.info(f"   🔍 Starting with {len(inst_info)} instances")

        # ── PHASE 1: HARD DROP ────────────────────────────────────────────
        self.logger.info(f"   📍 Phase 1: Hard Drop (directional boundary cleaning)")
        surviving_cells = []
        dropped_count = 0

        for inst_id, inst_data in inst_info.items():
            grid_info = inst_data.get('grid_info')
            if grid_info is None:
                surviving_cells.append((inst_id, inst_data))
                continue

            tile_row    = grid_info['tile_row']
            tile_col    = grid_info['tile_col']
            n_tiles_h   = grid_info['n_tiles_h']
            n_tiles_w   = grid_info['n_tiles_w']

            drop = False
            if inst_data.get('touches_top', False)    and tile_row > 0:              drop = True
            if inst_data.get('touches_bottom', False) and tile_row < n_tiles_h - 1:  drop = True
            if inst_data.get('touches_left', False)   and tile_col > 0:              drop = True
            if inst_data.get('touches_right', False)  and tile_col < n_tiles_w - 1:  drop = True

            if drop:
                dropped_count += 1
            else:
                surviving_cells.append((inst_id, inst_data))

        self.logger.info(f"   ✓ Hard Drop removed {dropped_count}, survivors: {len(surviving_cells)}")

        if len(surviving_cells) == 0:
            return {}

        # ── PHASE 2: ITERATIVE OVERLAP CLEANING ───────────────────────────
        self.logger.info(f"   🔄 Phase 2: Iterative overlap removal")

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
                        'inst_id':   inst_id,
                        'inst_data': inst_data,
                        'poly':      poly,
                        'area':      poly.area,
                    })
            except Exception:
                pass

        max_iterations   = 10
        overlap_threshold = 0.01

        for iteration in range(max_iterations):
            if len(current_cells) == 0:
                break

            geometries = [c['poly'] for c in current_cells]
            tree = strtree.STRtree(geometries)

            to_keep = []
            processed_indices = set()
            overlaps_found = 0

            for i, cell in enumerate(current_cells):
                if i in processed_indices:
                    continue

                candidates_idx = tree.query(cell['poly'])
                cluster = []

                for cand_idx in candidates_idx:
                    if cand_idx in processed_indices:
                        continue
                    candidate = current_cells[cand_idx]
                    inter_area = cell['poly'].intersection(candidate['poly']).area
                    if inter_area == 0:
                        continue
                    ratio_a = inter_area / cell['area']
                    ratio_b = inter_area / candidate['area']
                    if ratio_a > overlap_threshold or ratio_b > overlap_threshold or i == cand_idx:
                        cluster.append((cand_idx, candidate))

                if len(cluster) <= 1:
                    to_keep.append(cell)
                    processed_indices.add(i)
                else:
                    overlaps_found += 1
                    cluster.sort(key=lambda x: x[1]['area'], reverse=True)
                    to_keep.append(cluster[0][1])
                    for idx, _ in cluster:
                        processed_indices.add(idx)

            self.logger.info(f"      Iteration {iteration+1}: {overlaps_found} overlaps, kept {len(to_keep)}/{len(current_cells)}")

            if overlaps_found == 0:
                self.logger.info(f"      ✓ Converged!")
                break
            current_cells = to_keep

        inst_info_cleaned = {c['inst_id']: c['inst_data'] for c in current_cells}
        self.logger.info(f"   ✅ Final count: {len(inst_info_cleaned)} instances")
        return inst_info_cleaned

    def _save_visualization(self, image, inst_info, output_dir, object_type):
        """Save visualization with contours"""
        if image.shape[2] == 2:
            vis_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
            vis_image[:, :, 0] = image[:, :, 0]
            vis_image[:, :, 1] = image[:, :, 1]
            vis_image = (vis_image * 255).astype(np.uint8) if vis_image.max() <= 1.0 else vis_image.astype(np.uint8)
        elif image.shape[2] == 3:
            vis_image = image.copy()
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        for inst_id, inst_data in inst_info.items():
            contour = inst_data['contour']
            if isinstance(contour, np.ndarray) and len(contour) >= 3:
                contour = contour.reshape(-1, 1, 2).astype(np.int32) if contour.ndim == 2 else contour.astype(np.int32)
                cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)

            centroid = inst_data['centroid']
            centroid = centroid.astype(int) if isinstance(centroid, np.ndarray) else np.array(centroid, dtype=int)
            cv2.circle(vis_image, tuple(centroid), 3, (255, 0, 0), -1)

        if image.shape[2] == 2:
            cv2.imwrite(str(output_dir / f'{object_type}_boundaries.png'), vis_image)
        else:
            cv2.imwrite(str(output_dir / f'{object_type}_boundaries.png'), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
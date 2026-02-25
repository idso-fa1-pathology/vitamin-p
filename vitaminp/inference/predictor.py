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

MODEL_TRAINING_MPP = 0.2125  # Geometric mean of training MPPs (0.2125, 0.3250)

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
            mif_channel_config=None,  # <--- This argument comes in...
            tissue_dilation=1,
            gan_checkpoint_path=None,
            gan_use_attention=True,
            gan_use_spectral_norm=False,
            gan_n_residual_blocks=4,
            use_synthetic_mif=False,
            use_constrained_watershed=True,
            cell_threshold=0.5,
            batch_size=8,
        ):
            """Initialize WSI Predictor"""
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
            self.use_constrained_watershed = use_constrained_watershed
            self.cell_threshold = cell_threshold
            self.batch_size = batch_size
            
            # 🔥 RESTORED MISSING LINE 🔥
            self.mif_channel_config = mif_channel_config 
            
            model_class_name = type(model).__name__
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
                
                checkpoint = torch.load(gan_checkpoint_path, map_location=device)
                
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    use_attention = config.get('use_attention', gan_use_attention)
                    use_spectral_norm = config.get('use_spectral_norm', gan_use_spectral_norm)
                    n_residual_blocks = config.get('n_residual_blocks', gan_n_residual_blocks)
                else:
                    use_attention = gan_use_attention
                    use_spectral_norm = gan_use_spectral_norm
                    n_residual_blocks = gan_n_residual_blocks
                
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
                model_type = 'VitaminPDual (dual-modality)'
                if use_synthetic_mif:
                    model_type += ' + Synthetic MIF'
            else:
                model_type = 'VitaminPFlex (single-modality)'
            
            self.logger.info(f"WSIPredictor initialized:")
            self.logger.info(f"  Device: {device}")
            self.logger.info(f"  Model type: {model_type}")
            self.logger.info(f"  Batch Size: {batch_size}")
            if self.mif_channel_config:
                self.logger.info(f"  MIF Config: Present")

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
            """Run inference on WSI (Optimized Wrapper)"""
            
            if self.is_dual_model and wsi_path_mif is None and not self.use_synthetic_mif:
                raise ValueError("Dual model requires MIF path or synthetic MIF.")
            
            if not self.is_dual_model and wsi_path_mif is not None:
                self.logger.warning("Ignoring MIF path for single-modality model.")
                wsi_path_mif = None
            
            if branches is not None:
                branch_list = branches if isinstance(branches, list) else [branches]
            else:
                branch_list = [branch]
            
            for b in branch_list:
                if b not in self.SUPPORTED_BRANCHES:
                    raise ValueError(f"Unsupported branch: {b}")
            
            # Check constrained watershed requirements
            if self.use_constrained_watershed:
                for b in branch_list:
                    if b in _CELL_TO_NUCLEI_BRANCH:
                        needed_nuclei = _CELL_TO_NUCLEI_BRANCH[b]
                        if needed_nuclei not in branch_list:
                            self.logger.warning(f"  ⚠ '{b}' needs '{needed_nuclei}' for constrained watershed. Using fallback.")

            # ALWAYS use the optimized multi-branch batch processor
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

    # =========================================================================
    # NEW: Multi-branch processing — runs all branches per tile together
    # =========================================================================
    def _process_multi_branch(self, wsi_path, wsi_path_mif, branch_list, output_dir, clean_overlaps, iou_threshold, save_masks, save_json, save_geojson, save_csv, save_visualization, filter_tissue, tissue_threshold, detection_threshold, min_area_um, mpp_override, simplify_epsilon, coord_precision, save_parquet):
            """OPTIMIZED: Process tiles in batches for max GPU throughput."""
            
            # ── shared setup ─────────────────────
            mpp = mpp_override if mpp_override is not None else self.target_mpp
            detected_mag = None

            try:
                temp_reader = self.wsi_handler.get_wsi_reader(wsi_path)
                if hasattr(temp_reader, 'mpp') and temp_reader.mpp is not None:
                    mpp = temp_reader.mpp if mpp_override is None else mpp
                if hasattr(temp_reader, 'magnification') and temp_reader.magnification is not None:
                    detected_mag = temp_reader.magnification
                temp_reader.close()
            except Exception:
                pass

            scale_factor = mpp / MODEL_TRAINING_MPP
            magnification_to_use = detected_mag if detected_mag is not None else self.magnification
            self.logger.info(f"🔍 Resolution: MPP={mpp:.4f}, scale={scale_factor:.2f}x, Batch Size={self.batch_size}")

            # ── open WSI & build tile grid ────────────────────────────────────
            wsi_reader_he = self.wsi_handler.get_wsi_reader(wsi_path)
            image_mif = None
            if self.is_dual_model and not self.use_synthetic_mif and wsi_path_mif is not None:
                image_mif = self.wsi_handler.load_mif_image(wsi_path_mif, self.mif_channel_config)
                image_mif = np.transpose(image_mif, (1, 2, 0))
                self.tile_processor.mif_image = image_mif

            positions, (n_h, n_w), tile_mask = self.tile_processor.extract_tiles_streaming(
                wsi_reader_he, filter_tissue=filter_tissue, tissue_threshold=tissue_threshold,
                tissue_dilation=self.tissue_dilation, scale_factor=scale_factor,
            )

            constrained_cell_branches = set()
            if self.use_constrained_watershed:
                for b in branch_list:
                    if b in _CELL_TO_NUCLEI_BRANCH and _CELL_TO_NUCLEI_BRANCH[b] in branch_list:
                        constrained_cell_branches.add(b)

            all_cells_per_branch = {b: [] for b in branch_list}
            
            # ── BATCH CONTAINERS ──────────────────────────────────────────────
            batch_he_tiles = []
            batch_mif_tiles = [] 
            batch_metadata = [] 

            self.logger.info(f"🧠 Running Batch Inference...")

            for idx, position in enumerate(tqdm(positions, desc="Processing")):
                if tile_mask is not None and not tile_mask[idx]:
                    continue

                tile_row, tile_col = idx // n_w, idx % n_w
                grid_position = (tile_row, tile_col, n_h, n_w)

                # 1. LOAD TILE (CPU IO)
                tile_he = self.tile_processor.read_tile(position)
                
                # 2. FAST PRE-NORM (CPU)
                # Convert to float32 [0,1] immediately to save copies later
                if tile_he.dtype == np.uint8:
                    tile_he = tile_he.astype(np.float32) / 255.0
                elif tile_he.dtype != np.float32:
                    tile_he = tile_he.astype(np.float32)
                
                tile_mif = None
                if self.is_dual_model:
                    if self.use_synthetic_mif:
                        tile_mif = self._generate_synthetic_mif(tile_he) 
                        tile_mif = tile_mif.astype(np.float32) 
                    elif image_mif is not None:
                        tile_mif = self.tile_processor.read_tile_mif(position)
                        if tile_mif.max() > 1.0:
                            tile_mif = tile_mif.astype(np.float32) / 255.0

                # 3. ADD TO BATCH
                batch_he_tiles.append(tile_he)
                if self.is_dual_model:
                    batch_mif_tiles.append(tile_mif)
                batch_metadata.append({'pos': position, 'grid': grid_position})

                # 4. EXECUTE BATCH IF FULL OR LAST
                if len(batch_he_tiles) >= self.batch_size or idx == len(positions) - 1:
                    if len(batch_he_tiles) == 0: continue

                    self._process_batch_buffer(
                        batch_he_tiles, batch_mif_tiles, batch_metadata,
                        branch_list, constrained_cell_branches,
                        all_cells_per_branch, magnification_to_use, mpp,
                        detection_threshold, min_area_um, scale_factor
                    )
                    
                    # Clear buffers
                    batch_he_tiles = []
                    batch_mif_tiles = []
                    batch_metadata = []

            # ── clean overlaps + export ───────────────────────────────────────
            all_results = {}
            for b in branch_list:
                results = self._finalize_branch(
                    branch=b, all_cells=all_cells_per_branch[b],
                    wsi_path=wsi_path, wsi_path_mif=wsi_path_mif, output_dir=output_dir,
                    clean_overlaps=clean_overlaps, iou_threshold=iou_threshold,
                    save_masks=save_masks, save_json=save_json, save_geojson=save_geojson,
                    save_csv=save_csv, save_visualization=save_visualization,
                    simplify_epsilon=simplify_epsilon, coord_precision=coord_precision,
                    save_parquet=save_parquet,
                )
                all_results[b] = results

            return all_results

    def _process_batch_buffer(self, batch_he, batch_mif, batch_meta, branch_list, constrained_branches, accumulator, mag, mpp, thresh, min_area, scale):
            """Internal helper — executes one batch on GPU and unpacks results.
            
            FIX: Dual now mirrors Flex exactly:
              1. prepare_he_input() is applied to each H&E tile  (was missing for Dual)
              2. percentile_normalize() is applied PER-TILE, not per-batch
                 (batch-level normalisation caused cross-tile contamination → seam lines)
            """
            from vitaminp import prepare_he_input

            # ── 1. Per-tile prepare + normalize (identical to Flex _predict_tile) ──
            #
            # Flex path (_predict_tile, is_mif=False):
            #   tile_tensor = prepare_he_input(tile_tensor)           # stain aug / colour norm
            #   tile_tensor = preprocessor.percentile_normalize(tile_tensor)  # per-tile stats
            #
            # Previously Dual skipped prepare_he_input and called percentile_normalize
            # on the whole batch tensor, mixing statistics across tiles → seam lines.
            #
            # Solution: normalise each tile independently on GPU before stacking.

            prepared_he_list = []
            for tile_np in batch_he:
                t = torch.from_numpy(tile_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                t = prepare_he_input(t)                          # ← FIX 1: was missing
                t = self.preprocessor.percentile_normalize(t)   # ← FIX 2: per-tile, not per-batch
                prepared_he_list.append(t)
            tensor_he = torch.cat(prepared_he_list, dim=0)      # (B, C, H, W)

            tensor_mif = None
            if self.is_dual_model and batch_mif:
                prepared_mif_list = []
                for tile_np in batch_mif:
                    t = torch.from_numpy(tile_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                    t = self.preprocessor.percentile_normalize(t)  # per-tile MIF norm (matches Flex)
                    prepared_mif_list.append(t)
                tensor_mif = torch.cat(prepared_mif_list, dim=0)

            # ── 2. Inference (single batched forward pass) ────────────────────
            with torch.no_grad():
                if tensor_mif is not None:
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(tensor_he, tensor_mif)
                    else:
                        outputs = self.model(tensor_he, tensor_mif)
                else:
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(tensor_he)
                    else:
                        outputs = self.model(tensor_he)

            # ── 3. Unpack Results to CPU ──────────────────────────────────────
            cpu_outputs = {}
            sorted_branches = sorted(branch_list, key=lambda b: (0 if 'nuclei' in b else 1))

            for b in sorted_branches:
                cfg = self.SUPPORTED_BRANCHES[b]
                cpu_outputs[b] = {
                    'seg': outputs[cfg['seg_key']][:, 0].cpu().numpy(),
                    'hv': outputs[cfg['hv_key']].cpu().numpy()
                }

            # ── 4. Post-Process Each Tile in the Batch ────────────────────────
            for i in range(len(batch_meta)):
                meta = batch_meta[i]
                nuclei_inst_maps = {} 

                for b in sorted_branches:
                    tile_seg = cpu_outputs[b]['seg'][i]
                    tile_hv = cpu_outputs[b]['hv'][i]
                    tile_pred = {'seg': tile_seg, 'hv': tile_hv}

                    # Resolve nuclei dependency
                    nuclei_map = None
                    if b in constrained_branches:
                        nuclei_b = _CELL_TO_NUCLEI_BRANCH[b]
                        nuclei_map = nuclei_inst_maps.get(nuclei_b)

                    cells = self.tile_processor.process_tile_instances(
                        tile_pred=tile_pred, position=meta['pos'],
                        magnification=mag, mpp=mpp, detection_threshold=thresh, 
                        min_area_um=min_area, use_gpu=False, scale_factor=scale,
                        grid_position=meta['grid'], nuclei_inst_map=nuclei_map,
                        cell_threshold=self.cell_threshold,
                    )
                    accumulator[b].extend(cells)

                    if 'nuclei' in b and self.use_constrained_watershed:
                        from vitaminp.postprocessing.hv_postprocess import process_model_outputs
                        inst_map, _, _ = process_model_outputs(
                            seg_pred=tile_seg, h_map=tile_hv[0], v_map=tile_hv[1],
                            magnification=mag, binary_threshold=thresh,
                            min_area_um=min_area, use_gpu=False
                        )
                        nuclei_inst_maps[b] = inst_map
                        
    def _forward_pass_batch(self, tensor_he, tensor_mif=None):
        """Match direct inference script normalization exactly"""
        # Use preprocessor.percentile_normalize for BOTH — same as your working script
        tensor_he = self.preprocessor.percentile_normalize(tensor_he)
        
        if tensor_mif is not None:
            tensor_mif = self.preprocessor.percentile_normalize(tensor_mif)
        
        if tensor_mif is not None:
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(tensor_he, tensor_mif)
            else:
                outputs = self.model(tensor_he, tensor_mif)
        else:
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(tensor_he)
            else:
                outputs = self.model(tensor_he)
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
        """Clean overlapping instances at tile boundaries (NO Hard Drop).

        With the tile-core (owner-tile) filtering in TileProcessor, we should NOT
        directionally hard-drop boundary-touching instances here (it causes losses).
        This function now only deduplicates true overlaps.
        """
        from shapely.geometry import Polygon, MultiPolygon
        from shapely import strtree

        self.logger.info(f"   🔍 Starting with {len(inst_info)} instances")

        # ── Build polygons ───────────────────────────────────────────────────
        current_cells = []
        for inst_id, inst_data in inst_info.items():
            contour = np.array(inst_data['contour'])
            try:
                poly = Polygon(contour)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)
                if not poly.is_empty and poly.area > 0:
                    current_cells.append({
                        'inst_id':   inst_id,
                        'inst_data': inst_data,
                        'poly':      poly,
                        'area':      float(poly.area),
                    })
            except Exception:
                pass

        if len(current_cells) == 0:
            return {}

        # ── Iterative overlap removal ─────────────────────────────────────────
        # Use IOU-based dedupe (more stable than raw intersection ratios).
        max_iterations = 5  # usually enough once you do core filtering
        iou_thr = float(iou_threshold) if iou_threshold is not None else 0.5

        for iteration in range(max_iterations):
            if len(current_cells) <= 1:
                break

            geometries = [c['poly'] for c in current_cells]
            tree = strtree.STRtree(geometries)

            to_keep = []
            processed = set()
            overlaps_found = 0

            for i, cell in enumerate(current_cells):
                if i in processed:
                    continue

                # candidates includes itself; filter clusters by IoU
                cand_idx_list = tree.query(cell['poly'])
                cluster = [(i, cell)]  # always include itself

                for j in cand_idx_list:
                    if j == i or j in processed:
                        continue
                    other = current_cells[j]

                    inter = cell['poly'].intersection(other['poly']).area
                    if inter <= 0:
                        continue

                    union = cell['area'] + other['area'] - inter
                    if union <= 0:
                        continue

                    iou = inter / union
                    if iou >= iou_thr:
                        cluster.append((j, other))

                if len(cluster) == 1:
                    to_keep.append(cell)
                    processed.add(i)
                else:
                    overlaps_found += 1
                    # keep the largest polygon (most complete) in this overlap cluster
                    cluster.sort(key=lambda x: x[1]['area'], reverse=True)
                    to_keep.append(cluster[0][1])
                    for idx, _ in cluster:
                        processed.add(idx)

            self.logger.info(
                f"      Iteration {iteration+1}: {overlaps_found} overlap clusters, "
                f"kept {len(to_keep)}/{len(current_cells)}"
            )

            current_cells = to_keep
            if overlaps_found == 0:
                self.logger.info("      ✓ Converged!")
                break

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
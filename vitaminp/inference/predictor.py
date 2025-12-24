#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WSI Predictor - High-level interface for whole slide inference."""

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
from .postprocessing import process_model_outputs
from .overlap_cleaner import OverlapCleaner
from .utils import ResultExporter, setup_logger


class WSIPredictor:
    """High-level predictor for whole slide image inference
    
    Handles the complete pipeline:
    1. Load WSI (any format)
    2. Tile extraction
    3. Model inference
    4. Tile stitching
    5. Instance extraction
    6. Overlap cleaning
    7. Export results
    
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
    ):
        """Initialize WSI Predictor
        
        Args:
            model: Loaded model instance
            checkpoint_path: Path to checkpoint (optional, for reference)
            device: Device for inference ('cuda' or 'cpu')
            patch_size: Tile size (must match training)
            overlap: Overlap between tiles in pixels
            target_mpp: Target microns per pixel
            magnification: Magnification level (20 or 40)
            mixed_precision: Use FP16 for inference
            logger: Logger instance (creates one if None)
        """
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.target_mpp = target_mpp
        self.magnification = magnification
        self.mixed_precision = mixed_precision
        
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
        self.logger.info(f"  Patch size: {patch_size}")
        self.logger.info(f"  Overlap: {overlap}")
        self.logger.info(f"  Magnification: {magnification}")
    
    def predict(
        self,
        wsi_path,
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
    ):
        """Run inference on WSI
        
        Args:
            wsi_path: Path to WSI file
            output_dir: Output directory
            branch: Single branch to process ('he_nuclei', 'he_cell', etc.)
            branches: List of branches (overrides single branch)
            wsi_properties: Dict with slide_mpp, magnification
            filter_tissue: Apply tissue filtering
            tissue_threshold: Tissue threshold
            clean_overlaps: Clean overlapping detections
            iou_threshold: IoU threshold for overlap cleaning
            save_masks: Save binary masks
            save_json: Save JSON results
            save_geojson: Save GeoJSON results
            save_csv: Save CSV results
            save_heatmap: Save heatmap
            save_visualization: Save visualizations
            
        Returns:
            dict: Results with predictions, instances, timing
        """
        start_time = time.time()
        
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
                branch=branch_list[0],
                output_dir=output_dir,
                clean_overlaps=clean_overlaps,
                iou_threshold=iou_threshold,
                save_masks=save_masks,
                save_json=save_json,
                save_geojson=save_geojson,
                save_csv=save_csv,
                save_visualization=save_visualization,
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
                    branch=b,
                    output_dir=str(branch_output_dir),
                    clean_overlaps=clean_overlaps,
                    iou_threshold=iou_threshold,
                    save_masks=save_masks,
                    save_json=save_json,
                    save_geojson=save_geojson,
                    save_csv=save_csv,
                    save_visualization=save_visualization,
                )
                all_results[b] = results
            
            return all_results
    
    def _process_single_branch(
        self,
        wsi_path,
        branch,
        output_dir,
        clean_overlaps,
        iou_threshold,
        save_masks,
        save_json,
        save_geojson,
        save_csv,
        save_visualization,
    ):
        """Process a single branch"""
        start_time = time.time()
        
        # 1. Load image
        self.logger.info(f"📁 Loading image: {wsi_path}")
        image = self.wsi_handler.load_image(wsi_path)
        self.logger.info(f"   ✓ Size: {image.shape[1]}x{image.shape[0]} pixels")
        
        # 2. Extract tiles
        self.logger.info(f"📐 Extracting tiles...")
        tiles, positions, (n_h, n_w) = self.tile_processor.extract_tiles(image)
        self.logger.info(f"   ✓ Created {len(tiles)} tiles ({n_h}x{n_w} grid)")
        
        # 3. Run inference
        self.logger.info(f"🧠 Running predictions on {branch}...")
        tiles_preds = []
        
        for tile in tqdm(tiles, desc="Processing tiles"):
            pred = self._predict_tile(tile, branch)
            tiles_preds.append(pred)
        
        # 4. Stitch predictions
        self.logger.info(f"🧩 Stitching predictions...")
        stitched = self.tile_processor.stitch_predictions(
            tiles_preds=tiles_preds,
            positions=positions,
            image_shape=image.shape,
            branch_outputs=None
        )
        
        # Apply threshold
        stitched['seg'] = (stitched['seg'] > 0.5).astype(np.float32)
        coverage = (stitched['seg'] > 0).sum() / stitched['seg'].size * 100
        self.logger.info(f"   ✓ Coverage: {coverage:.2f}%")
        
        # 5. Extract instances
        self.logger.info(f"🔍 Extracting instances...")
        inst_map, inst_info, num_instances = process_model_outputs(
            seg_pred=stitched['seg'],
            h_map=stitched['hv'][:, :, 0],
            v_map=stitched['hv'][:, :, 1],
            magnification=self.magnification
        )
        self.logger.info(f"   ✓ Detected {num_instances} instances (before cleaning)")
        
        # 6. Clean overlaps
        if clean_overlaps and len(inst_info) > 0:
            self.logger.info(f"🧹 Cleaning overlapping instances...")
            inst_info = self._clean_overlaps(inst_info, iou_threshold)
            num_instances = len(inst_info)
            self.logger.info(f"   ✓ After cleaning: {num_instances} instances")
        
        # 7. Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger.info(f"💾 Saving results to {output_dir}...")
        
        # Determine object type from branch
        object_type = 'nuclei' if 'nuclei' in branch else 'cell'
        
        if save_json or save_geojson:
            ResultExporter.export_all_formats(
                inst_info_dict=inst_info,
                save_dir=output_dir,
                image_path=wsi_path,
                object_type=object_type
            )
        
        if save_masks:
            self._save_masks(stitched, output_dir, object_type)
        
        if save_visualization:
            self._save_visualization(image, inst_info, output_dir, object_type)
        
        processing_time = time.time() - start_time
        
        results = {
            'branch': branch,
            'num_detections': num_instances,
            'processing_time': processing_time,
            'output_dir': str(output_dir),
            'predictions': stitched,
            'instances': inst_info,
        }
        
        self.logger.info(f"✅ Complete! {num_instances} detections in {processing_time:.2f}s")
        
        return results
    
    def _predict_tile(self, tile, branch):
        """Run inference on a single tile"""
        # Prepare tile
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
        
        tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tile_tensor = prepare_he_input(tile_tensor)
        tile_tensor = self.preprocessor.percentile_normalize(tile_tensor)
        
        # Predict
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
        """Save visualization with contours"""
        vis_image = image.copy()
        
        for inst_id, inst_data in inst_info.items():
            contour = inst_data['contour']
            if len(contour) >= 3:
                cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
            
            centroid = inst_data['centroid'].astype(int)
            cv2.circle(vis_image, tuple(centroid), 3, (255, 0, 0), -1)
        
        cv2.imwrite(
            str(output_dir / f'{object_type}_boundaries.png'),
            cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        )
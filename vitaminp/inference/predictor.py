# -*- coding: utf-8 -*-
# WSI Predictor for VitaminP Inference
# Main API for running inference on Whole Slide Images

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import torch
import numpy as np

from vitaminp.inference.wsi_handler import WSIHandler
from vitaminp.inference.tile_processor import TileProcessor, estimate_gpu_batch_size
from vitaminp.inference.postprocessing import (
    VitaminPPostProcessor,
    aggregate_detections,
    save_detections_json,
    save_detections_csv,
    create_summary_statistics,
    create_heatmap_from_seg,
    save_heatmap,
)
from vitaminp.inference.overlap_cleaner import OverlapCleaner, mark_edge_detections
from vitaminp.inference.utils import (
    setup_logger,
    validate_checkpoint,
    format_time,
)


class WSIPredictor:
    """Main predictor class for WSI inference with VitaminP models.
    
    This class provides a simple API for running inference on whole slide images.
    
    Example:
        >>> from vitaminp import VitaminPFlex
        >>> from vitaminp.inference import WSIPredictor
        >>> 
        >>> # Initialize model
        >>> model = VitaminPFlex(model_size='large')
        >>> 
        >>> # Create predictor
        >>> predictor = WSIPredictor(
        ...     model=model,
        ...     checkpoint_path='checkpoints/best_model.pth',
        ...     device='cuda',
        ... )
        >>> 
        >>> # Run inference
        >>> results = predictor.predict(
        ...     wsi_path='slide.svs',
        ...     output_dir='results/',
        ...     branch='he_nuclei',
        ... )
    
    Args:
        model (torch.nn.Module): VitaminP model instance
        checkpoint_path (Union[str, Path]): Path to model checkpoint
        device (str): Device for inference ('cuda' or 'cpu'). Default: 'cuda'
        batch_size (Optional[int]): Batch size. If None, auto-estimated. Default: None
        patch_size (int): Size of tiles/patches. Default: 1024
        overlap (int): Overlap between patches. Default: 64
        target_mpp (float): Target microns per pixel. Default: 0.25
        magnification (int): Magnification level for HV postprocessing (20 or 40). Default: 40
        mixed_precision (bool): Use mixed precision (FP16). Default: False
        num_workers (int): DataLoader workers. Default: 4
        logger (Optional[logging.Logger]): Logger instance. Default: None
    
    Attributes:
        model (torch.nn.Module): Model for inference
        checkpoint_path (Path): Path to checkpoint
        device (str): Device
        batch_size (int): Batch size
        patch_size (int): Patch size
        overlap (int): Overlap
        target_mpp (float): Target MPP
        magnification (int): Magnification level
        mixed_precision (bool): Mixed precision flag
        num_workers (int): Number of workers
        logger (logging.Logger): Logger
        inference_transform (Callable): Transform for inference
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: Union[str, Path],
        device: str = 'cuda',
        batch_size: Optional[int] = None,
        patch_size: int = 1024,
        overlap: int = 64,
        target_mpp: float = 0.25,
        magnification: int = 40,
        mixed_precision: bool = False,
        num_workers: int = 4,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.target_mpp = target_mpp
        self.magnification = magnification
        self.mixed_precision = mixed_precision
        self.num_workers = num_workers
        
        # Setup logger
        self.logger = logger or setup_logger(name="vitaminp.inference")
        
        # Validate checkpoint
        validate_checkpoint(self.checkpoint_path)
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Estimate batch size if not provided
        if batch_size is None:
            self.batch_size = estimate_gpu_batch_size(
                model=self.model,
                input_shape=(1, 3, patch_size, patch_size),
                device=device,
            )
        else:
            self.batch_size = batch_size
        
        self.logger.info(
            f"WSIPredictor initialized: device={device}, batch_size={self.batch_size}, "
            f"patch_size={patch_size}, overlap={overlap}, target_mpp={target_mpp}, "
            f"magnification={magnification}"
        )
        
        # Setup inference transform (will be set during prediction)
        self.inference_transform: Optional[Callable] = None
    
    def _load_checkpoint(self) -> None:
        """Load model checkpoint."""
        self.logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            self.logger.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            self.logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
        
        self.logger.info("Checkpoint loaded successfully")
    
    def set_inference_transform(self, transform: Callable) -> None:
        """Set custom inference transform.
        
        Args:
            transform (Callable): Transform function/composition
        """
        self.inference_transform = transform
        self.logger.info("Custom inference transform set")
    
    def predict(
        self,
        wsi_path: Union[str, Path],
        output_dir: Union[str, Path],
        branch: str = 'he_nuclei',
        wsi_properties: Optional[Dict] = None,
        filter_tissue: bool = True,
        tissue_threshold: float = 0.1,
        clean_overlaps: bool = True,
        iou_threshold: float = 0.5,
        save_heatmap: bool = False,
        save_json: bool = True,
        save_csv: bool = False,
    ) -> Dict:
        """Run inference on a WSI.
        
        Args:
            wsi_path (Union[str, Path]): Path to WSI file
            output_dir (Union[str, Path]): Output directory for results
            branch (str): Which branch to use. Options: 'he_nuclei', 'he_cell', 'mif_nuclei', 'mif_cell'. Default: 'he_nuclei'
            wsi_properties (Optional[Dict]): WSI properties (slide_mpp, magnification)
            filter_tissue (bool): Filter tiles by tissue content. Default: True
            tissue_threshold (float): Tissue threshold for filtering. Default: 0.1
            clean_overlaps (bool): Remove overlapping detections. Default: True
            iou_threshold (float): IoU threshold for overlap cleaning. Default: 0.5
            save_heatmap (bool): Save prediction heatmap. Default: False
            save_json (bool): Save detections as JSON. Default: True
            save_csv (bool): Save detections as CSV. Default: False
            
        Returns:
            Dict: Results dictionary containing:
                - detections: List of detections
                - num_detections: Total number of detections
                - summary: Summary statistics
                - processing_time: Total processing time
                - wsi_metadata: WSI metadata
                - branch: Which branch was used
        """
        start_time = time.time()
        
        wsi_path = Path(wsi_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting inference on WSI: {wsi_path.name}")
        self.logger.info(f"Using branch: {branch}")
        
        # Step 1: Load WSI
        self.logger.info("Step 1/5: Loading WSI and generating tiles...")
        wsi_handler = WSIHandler(
            wsi_path=wsi_path,
            patch_size=self.patch_size,
            overlap=self.overlap,
            target_mpp=self.target_mpp,
            wsi_properties=wsi_properties,
            tissue_threshold=tissue_threshold,
            logger=self.logger,
        )
        
        # Step 2: Process tiles through model
        self.logger.info("Step 2/5: Processing tiles through model...")
        tile_processor = TileProcessor(
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            mixed_precision=self.mixed_precision,
            logger=self.logger,
        )
        
        predictions, tile_metadata = tile_processor.process_wsi(
            wsi_handler=wsi_handler,
            transform=self.inference_transform,
            filter_tissue=filter_tissue,
            tissue_threshold=tissue_threshold,
        )
        
        # Step 3: Post-process predictions
        self.logger.info("Step 3/5: Post-processing predictions...")
        postprocessor = VitaminPPostProcessor(
            magnification=self.magnification,
            logger=self.logger,
        )
        
        # Process predictions to extract detections
        all_detections = []
        
        # Flatten predictions and metadata if batched
        # Flatten predictions and metadata - UNBATCH the predictions
        flat_predictions = []
        flat_metadata = []

        for pred_batch, meta_list in zip(predictions, tile_metadata):
            # pred_batch is a dict with batched tensors: {key: [B, C, H, W]}
            # meta_list is a list of metadata dicts
            
            batch_size = len(meta_list) if isinstance(meta_list, list) else 1
            
            # Split the batch into individual predictions
            for i in range(batch_size):
                single_pred = {}
                for key, value in pred_batch.items():
                    single_pred[key] = value[i]  # Extract single prediction from batch
                
                flat_predictions.append(single_pred)
                
                if isinstance(meta_list, list):
                    flat_metadata.append(meta_list[i])
                else:
                    flat_metadata.append(meta_list)
        
        # Process all tiles
        all_detections = postprocessor.process_batch_predictions(
            predictions_list=flat_predictions,
            tile_metadata_list=flat_metadata,
            branch=branch,
        )
        
        self.logger.info(f"Extracted {len(all_detections)} detections from tiles")
        
        # Step 4: Clean overlapping detections
        if clean_overlaps and len(all_detections) > 0:
            self.logger.info("Step 4/5: Cleaning overlapping detections...")
            
            # Mark edge detections
            all_detections = mark_edge_detections(
                detections=all_detections,
                patch_size=self.patch_size,
                margin=self.overlap,
            )
            
            # Clean overlaps
            cleaner = OverlapCleaner(
                detections=all_detections,
                logger=self.logger,
                iou_threshold=iou_threshold,
            )
            
            cleaned_df = cleaner.clean()
            all_detections = cleaned_df.to_dict('records')
            
            self.logger.info(f"After cleaning: {len(all_detections)} detections")
        else:
            self.logger.info("Step 4/5: Skipping overlap cleaning")
        
        # Step 5: Save results
        self.logger.info("Step 5/5: Saving results...")
        
        # Create summary statistics
        summary = create_summary_statistics(
            detections=all_detections,
            branch=branch,
        )
        
        self.logger.info(f"Summary: {summary}")
        
        # Save detections
        wsi_name = wsi_path.stem
        
        if save_json:
            json_path = output_dir / f"{wsi_name}_{branch}_detections.json"
            save_detections_json(
                detections=all_detections,
                output_path=json_path,
                wsi_metadata=wsi_handler.metadata.to_dict(),
                branch=branch,
            )
            self.logger.info(f"Saved JSON: {json_path}")
        
        if save_csv:
            csv_path = output_dir / f"{wsi_name}_{branch}_detections.csv"
            save_detections_csv(
                detections=all_detections,
                output_path=csv_path,
            )
            self.logger.info(f"Saved CSV: {csv_path}")
        
        # Save heatmap if requested
        # Save heatmap if requested
        if save_heatmap:
            from vitaminp.inference.postprocessing import save_heatmap as save_heatmap_func
            
            self.logger.info("Generating heatmap...")
            
            heatmap = create_heatmap_from_seg(
                predictions_list=flat_predictions,
                tile_metadata_list=flat_metadata,
                wsi_shape=(wsi_handler.metadata.height, wsi_handler.metadata.width),
                branch=branch,
            )
            
            heatmap_path = output_dir / f"{wsi_name}_{branch}_heatmap.png"
            save_heatmap_func(heatmap, heatmap_path)  # Use renamed function
            self.logger.info(f"Saved heatmap: {heatmap_path}")
        
        # Close WSI handler
        wsi_handler.close()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        self.logger.info(
            f"Inference complete! Total time: {format_time(total_time)}"
        )
        
        # Prepare results
        results = {
            'detections': all_detections,
            'num_detections': len(all_detections),
            'summary': summary,
            'processing_time': total_time,
            'wsi_metadata': wsi_handler.metadata.to_dict(),
            'branch': branch,
        }
        
        return results
    
    def predict_batch(
        self,
        wsi_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        **kwargs
    ) -> Dict[str, Dict]:
        """Run inference on multiple WSIs.
        
        Args:
            wsi_paths (List[Union[str, Path]]): List of WSI paths
            output_dir (Union[str, Path]): Output directory
            **kwargs: Additional arguments passed to predict()
            
        Returns:
            Dict[str, Dict]: Dictionary mapping WSI names to their results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for wsi_path in wsi_paths:
            wsi_path = Path(wsi_path)
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing WSI {len(all_results)+1}/{len(wsi_paths)}: {wsi_path.name}")
            self.logger.info(f"{'='*60}\n")
            
            try:
                # Create subfolder for this WSI
                wsi_output_dir = output_dir / wsi_path.stem
                wsi_output_dir.mkdir(exist_ok=True)
                
                # Run inference
                results = self.predict(
                    wsi_path=wsi_path,
                    output_dir=wsi_output_dir,
                    **kwargs
                )
                
                all_results[wsi_path.name] = results
                
            except Exception as e:
                self.logger.error(f"Error processing {wsi_path.name}: {e}")
                all_results[wsi_path.name] = {'error': str(e)}
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Batch processing complete! Processed {len(all_results)} WSIs")
        self.logger.info(f"{'='*60}\n")
        
        return all_results
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WSIPredictor(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  checkpoint={self.checkpoint_path.name},\n"
            f"  device={self.device},\n"
            f"  batch_size={self.batch_size},\n"
            f"  patch_size={self.patch_size},\n"
            f"  overlap={self.overlap},\n"
            f"  magnification={self.magnification}\n"
            f")"
        )
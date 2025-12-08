# -*- coding: utf-8 -*-
"""
WSI Inference Module for Vitamin-P
Handles whole slide image inference with patch-based processing
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import json

# Import your existing postprocessing
import sys
sys.path.append('..')
from postprocessing import process_model_outputs

from .utils import (
    get_cell_position,
    get_cell_position_margin,
    get_edge_patch,
    convert_to_global_coordinates,
    create_patch_grid_info
)
from .overlap_cleaner import OverlapCleaner


class WSIInference:
    """Whole Slide Image Inference with patch-based processing"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda:0",
        patch_size: int = 1024,
        overlap: int = 64,
        batch_size: int = 8,
        magnification: int = 40,
        num_classes: int = 6,
        output_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize WSI Inference
        
        Args:
            model (nn.Module): Trained model for inference
            device (str): Device for inference (e.g., 'cuda:0', 'cpu')
            patch_size (int): Size of patches to extract
            overlap (int): Overlap between adjacent patches
            batch_size (int): Batch size for inference
            magnification (int): Magnification level (20 or 40)
            num_classes (int): Number of cell classes
            output_dir (str or Path, optional): Directory to save results
            logger (logging.Logger, optional): Logger instance
        """
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.magnification = magnification
        self.num_classes = num_classes
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Setup logger
        self.logger = logger or self._setup_logger()
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Initialized WSI Inference on {device}")
        self.logger.info(f"Patch size: {patch_size}, Overlap: {overlap}")
        self.logger.info(f"Batch size: {batch_size}, Magnification: {magnification}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger("WSIInference")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_wsi(
        self,
        wsi_array: np.ndarray,
        wsi_name: str = "wsi",
        modality: str = "he",  # 'he' or 'mif'
        target: str = "nuclei",  # 'nuclei' or 'cell'
    ) -> Dict:
        """Process a whole slide image
        
        Args:
            wsi_array (np.ndarray): WSI as numpy array (H, W, C)
            wsi_name (str): Name of the WSI for saving
            modality (str): Image modality ('he' or 'mif')
            target (str): Target for segmentation ('nuclei' or 'cell')
        
        Returns:
            Dict: Results containing:
                - 'cells': List of cell dictionaries
                - 'num_cells': Total number of cells
                - 'wsi_metadata': Metadata about the WSI
        """
        self.logger.info(f"Processing WSI: {wsi_name}")
        self.logger.info(f"WSI shape: {wsi_array.shape}")
        self.logger.info(f"Modality: {modality}, Target: {target}")
        
        # Create patch grid
        wsi_height, wsi_width = wsi_array.shape[:2]
        n_rows, n_cols, patch_coords = create_patch_grid_info(
            wsi_width, wsi_height, self.patch_size, self.overlap
        )
        
        self.logger.info(f"Created patch grid: {n_rows} rows x {n_cols} cols = {len(patch_coords)} patches")
        
        # Extract and process patches
        all_cells = []
        
        # Process patches in batches
        num_batches = int(np.ceil(len(patch_coords) / self.batch_size))
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing patches"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(patch_coords))
                batch_coords = patch_coords[start_idx:end_idx]
                
                # Extract patches
                batch_patches = []
                batch_metadata = []
                
                for row, col, y_start, x_start in batch_coords:
                    patch = wsi_array[
                        y_start:y_start + self.patch_size,
                        x_start:x_start + self.patch_size
                    ]
                    batch_patches.append(patch)
                    batch_metadata.append({
                        'row': row,
                        'col': col,
                        'y_start': y_start,
                        'x_start': x_start
                    })
                
                # Process batch
                batch_cells = self._process_batch(
                    batch_patches,
                    batch_metadata,
                    modality,
                    target
                )
                
                all_cells.extend(batch_cells)
        
        self.logger.info(f"Detected {len(all_cells)} cells before cleaning")
        
        # Clean overlapping cells
        if len(all_cells) > 0:
            cleaned_cells = self._clean_overlaps(all_cells)
        else:
            cleaned_cells = []
        
        self.logger.info(f"Detected {len(cleaned_cells)} cells after cleaning")
        
        # Prepare results
        results = {
            'cells': cleaned_cells,
            'num_cells': len(cleaned_cells),
            'wsi_metadata': {
                'name': wsi_name,
                'shape': wsi_array.shape,
                'patch_size': self.patch_size,
                'overlap': self.overlap,
                'num_patches': len(patch_coords),
                'modality': modality,
                'target': target
            }
        }
        
        # Save results if output directory is provided
        if self.output_dir:
            self._save_results(results, wsi_name)
        
        return results
    
    def _process_batch(
        self,
        patches: List[np.ndarray],
        metadata: List[Dict],
        modality: str,
        target: str
    ) -> List[Dict]:
        """Process a batch of patches
        
        Args:
            patches (List[np.ndarray]): List of patch arrays
            metadata (List[Dict]): List of patch metadata
            modality (str): Image modality
            target (str): Segmentation target
        
        Returns:
            List[Dict]: List of cell dictionaries
        """
        # Prepare batch tensor
        batch_tensor = self._prepare_batch_tensor(patches)
        
        # Run inference
        outputs = self.model(batch_tensor)
        
        # Post-process each patch in the batch
        batch_cells = []
        
        for i in range(len(patches)):
            patch_cells = self._process_single_patch(
                outputs, i, metadata[i], modality, target
            )
            batch_cells.extend(patch_cells)
        
        return batch_cells
    
    def _prepare_batch_tensor(self, patches: List[np.ndarray]) -> torch.Tensor:
        """Prepare batch tensor from list of patches
        
        Args:
            patches (List[np.ndarray]): List of patches
        
        Returns:
            torch.Tensor: Batch tensor (B, C, H, W)
        """
        batch = []
        
        for patch in patches:
            # Normalize to [0, 1] if needed
            if patch.max() > 1.0:
                patch = patch.astype(np.float32) / 255.0
            
            # Convert to tensor and permute to (C, H, W)
            if len(patch.shape) == 2:
                patch = np.expand_dims(patch, axis=-1)
            
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
            batch.append(patch_tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch).to(self.device)
        
        return batch_tensor
    
    def _process_single_patch(
        self,
        outputs: Dict[str, torch.Tensor],
        patch_idx: int,
        metadata: Dict,
        modality: str,
        target: str
    ) -> List[Dict]:
        """Process a single patch and extract cells
        
        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs
            patch_idx (int): Index of patch in batch
            metadata (Dict): Patch metadata
            modality (str): Image modality
            target (str): Segmentation target
        
        Returns:
            List[Dict]: List of cell dictionaries
        """
        # Get the correct output based on modality and target
        output_key_seg = f"{modality}_{target}_seg"
        output_key_hv = f"{modality}_{target}_hv"
        
        # Extract predictions for this patch
        seg_pred = outputs[output_key_seg][patch_idx, 0].cpu().numpy()
        h_map = outputs[output_key_hv][patch_idx, 0].cpu().numpy()
        v_map = outputs[output_key_hv][patch_idx, 1].cpu().numpy()
        
        # Apply HV post-processing using your existing function
        instance_map, inst_info, num_instances = process_model_outputs(
            seg_pred=seg_pred,
            h_map=h_map,
            v_map=v_map,
            magnification=self.magnification
        )
        
        # Convert to cell dictionaries with global coordinates
        patch_cells = self._create_cell_dicts(
            inst_info, metadata, instance_map
        )
        
        return patch_cells
    
    def _create_cell_dicts(
        self,
        inst_info: Dict,
        metadata: Dict,
        instance_map: np.ndarray
    ) -> List[Dict]:
        """Create cell dictionaries from instance info
        
        Args:
            inst_info (Dict): Instance information from postprocessing
            metadata (Dict): Patch metadata
            instance_map (np.ndarray): Instance segmentation map
        
        Returns:
            List[Dict]: List of cell dictionaries
        """
        cells = []
        
        # Calculate stride and offset
        stride = self.patch_size - self.overlap
        offset_y = metadata['row'] * stride
        offset_x = metadata['col'] * stride
        offset = np.array([offset_y, offset_x])
        
        for inst_id, inst_data in inst_info.items():
            # Create cell dict with GLOBAL coordinates
            cell_dict = {
                'bbox': inst_data['bbox'] + offset,
                'centroid': inst_data['centroid'] + offset,
                'contour': inst_data['contour'] + offset,
                'type_prob': inst_data.get('type_prob'),
                'type': inst_data.get('type'),
                'patch_coordinates': [metadata['row'], metadata['col']],
                'offset_global': offset.tolist(),
                'instance_id': inst_id
            }
            
            # Determine cell position status (use LOCAL bbox for this)
            cell_dict['cell_status'] = get_cell_position_margin(
                bbox=inst_data['bbox'],  # Use local bbox
                patch_size=self.patch_size,
                margin=self.overlap
            )
            
            # Check if cell is at edge (use LOCAL bbox)
            if np.max(inst_data['bbox']) >= self.patch_size or np.min(inst_data['bbox']) == 0:
                position = get_cell_position(inst_data['bbox'], self.patch_size)
                cell_dict['edge_position'] = True
                cell_dict['edge_information'] = {
                    'position': position,
                    'edge_patches': get_edge_patch(
                        position, metadata['row'], metadata['col']
                    )
                }
            else:
                cell_dict['edge_position'] = False
            
            cells.append(cell_dict)
        
        return cells
    
    def _clean_overlaps(self, cells: List[Dict]) -> List[Dict]:
        """Clean overlapping cells using OverlapCleaner
        
        Args:
            cells (List[Dict]): List of cell dictionaries
        
        Returns:
            List[Dict]: Cleaned list of cells
        """
        self.logger.info("Cleaning overlapping cells...")
        
        cleaner = OverlapCleaner(cells, self.logger)
        cleaned_df = cleaner.clean_detected_cells()
        
        # Convert back to list of dicts
        cleaned_cells = cleaned_df.to_dict('records')
        
        return cleaned_cells
    
    def _save_results(self, results: Dict, wsi_name: str) -> None:
        """Save results to JSON file
        
        Args:
            results (Dict): Results dictionary
            wsi_name (str): Name of the WSI
        """
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = self.output_dir / f"{wsi_name}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved results to {output_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
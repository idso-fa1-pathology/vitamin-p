# -*- coding: utf-8 -*-
# Post-processing for VitaminP Inference
# Handles HV-map based segmentation outputs

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from pathlib import Path
import json

# Import your HV postprocessor
from vitaminp.postprocessing import process_model_outputs

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")


class VitaminPPostProcessor:
    """Post-process VitaminP model predictions (HV-based segmentation).
    
    Args:
        magnification (int): Magnification level (20 or 40). Default: 40
        logger (Optional[logging.Logger]): Logger instance
    
    Attributes:
        magnification (int): Magnification level
        logger (logging.Logger): Logger
    """
    
    def __init__(
        self,
        magnification: int = 40,
        logger: Optional[logging.Logger] = None,
    ):
        self.magnification = magnification
        self.logger = logger or logging.getLogger(__name__)
    
    def process_tile_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        tile_metadata: Dict,
        branch: str = 'he_nuclei',
    ) -> List[Dict]:
        """Process predictions for a single tile.
        
        Args:
            predictions (Dict[str, torch.Tensor]): Model outputs with keys like 'he_nuclei_seg', 'he_nuclei_hv'
            tile_metadata (Dict): Tile metadata with 'x', 'y', 'row', 'col'
            branch (str): Which branch to process. Options: 'he_nuclei', 'he_cell', 'mif_nuclei', 'mif_cell'
            
        Returns:
            List[Dict]: List of detected cells/nuclei
        """
        # Extract seg and hv maps for the specified branch
        seg_key = f'{branch}_seg'
        hv_key = f'{branch}_hv'
        
        if seg_key not in predictions or hv_key not in predictions:
            self.logger.warning(f"Branch '{branch}' not found in predictions")
            return []
        
        # Convert to numpy (predictions are already on CPU from tile_processor)
        seg_map = predictions[seg_key].squeeze().numpy()  # (H, W)
        hv_maps = predictions[hv_key].squeeze().numpy()   # (2, H, W)
        
        # Transpose HV maps to (H, W, 2)
        h_map = hv_maps[0]  # (H, W)
        v_map = hv_maps[1]  # (H, W)
        
        # Apply HV post-processing
        instance_map, inst_info, num_instances = process_model_outputs(
            seg_pred=seg_map,
            h_map=h_map,
            v_map=v_map,
            magnification=self.magnification,
        )
        
        # Convert to detection format with global coordinates
        detections = []
        for inst_id, info in inst_info.items():
            # Adjust coordinates to global WSI coordinates
            bbox = info['bbox'].copy()
            bbox[0][0] += tile_metadata['y']  # ymin
            bbox[0][1] += tile_metadata['x']  # xmin
            bbox[1][0] += tile_metadata['y']  # ymax
            bbox[1][1] += tile_metadata['x']  # xmax
            
            centroid = info['centroid'].copy()
            centroid[0] += tile_metadata['x']  # x
            centroid[1] += tile_metadata['y']  # y
            
            contour = info['contour'].copy()
            contour[:, 0] += tile_metadata['x']  # x coordinates
            contour[:, 1] += tile_metadata['y']  # y coordinates
            
            detection = {
                'bbox': bbox.tolist(),
                'centroid': centroid.tolist(),
                'contour': contour.tolist(),
                'type': info.get('type'),
                'type_prob': info.get('type_prob'),
                'patch_coordinates': [tile_metadata['row'], tile_metadata['col']],
                'branch': branch,
            }
            
            detections.append(detection)
        
        return detections
    
    def process_batch_predictions(
        self,
        predictions_list: List[Dict[str, torch.Tensor]],
        tile_metadata_list: List[Dict],
        branch: str = 'he_nuclei',
    ) -> List[Dict]:
        """Process a batch of tile predictions.
        
        Args:
            predictions_list (List[Dict]): List of prediction dicts
            tile_metadata_list (List[Dict]): List of tile metadata
            branch (str): Which branch to process
            
        Returns:
            List[Dict]: All detections from all tiles
        """
        all_detections = []
        
        for predictions, tile_meta in zip(predictions_list, tile_metadata_list):
            tile_detections = self.process_tile_predictions(
                predictions=predictions,
                tile_metadata=tile_meta,
                branch=branch,
            )
            all_detections.extend(tile_detections)
        
        return all_detections


def aggregate_detections(detections_list: List[List[Dict]]) -> List[Dict]:
    """Aggregate detections from multiple tiles."""
    all_detections = []
    for tile_detections in detections_list:
        all_detections.extend(tile_detections)
    return all_detections


def save_detections_json(
    detections: List[Dict],
    output_path: Path,
    wsi_metadata: Optional[Dict] = None,
    branch: Optional[str] = None,
) -> None:
    """Save detections to JSON file."""
    output_data = {
        'num_detections': len(detections),
        'detections': detections,
        'branch': branch,
    }
    
    if wsi_metadata is not None:
        output_data['wsi_metadata'] = wsi_metadata
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def save_detections_csv(detections: List[Dict], output_path: Path) -> None:
    """Save detections to CSV file."""
    import pandas as pd
    
    flattened = []
    for det in detections:
        flat_det = {
            'centroid_x': det['centroid'][0],
            'centroid_y': det['centroid'][1],
            'bbox_xmin': det['bbox'][0][1],
            'bbox_ymin': det['bbox'][0][0],
            'bbox_xmax': det['bbox'][1][1],
            'bbox_ymax': det['bbox'][1][0],
            'branch': det.get('branch', ''),
        }
        flattened.append(flat_det)
    
    df = pd.DataFrame(flattened)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def create_summary_statistics(detections: List[Dict], branch: Optional[str] = None) -> Dict:
    """Create summary statistics from detections."""
    if len(detections) == 0:
        return {
            'total_detections': 0,
            'branch': branch,
        }
    
    summary = {
        'total_detections': len(detections),
        'branch': branch,
    }
    
    return summary


def create_heatmap_from_seg(
    predictions_list: List[Dict[str, torch.Tensor]],
    tile_metadata_list: List[Dict],
    wsi_shape: Tuple[int, int],
    branch: str = 'he_nuclei',
) -> np.ndarray:
    """Create heatmap from segmentation predictions.
    
    Args:
        predictions_list: List of prediction dicts
        tile_metadata_list: List of tile metadata
        wsi_shape: (height, width) of WSI
        branch: Which branch to visualize
        
    Returns:
        np.ndarray: Heatmap (H, W) with values 0-1
    """
    height, width = wsi_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    
    seg_key = f'{branch}_seg'
    
    for pred, tile_meta in zip(predictions_list, tile_metadata_list):
        if seg_key not in pred:
            continue
        
        x = tile_meta['x']
        y = tile_meta['y']
        w = tile_meta['width']
        h = tile_meta['height']
        
        # Get segmentation probability
        seg_prob = pred[seg_key].squeeze().numpy()
        
        # Resize if needed
        if seg_prob.shape != (h, w):
            if CV2_AVAILABLE:
                seg_prob = cv2.resize(seg_prob, (w, h))
            else:
                continue
        
        # Add to heatmap
        y_end = min(y + h, height)
        x_end = min(x + w, width)
        
        heatmap[y:y_end, x:x_end] += seg_prob[:y_end-y, :x_end-x]
        count_map[y:y_end, x:x_end] += 1
    
    # Average overlapping regions
    count_map[count_map == 0] = 1
    heatmap = heatmap / count_map
    
    return heatmap


def save_heatmap(heatmap: np.ndarray, output_path: Path) -> None:
    """Save heatmap as image."""
    if not CV2_AVAILABLE:
        print("OpenCV required for saving heatmap")
        return
    
    # Normalize to 0-255
    heatmap_norm = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), heatmap_colored)
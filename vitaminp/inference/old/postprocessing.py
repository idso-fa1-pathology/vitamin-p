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

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")


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
        """Process predictions for a single tile."""
        
        seg_key = f'{branch}_seg'
        hv_key = f'{branch}_hv'
        
        if seg_key not in predictions or hv_key not in predictions:
            self.logger.warning(f"Branch '{branch}' not found in predictions")
            return []
        
        # Convert to numpy - CAREFUL WITH DIMENSIONS!
        seg_tensor = predictions[seg_key]
        hv_tensor = predictions[hv_key]
        
        # ✅ DEBUG: Log shapes before processing
        self.logger.debug(f"Raw shapes - seg: {seg_tensor.shape}, hv: {hv_tensor.shape}")
        
        # Handle segmentation map: [1, H, W] or [H, W] -> [H, W]
        if seg_tensor.dim() == 3:
            seg_map = seg_tensor.squeeze(0).cpu().numpy()  # Remove batch/channel dim
        else:
            seg_map = seg_tensor.cpu().numpy()
        
        # ✅ FIX: Handle HV maps carefully
        # Expected: [2, H, W] - DO NOT SQUEEZE!
        # If it's [1, 2, H, W], remove batch dimension only
        # If it's [2, H, W], keep as-is
        if hv_tensor.dim() == 4:  # [1, 2, H, W]
            hv_maps = hv_tensor.squeeze(0).cpu().numpy()  # -> [2, H, W]
        elif hv_tensor.dim() == 3:  # [2, H, W] - correct shape
            hv_maps = hv_tensor.cpu().numpy()
        elif hv_tensor.dim() == 2:  # [2, 512] - WRONG! This is your bug
            self.logger.error(
                f"HV tensor has wrong shape: {hv_tensor.shape}. "
                f"Expected [2, H, W], got 2D array. This indicates an issue in model output or tile processing."
            )
            return []
        else:
            self.logger.error(f"Unexpected HV tensor shape: {hv_tensor.shape}")
            return []
        
        # ✅ Verify HV maps shape
        if hv_maps.ndim != 3 or hv_maps.shape[0] != 2:
            self.logger.error(
                f"Invalid HV maps shape: {hv_maps.shape}. Expected [2, H, W]"
            )
            return []
        
        # Crop to original size if tile was padded
        original_h = tile_metadata.get('original_height', seg_map.shape[0])
        original_w = tile_metadata.get('original_width', seg_map.shape[1])
        
        if seg_map.shape[0] != original_h or seg_map.shape[1] != original_w:
            seg_map = seg_map[:original_h, :original_w]
            hv_maps = hv_maps[:, :original_h, :original_w]
        
        # Extract H and V maps
        h_map = hv_maps[0]  # [H, W]
        v_map = hv_maps[1]  # [H, W]
        
        # Verify all shapes match
        if not (seg_map.shape == h_map.shape == v_map.shape):
            self.logger.error(
                f"Shape mismatch! seg: {seg_map.shape}, h_map: {h_map.shape}, v_map: {v_map.shape}"
            )
            return []
        
        # Apply HV post-processing
        instance_map, inst_info, num_instances = process_model_outputs(
            seg_pred=seg_map,
            h_map=h_map,
            v_map=v_map,
            magnification=self.magnification,
        )
        
        # Apply HV post-processing
        instance_map, inst_info, num_instances = process_model_outputs(
            seg_pred=seg_map,
            h_map=h_map,
            v_map=v_map,
            magnification=self.magnification,
        )
        
        # Convert to detection format with global coordinates
        detections = []
        tile_x = tile_metadata['x']
        tile_y = tile_metadata['y']
        
        for inst_id, info in inst_info.items():
            bbox = info['bbox'].copy()
            bbox[0][0] = bbox[0][0] + tile_y  # ymin
            bbox[0][1] = bbox[0][1] + tile_x  # xmin
            bbox[1][0] = bbox[1][0] + tile_y  # ymax
            bbox[1][1] = bbox[1][1] + tile_x  # xmax
            
            centroid = info['centroid'].copy()
            centroid[0] = centroid[0] + tile_x  # x
            centroid[1] = centroid[1] + tile_y  # y
            
            contour = info['contour'].copy()
            contour[:, 0] = contour[:, 0] + tile_x  # x
            contour[:, 1] = contour[:, 1] + tile_y  # y
            
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
        """Process a batch of tile predictions."""
        all_detections = []
        
        self.logger.info(f"Processing {len(predictions_list)} tiles")
        
        for idx, (predictions, tile_meta) in enumerate(zip(predictions_list, tile_metadata_list)):
            tile_detections = self.process_tile_predictions(
                predictions=predictions,
                tile_metadata=tile_meta,
                branch=branch,
            )
            
            # ✅ ADD THIS DEBUG LOG
            if idx < 5:  # Log first 5 tiles
                self.logger.info(
                    f"  Tile {idx} at ({tile_meta.get('row', '?')}, {tile_meta.get('col', '?')}): "
                    f"{len(tile_detections)} detections"
                )
            
            all_detections.extend(tile_detections)
        
        self.logger.info(f"Total detections from all tiles: {len(all_detections)}")
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


def save_detections_geojson(
    detections: List[Dict],
    output_path: Path,
    wsi_metadata: Optional[Dict] = None,
    branch: Optional[str] = None,
) -> None:
    """Save detections to GeoJSON file with proper polygon geometries.
    
    GeoJSON format is useful for visualization in tools like QuPath, ASAP, etc.
    
    Args:
        detections: List of detection dictionaries with 'contour', 'centroid', 'bbox', etc.
        output_path: Path to save GeoJSON file
        wsi_metadata: Optional WSI metadata
        branch: Branch name (e.g., 'he_nuclei')
    """
    import uuid
    
    features = []
    
    for idx, det in enumerate(detections):
        # Create polygon from contour
        contour = det['contour']
        
        # GeoJSON requires closed polygons (first point == last point)
        if not np.array_equal(contour[0], contour[-1]):
            contour = contour + [contour[0]]
        
        # Convert to GeoJSON coordinate format [[x, y], [x, y], ...]
        coordinates = [[float(pt[0]), float(pt[1])] for pt in contour]
        
        # Create feature
        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]  # GeoJSON polygons need list of rings
            },
            "properties": {
                "objectType": "detection",
                "name": f"Cell_{idx+1}",
                "classification": {
                    "name": det.get('branch', branch or 'unknown'),
                    "colorRGB": -3140401  # Default color (can customize)
                },
                "centroid_x": float(det['centroid'][0]),
                "centroid_y": float(det['centroid'][1]),
                "bbox": det['bbox'],
                "area": float(calculate_contour_area(det['contour'])),
                "type": det.get('type'),
                "type_prob": det.get('type_prob'),
                "branch": det.get('branch', branch),
            }
        }
        
        features.append(feature)
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "num_detections": len(detections),
            "branch": branch,
        }
    }
    
    if wsi_metadata is not None:
        geojson["metadata"]["wsi"] = wsi_metadata
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)


def calculate_contour_area(contour: List[List[float]]) -> float:
    """Calculate area of a polygon contour using the Shoelace formula."""
    contour = np.array(contour)
    x = contour[:, 0]
    y = contour[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


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


def create_overlay_visualization(
    wsi_path: Path,
    detections: List[Dict],
    output_path: Path,
    downsample: int = 8,
    contour_color: Tuple[int, int, int] = (0, 255, 0),  # Green
    contour_thickness: int = 2,
    show_centroids: bool = True,
    centroid_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    centroid_radius: int = 3,
) -> None:
    """Create visualization with cell contours overlaid on original image.
    
    Args:
        wsi_path: Path to original WSI/image
        detections: List of detections with 'contour' and 'centroid'
        output_path: Where to save the visualization
        downsample: Downsample factor for large images. Default: 8
        contour_color: RGB color for contours. Default: Green (0, 255, 0)
        contour_thickness: Thickness of contour lines. Default: 2
        show_centroids: Whether to show centroids. Default: True
        centroid_color: RGB color for centroids. Default: Red (255, 0, 0)
        centroid_radius: Radius of centroid dots. Default: 3
    """
    if not CV2_AVAILABLE:
        print("OpenCV required for overlay visualization")
        return
    
    # Load the original image
    try:
        # Try loading with PIL first (works for PNG, JPEG, etc.)
        if PIL_AVAILABLE:
            from PIL import Image
            img_pil = Image.open(str(wsi_path))
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            img = np.array(img_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        else:
            # Fallback to OpenCV
            img = cv2.imread(str(wsi_path))
            if img is None:
                print(f"Could not load image: {wsi_path}")
                return
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Downsample if image is large
    original_h, original_w = img.shape[:2]
    if downsample > 1:
        new_w = original_w // downsample
        new_h = original_h // downsample
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_factor = 1.0 / downsample
    else:
        scale_factor = 1.0
    
    # Draw each detection
    for det in detections:
        # Draw contour
        contour = np.array(det['contour'], dtype=np.float32)
        contour_scaled = (contour * scale_factor).astype(np.int32)
        cv2.polylines(img, [contour_scaled], isClosed=True, color=contour_color, thickness=contour_thickness)
        
        # Draw centroid if requested
        if show_centroids:
            centroid = det['centroid']
            cx = int(centroid[0] * scale_factor)
            cy = int(centroid[1] * scale_factor)
            cv2.circle(img, (cx, cy), centroid_radius, centroid_color, -1)
    
    # Add text with detection count
    text = f"Detections: {len(detections)}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def save_heatmap(heatmap: np.ndarray, output_path: Path) -> None:
    """Save heatmap as image.
    
    DEPRECATED: Use create_overlay_visualization() instead for better visualization.
    This function only shows the heatmap, not the original image.
    """
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
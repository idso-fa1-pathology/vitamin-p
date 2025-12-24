# -*- coding: utf-8 -*-
# Overlap Cell Cleaner for VitaminP Inference
# Removes duplicate detections from overlapping tiles

import logging
from typing import List, Dict, Set
from collections import deque
import numpy as np
import pandas as pd

try:
    from shapely import strtree
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: Shapely not available. Install with: pip install shapely")


class OverlapCleaner:
    """Remove overlapping/duplicate detections from WSI inference.
    
    When processing WSI with overlapping tiles, the same object may be detected
    multiple times. This class identifies and removes duplicates, keeping only
    the best detection for each object.
    
    Args:
        detections (List[Dict]): List of detection dictionaries
        logger (logging.Logger): Logger instance
        iou_threshold (float): IoU threshold for considering detections as duplicates. Default: 0.5
        max_iterations (int): Maximum iterations for cleaning. Default: 20
    
    Attributes:
        detections_df (pd.DataFrame): DataFrame with all detections
        logger (logging.Logger): Logger
        iou_threshold (float): IoU threshold
        max_iterations (int): Maximum cleaning iterations
    """
    
    def __init__(
        self,
        detections: List[Dict],
        logger: logging.Logger,
        iou_threshold: float = 0.5,
        max_iterations: int = 20,
    ):
        if not SHAPELY_AVAILABLE:
            raise RuntimeError(
                "Shapely is required for overlap cleaning. "
                "Install with: pip install shapely"
            )
        
        self.logger = logger
        self.iou_threshold = iou_threshold
        self.max_iterations = max_iterations
        
        # Validate input
        self._validate_detections(detections)
        
        # Convert to DataFrame for easier processing
        self.detections_df = pd.DataFrame(detections)
        
        # Add patch coordinate columns for faster lookup
        if "patch_coordinates" in self.detections_df.columns:
            self.detections_df[["patch_row", "patch_col"]] = pd.DataFrame(
                self.detections_df["patch_coordinates"].tolist(),
                index=self.detections_df.index
            )
    
    def _validate_detections(self, detections: List[Dict]) -> None:
        """Validate that detections have required fields.
        
        Args:
            detections (List[Dict]): List of detection dictionaries
            
        Raises:
            ValueError: If required fields are missing
        """
        if len(detections) == 0:
            return
        
        required_keys = ["bbox", "centroid"]
        first_detection = detections[0]
        
        for key in required_keys:
            if key not in first_detection:
                raise ValueError(f"Detection missing required key: {key}")
    
    def clean(self) -> pd.DataFrame:
        """Main cleaning method to remove overlapping detections.
        
        Returns:
            pd.DataFrame: Cleaned detections DataFrame
        """
        self.logger.info(f"Starting overlap cleaning with {len(self.detections_df)} detections")
        
        if len(self.detections_df) == 0:
            return self.detections_df
        
        # Separate detections by position
        mid_detections = self._get_mid_detections()
        edge_detections = self._get_edge_detections()
        
        self.logger.info(
            f"Split detections: {len(mid_detections)} mid, {len(edge_detections)} edge"
        )
        
        # Clean edge detections
        cleaned_edge = self._remove_duplicates(edge_detections)
        
        # Combine and return
        cleaned_all = pd.concat([mid_detections, cleaned_edge]).sort_index()
        
        self.logger.info(
            f"Cleaning complete: {len(self.detections_df)} -> {len(cleaned_all)} "
            f"({len(self.detections_df) - len(cleaned_all)} removed)"
        )
        
        return cleaned_all
    
    def _get_mid_detections(self) -> pd.DataFrame:
        """Get detections that are fully inside tiles (not touching edges).
        
        Returns:
            pd.DataFrame: Mid detections
        """
        if "cell_status" in self.detections_df.columns:
            return self.detections_df[self.detections_df["cell_status"] == 0].copy()
        else:
            # If cell_status not available, return empty DataFrame
            return pd.DataFrame(columns=self.detections_df.columns)
    
    def _get_edge_detections(self) -> pd.DataFrame:
        """Get detections that touch tile edges.
        
        Returns:
            pd.DataFrame: Edge detections
        """
        if "cell_status" in self.detections_df.columns:
            return self.detections_df[self.detections_df["cell_status"] != 0].copy()
        else:
            # If cell_status not available, assume all are edge detections
            return self.detections_df.copy()
    
    def _remove_duplicates(self, detections: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate/overlapping detections iteratively.
        
        Args:
            detections (pd.DataFrame): Detections to clean
            
        Returns:
            pd.DataFrame: Cleaned detections
        """
        if len(detections) == 0:
            return detections
        
        cleaned = detections.copy()
        
        for iteration in range(self.max_iterations):
            # Create polygon list
            poly_list = self._create_polygon_list(cleaned)
            
            if len(poly_list) == 0:
                break
            
            # Build spatial index
            tree = strtree.STRtree(poly_list)
            
            # Find and resolve overlaps
            keep_indices = []
            checked_indices = set()
            overlap_count = 0
            
            for query_poly in poly_list:
                if query_poly.uid not in checked_indices:
                    # Query overlapping polygons
                    intersected = tree.query(query_poly)
                    
                    if len(intersected) > 1:
                        # Find overlapping polygons (excluding self)
                        overlaps = []
                        for inter_poly in intersected:
                            if inter_poly.uid != query_poly.uid and inter_poly.uid not in checked_indices:
                                # Calculate overlap
                                intersection_area = query_poly.intersection(inter_poly).area
                                iou = intersection_area / min(query_poly.area, inter_poly.area)
                                
                                if iou > self.iou_threshold:
                                    overlaps.append(inter_poly)
                                    checked_indices.add(inter_poly.uid)
                                    overlap_count += 1
                        
                        if len(overlaps) > 0:
                            # Select best polygon (largest area)
                            all_candidates = [query_poly] + overlaps
                            areas = [p.area for p in all_candidates]
                            best_idx = np.argmax(areas)
                            keep_indices.append(all_candidates[best_idx].uid)
                        else:
                            keep_indices.append(query_poly.uid)
                    else:
                        keep_indices.append(query_poly.uid)
                    
                    checked_indices.add(query_poly.uid)
            
            self.logger.info(
                f"Iteration {iteration + 1}: Found {overlap_count} overlaps, "
                f"keeping {len(keep_indices)}/{len(cleaned)} detections"
            )
            
            if overlap_count == 0:
                self.logger.info("No more overlaps found, stopping early")
                break
            
            # Update cleaned detections
            cleaned = cleaned.loc[cleaned.index.isin(keep_indices)].copy()
        
        return cleaned
    
    def _create_polygon_list(self, detections: pd.DataFrame) -> List[Polygon]:
        """Create list of Shapely polygons from detections.
        
        Args:
            detections (pd.DataFrame): Detections DataFrame
            
        Returns:
            List[Polygon]: List of polygons with uid attribute
        """
        poly_list = []
        
        for idx, row in detections.iterrows():
            # Create polygon from bbox or contour
            if "contour" in row and row["contour"] is not None:
                try:
                    poly = Polygon(row["contour"])
                except:
                    # Fall back to bbox if contour is invalid
                    poly = self._bbox_to_polygon(row["bbox"])
            else:
                poly = self._bbox_to_polygon(row["bbox"])
            
            # Fix invalid polygons
            if not poly.is_valid:
                poly = poly.buffer(0)
                if isinstance(poly, MultiPolygon):
                    # Take largest polygon
                    poly = max(poly.geoms, key=lambda p: p.area)
            
            # Attach unique identifier
            poly.uid = idx
            poly_list.append(poly)
        
        return poly_list
    
    def _bbox_to_polygon(self, bbox: List) -> Polygon:
        """Convert bounding box to Shapely polygon.
        
        Args:
            bbox (List): Bounding box [[xmin, ymin], [xmax, ymax]]
            
        Returns:
            Polygon: Shapely polygon
        """
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]
        
        coords = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),  # Close the polygon
        ]
        
        return Polygon(coords)


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1 (np.ndarray): First bbox [[xmin, ymin], [xmax, ymax]]
        bbox2 (np.ndarray): Second bbox [[xmin, ymin], [xmax, ymax]]
        
    Returns:
        float: IoU score (0-1)
    """
    # Extract coordinates
    x1_min, y1_min = bbox1[0]
    x1_max, y1_max = bbox1[1]
    x2_min, y2_min = bbox2[0]
    x2_max, y2_max = bbox2[1]
    
    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Calculate union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def mark_edge_detections(
    detections: List[Dict],
    patch_size: int = 1024,
    margin: int = 64,
) -> List[Dict]:
    """Mark detections based on their position in the patch.
    
    Adds 'cell_status' field:
        0 = mid (not touching edges)
        1-8 = edge position (clockwise from top-left)
    
    Args:
        detections (List[Dict]): List of detection dictionaries with 'bbox' key
        patch_size (int): Size of the patch. Default: 1024
        margin (int): Margin size for edge detection. Default: 64
        
    Returns:
        List[Dict]: Detections with 'cell_status' field added
    """
    for detection in detections:
        bbox = np.array(detection["bbox"])
        cell_status = _get_cell_status(bbox, patch_size, margin)
        detection["cell_status"] = cell_status
    
    return detections


def _get_cell_status(bbox: np.ndarray, patch_size: int, margin: int) -> int:
    """Get cell status based on position in patch.
    
    Status codes:
        0 = mid
        1 = top-left, 2 = top, 3 = top-right
        4 = right, 5 = bottom-right, 6 = bottom
        7 = bottom-left, 8 = left
    
    Args:
        bbox (np.ndarray): Bounding box [[xmin, ymin], [xmax, ymax]]
        patch_size (int): Patch size
        margin (int): Margin size
        
    Returns:
        int: Cell status code
    """
    xmin, ymin = bbox[0]
    xmax, ymax = bbox[1]
    
    # Check if in margin
    if xmax < margin or ymin < margin or xmax > patch_size - margin or ymax > patch_size - margin:
        return 0  # Mid
    
    # Determine edge position
    if ymin < margin:
        if xmin < margin:
            return 1  # top-left
        elif xmax > patch_size - margin:
            return 3  # top-right
        else:
            return 2  # top
    elif ymax > patch_size - margin:
        if xmin < margin:
            return 7  # bottom-left
        elif xmax > patch_size - margin:
            return 5  # bottom-right
        else:
            return 6  # bottom
    elif xmin < margin:
        return 8  # left
    elif xmax > patch_size - margin:
        return 4  # right
    else:
        return 0  # mid
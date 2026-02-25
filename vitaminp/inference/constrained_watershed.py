#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HV-Constrained watershed for cell segmentation.

Uses detected nuclei as seeds, model HV maps as boundaries, 
and recovers orphan cells without nuclei to guarantee accurate, 
1-to-1 or independent cell geometries.
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


def apply_hv_constrained_watershed(nuclei_labels, cell_prob_map, cell_h_map, cell_v_map, prob_threshold=0.5, min_orphan_area=30):
    """Forces 1 cell per nucleus, uses HV edges as walls, and rescues orphan cells.

    Args:
        nuclei_labels: Instance-labeled nuclei map (H, W) with unique int IDs per nucleus.
        cell_prob_map: Raw cell segmentation probability map (H, W) in [0, 1].
        cell_h_map: Horizontal distance map from the model (H, W).
        cell_v_map: Vertical distance map from the model (H, W).
        prob_threshold: Probability threshold for binarizing cell_prob_map.
        min_orphan_area: Minimum area in pixels to consider an orphan blob as a valid cell.

    Returns:
        constrained_cells: Label map (H, W) where each pixel belongs to exactly one cell
                           (or 0 for background).
    """
    cell_binary = cell_prob_map > prob_threshold
    nuclei_binary = nuclei_labels > 0
    
    # Convert to uint8 for OpenCV morphological operations
    combined_mask = np.logical_or(cell_binary, nuclei_binary).astype(np.uint8)
    
    # Fill in small indents and smooth boundaries before watershed
    kernel_global = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_global)
    
    # Convert back to boolean for downstream distance transforms and masks
    combined_mask = combined_mask > 0
    
    # ---- 1. ORPHAN RECOVERY (Find cells without nuclei) ----
    markers = nuclei_labels.astype(np.int32).copy()
    max_marker_id = np.max(markers) if np.max(markers) > 0 else 0
    
    # Dilate nuclei slightly to prevent the fringes of nucleated cells from being flagged as orphans
    dilated_nuclei = cv2.dilate(nuclei_binary.astype(np.uint8), np.ones((7,7), np.uint8), iterations=2)
    orphan_mask = (cell_binary.astype(np.uint8) > 0) & (dilated_nuclei == 0)
    
    # Find independent blobs in the orphan mask
    num_labels, orphan_labels = cv2.connectedComponents(orphan_mask.astype(np.uint8))
    for i in range(1, num_labels):
        mask_i = orphan_labels == i
        if np.sum(mask_i) > min_orphan_area:  
            max_marker_id += 1
            # Find the thickest center point of this orphan cell to drop a new seed
            dist_i = distance_transform_edt(mask_i)
            cy, cx = np.unravel_index(np.argmax(dist_i), dist_i.shape)
            markers[cy, cx] = max_marker_id
    # --------------------------------------------------------

    # 2. Get HV Edges (Boundary Walls)
    sobel_h_x = cv2.Sobel(cell_h_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_h_y = cv2.Sobel(cell_h_map, cv2.CV_64F, 0, 1, ksize=3)
    sobel_v_x = cv2.Sobel(cell_v_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_v_y = cv2.Sobel(cell_v_map, cv2.CV_64F, 0, 1, ksize=3)
    
    hv_edges = np.sqrt(sobel_h_x**2 + sobel_h_y**2 + sobel_v_x**2 + sobel_v_y**2)
    if hv_edges.max() > 0:
        hv_edges = hv_edges / hv_edges.max()
        
    # 3. Distance from background (Basins)
    dist = distance_transform_edt(combined_mask)
    if dist.max() > 0:
        dist_norm = dist / dist.max()
    else:
        dist_norm = dist
        
    # 4. Model uncertainty
    uncertainty = 1.0 - cell_prob_map
    
    # 5. FINAL TOPOGRAPHY
    topography = -dist_norm + (2.0 * hv_edges) + uncertainty
    
    # Run watershed with the updated markers (nuclei + orphan seeds)
    constrained_cells = watershed(topography, markers, mask=combined_mask)
    
    return constrained_cells


def extract_instances_from_labels(label_map, min_area=0):
    """Extract per-instance info from a label map (matches postprocessing format).

    Args:
        label_map: (H, W) integer label map. Background = 0.
        min_area:  Minimum instance area in pixels. Smaller instances are skipped.

    Returns:
        inst_info_dict: {int_id: {'centroid', 'contour', 'bbox', 'area', 'type'}}
    """
    inst_info_dict = {}

    instance_ids = np.unique(label_map)
    instance_ids = instance_ids[instance_ids > 0]

    for inst_id in instance_ids:
        mask = (label_map == inst_id).astype(np.uint8)
        area = int(mask.sum())

        if area < min_area:
            continue

        # Bounding box
        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # Centroid
        cx = float(xs.mean())
        cy = float(ys.mean())
        centroid = np.array([cx, cy], dtype=np.float32)

        # Contour — extract only from the bounding-box ROI for speed
        roi_y_start = max(0, y_min - 1)
        roi_y_end   = min(label_map.shape[0], y_max + 2)
        roi_x_start = max(0, x_min - 1)
        roi_x_end   = min(label_map.shape[1], x_max + 2)

        roi_mask = mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # Final smoothing to ensure perfect, indent-free polygons
        kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_roi)

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 3:
                contour = contour.astype(np.float32).squeeze()          # (N, 2)
                if contour.ndim == 2:
                    contour[:, 0] += roi_x_start                        # shift x back
                    contour[:, 1] += roi_y_start                        # shift y back
                else:
                    contour = np.array([
                        [x_min, y_min], [x_max, y_min],
                        [x_max, y_max], [x_min, y_max]
                    ], dtype=np.float32)
            else:
                contour = np.array([
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max]
                ], dtype=np.float32)
        else:
            contour = np.array([
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ], dtype=np.float32)

        inst_info_dict[int(inst_id)] = {
            'centroid': centroid,
            'contour': contour,
            'bbox':    np.array([[x_min, y_min], [x_max, y_max]], dtype=np.float32),
            'area':    float(area),
            'type':    'unknown',
        }

    return inst_info_dict
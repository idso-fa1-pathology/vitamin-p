#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nuclei-constrained watershed for cell segmentation.

Uses detected nuclei as seeds and the model's cell prediction as the
boundary mask, guaranteeing a strict 1-to-1 nucleus→cell relationship.
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


def apply_nuclei_constrained_watershed(nuclei_labels, cell_seg_map, cell_threshold=0.5):
    """Refine cell boundaries using nuclei as watershed seeds.

    Args:
        nuclei_labels: Instance-labeled nuclei map (H, W) with unique int IDs per nucleus.
                       Background = 0.  This is the inst_map output from process_model_outputs
                       run on the nuclei branch.
        cell_seg_map:  Raw cell segmentation probability map (H, W) in [0, 1].
                       This is tile_pred['seg'] from the cell branch — do NOT threshold it
                       before passing it in; the function handles that internally.
        cell_threshold: Probability threshold for binarizing cell_seg_map (default 0.5).

    Returns:
        constrained_cells: Label map (H, W) where each pixel belongs to exactly one cell
                           (or 0 for background).  Labels match the input nuclei IDs, so
                           label N in the output corresponds to nucleus N in nuclei_labels.
    """
    # --- Markers: the nuclei instance map is used directly as seeds ---
    markers = nuclei_labels.astype(np.int32)

    # --- Mask: union of cell prediction and nuclei binary ---
    # OR-ing with nuclei ensures no seed ever falls outside the allowed region,
    # even if the cell prediction slightly under-segments around that nucleus.
    cell_binary = cell_seg_map > cell_threshold
    nuclei_binary = markers > 0
    combined_mask = np.logical_or(cell_binary, nuclei_binary)

    # --- Energy landscape: distance from the combined mask boundary ---
    # Pixels deep inside a cell get high distance values (low energy when negated);
    # pixels near the edge get low distance values (high energy = watershed ridge).
    dist = distance_transform_edt(combined_mask)

    # --- Watershed ---
    # -dist   → topography (inverted so basins are at cell centres)
    # markers → seeds (one per nucleus)
    # mask    → hard boundary; watershed never floods outside this region
    constrained_cells = watershed(-dist, markers, mask=combined_mask)

    return constrained_cells


def extract_instances_from_labels(label_map, min_area=0):
    """Extract per-instance info from a label map (same format as postprocessing output).

    This produces the same dict structure that process_model_outputs returns in
    inst_info_dict, so the rest of the pipeline (coordinate conversion, boundary
    detection, export) needs no changes.

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
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 3:
                contour = contour.astype(np.float32).squeeze()          # (N, 2)
                contour[:, 0] += roi_x_start                            # shift x back
                contour[:, 1] += roi_y_start                            # shift y back
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
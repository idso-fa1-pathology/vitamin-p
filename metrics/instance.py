"""
Instance segmentation metrics for evaluating instance-level predictions
Includes PQ, AJI, AJI+, DQ, SQ metrics used in cell/nuclei segmentation
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_fast_pq(true, pred, match_iou=0.5):
    """
    Fast Panoptic Quality (PQ) computation
    PQ = (TP * SQ) / (TP + 0.5*FP + 0.5*FN)
    
    Args:
        true: Ground truth instance map (H, W), each instance has unique ID > 0
        pred: Predicted instance map (H, W), each instance has unique ID > 0
        match_iou: IoU threshold for matching instances
    
    Returns:
        pq: Panoptic Quality
        dq: Detection Quality (TP / (TP + 0.5*FP + 0.5*FN))
        sq: Segmentation Quality (average IoU of matched instances)
    """
    # Convert to numpy if torch tensor
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    true = true.astype(np.int32)
    pred = pred.astype(np.int32)
    
    # Get unique instance IDs (excluding background 0)
    true_id_list = np.unique(true)
    pred_id_list = np.unique(pred)
    
    true_id_list = true_id_list[true_id_list != 0]
    pred_id_list = pred_id_list[pred_id_list != 0]
    
    # Quick return for empty cases
    if len(true_id_list) == 0 and len(pred_id_list) == 0:
        return 1.0, 1.0, 1.0
    if len(true_id_list) == 0:
        return 0.0, 0.0, 0.0
    if len(pred_id_list) == 0:
        return 0.0, 0.0, 0.0
    
    # Compute IoU for all pairs
    true_masks = [(true == i) for i in true_id_list]
    pred_masks = [(pred == i) for i in pred_id_list]
    
    # Pairwise IoU matrix
    pairwise_iou = np.zeros((len(true_id_list), len(pred_id_list)))
    
    for i, true_mask in enumerate(true_masks):
        for j, pred_mask in enumerate(pred_masks):
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            if union > 0:
                pairwise_iou[i, j] = intersection / union
    
    # Match instances using Hungarian algorithm
    paired_true = []
    paired_pred = []
    paired_iou = []
    
    if pairwise_iou.shape[0] > 0 and pairwise_iou.shape[1] > 0:
        # Find best matches
        true_indices, pred_indices = linear_sum_assignment(-pairwise_iou)
        
        for t_idx, p_idx in zip(true_indices, pred_indices):
            iou_val = pairwise_iou[t_idx, p_idx]
            if iou_val > match_iou:
                paired_true.append(t_idx)
                paired_pred.append(p_idx)
                paired_iou.append(iou_val)
    
    # Calculate metrics
    tp = len(paired_iou)
    fp = len(pred_id_list) - tp
    fn = len(true_id_list) - tp
    
    # Segmentation Quality (average IoU of matched pairs)
    sq = np.mean(paired_iou) if len(paired_iou) > 0 else 0.0
    
    # Detection Quality
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    
    # Panoptic Quality
    pq = sq * dq
    
    return pq, dq, sq


def panoptic_quality(true, pred, match_iou=0.5):
    """
    Panoptic Quality metric (alias for get_fast_pq)
    
    Args:
        true: Ground truth instance map (H, W)
        pred: Predicted instance map (H, W)
        match_iou: IoU threshold for matching
    
    Returns:
        PQ score
    """
    pq, _, _ = get_fast_pq(true, pred, match_iou)
    return pq


def detection_quality(true, pred, match_iou=0.5):
    """
    Detection Quality metric
    
    Args:
        true: Ground truth instance map (H, W)
        pred: Predicted instance map (H, W)
        match_iou: IoU threshold for matching
    
    Returns:
        DQ score
    """
    _, dq, _ = get_fast_pq(true, pred, match_iou)
    return dq


def segmentation_quality(true, pred, match_iou=0.5):
    """
    Segmentation Quality metric
    
    Args:
        true: Ground truth instance map (H, W)
        pred: Predicted instance map (H, W)
        match_iou: IoU threshold for matching
    
    Returns:
        SQ score
    """
    _, _, sq = get_fast_pq(true, pred, match_iou)
    return sq


def get_fast_aji(true, pred):
    """
    Aggregated Jaccard Index (AJI)
    
    AJI = sum(intersections) / sum(unions + unmatched_pred_areas)
    
    Args:
        true: Ground truth instance map (H, W)
        pred: Predicted instance map (H, W)
    
    Returns:
        AJI score
    """
    # Convert to numpy if torch tensor
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    true = true.astype(np.int32)
    pred = pred.astype(np.int32)
    
    # Get unique IDs
    true_id_list = np.unique(true)
    pred_id_list = np.unique(pred)
    
    true_id_list = true_id_list[true_id_list != 0]
    pred_id_list = pred_id_list[pred_id_list != 0]
    
    # Quick return for empty cases
    if len(true_id_list) == 0 and len(pred_id_list) == 0:
        return 1.0
    if len(true_id_list) == 0:
        return 0.0
    
    # Compute pairwise IoU
    true_masks = {i: (true == i) for i in true_id_list}
    pred_masks = {i: (pred == i) for i in pred_id_list}
    
    # For each true instance, find best matching pred instance
    total_intersection = 0
    total_union = 0
    matched_pred = set()
    
    for true_id in true_id_list:
        true_mask = true_masks[true_id]
        max_iou = 0
        max_pred_id = None
        
        for pred_id in pred_id_list:
            if pred_id in matched_pred:
                continue
            pred_mask = pred_masks[pred_id]
            
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > max_iou:
                    max_iou = iou
                    max_pred_id = pred_id
        
        # Add to totals
        if max_pred_id is not None:
            pred_mask = pred_masks[max_pred_id]
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            
            total_intersection += intersection
            total_union += union
            matched_pred.add(max_pred_id)
        else:
            # No match found, add true mask area to union
            total_union += true_mask.sum()
    
    # Add unmatched predicted instances to union
    for pred_id in pred_id_list:
        if pred_id not in matched_pred:
            total_union += pred_masks[pred_id].sum()
    
    aji = total_intersection / total_union if total_union > 0 else 0.0
    
    return aji


def get_fast_aji_plus(true, pred):
    """
    Aggregated Jaccard Index Plus (AJI+)
    Extension of AJI that handles fragmentation better
    
    Args:
        true: Ground truth instance map (H, W)
        pred: Predicted instance map (H, W)
    
    Returns:
        AJI+ score
    """
    # Convert to numpy if torch tensor
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    true = true.astype(np.int32)
    pred = pred.astype(np.int32)
    
    # Get unique IDs
    true_id_list = np.unique(true)
    pred_id_list = np.unique(pred)
    
    true_id_list = true_id_list[true_id_list != 0]
    pred_id_list = pred_id_list[pred_id_list != 0]
    
    # Quick return for empty cases
    if len(true_id_list) == 0 and len(pred_id_list) == 0:
        return 1.0
    if len(true_id_list) == 0:
        return 0.0
    
    # Compute intersection matrix
    true_masks = {i: (true == i) for i in true_id_list}
    pred_masks = {i: (pred == i) for i in pred_id_list}
    
    # Build intersection matrix
    intersection_matrix = np.zeros((len(true_id_list), len(pred_id_list)))
    
    for i, true_id in enumerate(true_id_list):
        true_mask = true_masks[true_id]
        for j, pred_id in enumerate(pred_id_list):
            pred_mask = pred_masks[pred_id]
            intersection_matrix[i, j] = np.logical_and(true_mask, pred_mask).sum()
    
    # For each true, find all overlapping predictions
    total_intersection = 0
    total_union = 0
    used_pred = set()
    
    for i, true_id in enumerate(true_id_list):
        true_mask = true_masks[true_id]
        
        # Find all predictions that overlap with this true instance
        overlapping_preds = []
        for j, pred_id in enumerate(pred_id_list):
            if intersection_matrix[i, j] > 0:
                overlapping_preds.append(pred_id)
                used_pred.add(pred_id)
        
        if len(overlapping_preds) > 0:
            # Union of all overlapping predictions
            combined_pred = np.zeros_like(true_mask, dtype=bool)
            for pred_id in overlapping_preds:
                combined_pred = np.logical_or(combined_pred, pred_masks[pred_id])
            
            intersection = np.logical_and(true_mask, combined_pred).sum()
            union = np.logical_or(true_mask, combined_pred).sum()
            
            total_intersection += intersection
            total_union += union
        else:
            # No overlapping predictions
            total_union += true_mask.sum()
    
    # Add unmatched predictions
    for pred_id in pred_id_list:
        if pred_id not in used_pred:
            total_union += pred_masks[pred_id].sum()
    
    aji_plus = total_intersection / total_union if total_union > 0 else 0.0
    
    return aji_plus


def aggregated_jaccard_index(true, pred):
    """
    Aggregated Jaccard Index (alias for get_fast_aji)
    
    Args:
        true: Ground truth instance map (H, W)
        pred: Predicted instance map (H, W)
    
    Returns:
        AJI score
    """
    return get_fast_aji(true, pred)


def aggregated_jaccard_index_plus(true, pred):
    """
    Aggregated Jaccard Index Plus (alias for get_fast_aji_plus)
    
    Args:
        true: Ground truth instance map (H, W)
        pred: Predicted instance map (H, W)
    
    Returns:
        AJI+ score
    """
    return get_fast_aji_plus(true, pred)


def batch_panoptic_quality(true_batch, pred_batch, match_iou=0.5):
    """
    Compute PQ for a batch of instance maps
    
    Args:
        true_batch: Batch of ground truth instance maps (B, H, W)
        pred_batch: Batch of predicted instance maps (B, H, W)
        match_iou: IoU threshold for matching
    
    Returns:
        List of (pq, dq, sq) tuples for each sample
    """
    batch_size = true_batch.shape[0]
    results = []
    
    for i in range(batch_size):
        pq, dq, sq = get_fast_pq(true_batch[i], pred_batch[i], match_iou)
        results.append((pq, dq, sq))
    
    return results


def batch_aji(true_batch, pred_batch):
    """
    Compute AJI for a batch of instance maps
    
    Args:
        true_batch: Batch of ground truth instance maps (B, H, W)
        pred_batch: Batch of predicted instance maps (B, H, W)
    
    Returns:
        List of AJI scores for each sample
    """
    batch_size = true_batch.shape[0]
    aji_scores = []
    
    for i in range(batch_size):
        aji = get_fast_aji(true_batch[i], pred_batch[i])
        aji_scores.append(aji)
    
    return aji_scores
"""
Utility functions for metric computation
Includes batch processing and result aggregation helpers
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_all_metrics(pred_seg, pred_instance, gt_seg, gt_instance, prefix=''):
    """
    Compute all metrics (binary segmentation + instance) for one task
    
    Args:
        pred_seg: Predicted binary mask (H, W), values in [0, 1]
        pred_instance: Predicted instance map (H, W), integer IDs
        gt_seg: Ground truth binary mask (H, W), values in [0, 1]
        gt_instance: Ground truth instance map (H, W), integer IDs
        prefix: Prefix for metric names (e.g., 'he_nuclei_')
    
    Returns:
        Dictionary with all computed metrics
    """
    from .segmentation import dice_coefficient, iou_score, precision_score, recall_score
    from .instance import panoptic_quality, aggregated_jaccard_index, aggregated_jaccard_index_plus
    
    metrics = {}
    
    # Binary segmentation metrics
    metrics[f'{prefix}dice'] = dice_coefficient(pred_seg, gt_seg)
    metrics[f'{prefix}iou'] = iou_score(pred_seg, gt_seg)
    metrics[f'{prefix}precision'] = precision_score(pred_seg, gt_seg)
    metrics[f'{prefix}recall'] = recall_score(pred_seg, gt_seg)
    
    # Instance segmentation metrics
    pq, dq, sq = panoptic_quality(gt_instance, pred_instance, match_iou=0.5)
    metrics[f'{prefix}pq'] = pq
    metrics[f'{prefix}dq'] = dq
    metrics[f'{prefix}sq'] = sq
    
    metrics[f'{prefix}aji'] = aggregated_jaccard_index(gt_instance, pred_instance)
    metrics[f'{prefix}aji_plus'] = aggregated_jaccard_index_plus(gt_instance, pred_instance)
    
    return metrics


def compute_batch_metrics(outputs, batch, device='cuda'):
    """
    Compute metrics for all 4 tasks in a batch
    
    Args:
        outputs: Model outputs dictionary with seg and instance predictions
        batch: Batch dictionary with ground truth
        device: Device string
    
    Returns:
        Dictionary with averaged metrics across batch
    """
    from .segmentation import dice_coefficient, iou_score
    from .instance import get_fast_pq, get_fast_aji
    
    batch_size = outputs['he_nuclei_seg'].shape[0]
    
    # Initialize metric accumulators
    metrics = {
        'he_nuclei_dice': 0.0,
        'he_nuclei_iou': 0.0,
        'he_nuclei_pq': 0.0,
        'he_nuclei_aji': 0.0,
        
        'he_cell_dice': 0.0,
        'he_cell_iou': 0.0,
        'he_cell_pq': 0.0,
        'he_cell_aji': 0.0,
        
        'mif_nuclei_dice': 0.0,
        'mif_nuclei_iou': 0.0,
        'mif_nuclei_pq': 0.0,
        'mif_nuclei_aji': 0.0,
        
        'mif_cell_dice': 0.0,
        'mif_cell_iou': 0.0,
        'mif_cell_pq': 0.0,
        'mif_cell_aji': 0.0,
    }
    
    # Compute metrics for each sample in batch
# Compute metrics for each sample in batch
    for i in range(batch_size):
        # HE Nuclei
        metrics['he_nuclei_dice'] += dice_coefficient(
            outputs['he_nuclei_seg'][i].cpu(), 
            batch['he_nuclei_mask'][i].unsqueeze(0).cpu()
        )
        metrics['he_nuclei_iou'] += iou_score(
            outputs['he_nuclei_seg'][i].cpu(), 
            batch['he_nuclei_mask'][i].unsqueeze(0).cpu()
        )
        pq, _, _ = get_fast_pq(
            batch['he_nuclei_instance'][i].cpu().numpy(),
            (outputs['he_nuclei_seg'][i].squeeze() > 0.5).long().cpu().numpy()
        )
        metrics['he_nuclei_pq'] += pq
        metrics['he_nuclei_aji'] += get_fast_aji(
            batch['he_nuclei_instance'][i].cpu().numpy(),
            (outputs['he_nuclei_seg'][i].squeeze() > 0.5).long().cpu().numpy()
        )
        
        # HE Cell
        metrics['he_cell_dice'] += dice_coefficient(
            outputs['he_cell_seg'][i].cpu(), 
            batch['he_cell_mask'][i].unsqueeze(0).cpu()
        )
        metrics['he_cell_iou'] += iou_score(
            outputs['he_cell_seg'][i].cpu(), 
            batch['he_cell_mask'][i].unsqueeze(0).cpu()
        )
        pq, _, _ = get_fast_pq(
            batch['he_cell_instance'][i].cpu().numpy(),
            (outputs['he_cell_seg'][i].squeeze() > 0.5).long().cpu().numpy()
        )
        metrics['he_cell_pq'] += pq
        metrics['he_cell_aji'] += get_fast_aji(
            batch['he_cell_instance'][i].cpu().numpy(),
            (outputs['he_cell_seg'][i].squeeze() > 0.5).long().cpu().numpy()
        )
        
        # MIF Nuclei
        metrics['mif_nuclei_dice'] += dice_coefficient(
            outputs['mif_nuclei_seg'][i].cpu(), 
            batch['mif_nuclei_mask'][i].unsqueeze(0).cpu()
        )
        metrics['mif_nuclei_iou'] += iou_score(
            outputs['mif_nuclei_seg'][i].cpu(), 
            batch['mif_nuclei_mask'][i].unsqueeze(0).cpu()
        )
        pq, _, _ = get_fast_pq(
            batch['mif_nuclei_instance'][i].cpu().numpy(),
            (outputs['mif_nuclei_seg'][i].squeeze() > 0.5).long().cpu().numpy()
        )
        metrics['mif_nuclei_pq'] += pq
        metrics['mif_nuclei_aji'] += get_fast_aji(
            batch['mif_nuclei_instance'][i].cpu().numpy(),
            (outputs['mif_nuclei_seg'][i].squeeze() > 0.5).long().cpu().numpy()
        )
        
        # MIF Cell
        metrics['mif_cell_dice'] += dice_coefficient(
            outputs['mif_cell_seg'][i].cpu(), 
            batch['mif_cell_mask'][i].unsqueeze(0).cpu()
        )
        metrics['mif_cell_iou'] += iou_score(
            outputs['mif_cell_seg'][i].cpu(), 
            batch['mif_cell_mask'][i].unsqueeze(0).cpu()
        )
        pq, _, _ = get_fast_pq(
            batch['mif_cell_instance'][i].cpu().numpy(),
            (outputs['mif_cell_seg'][i].squeeze() > 0.5).long().cpu().numpy()
        )
        metrics['mif_cell_pq'] += pq
        metrics['mif_cell_aji'] += get_fast_aji(
            batch['mif_cell_instance'][i].cpu().numpy(),
            (outputs['mif_cell_seg'][i].squeeze() > 0.5).long().cpu().numpy()
        )
    # Average over batch
    for key in metrics:
        metrics[key] /= batch_size
    
    return metrics


def aggregate_metrics(metric_list: List[Dict]) -> Dict:
    """
    Aggregate metrics from multiple batches
    
    Args:
        metric_list: List of metric dictionaries from each batch
    
    Returns:
        Dictionary with averaged metrics
    """
    if len(metric_list) == 0:
        return {}
    
    # Get all metric keys
    keys = metric_list[0].keys()
    
    # Average each metric
    aggregated = {}
    for key in keys:
        values = [m[key] for m in metric_list]
        aggregated[key] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title to print
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Group by task
    tasks = ['he_nuclei', 'he_cell', 'mif_nuclei', 'mif_cell']
    
    for task in tasks:
        task_metrics = {k: v for k, v in metrics.items() if k.startswith(task)}
        
        if len(task_metrics) > 0:
            print(f"\n{task.upper().replace('_', ' ')}:")
            for key, value in task_metrics.items():
                metric_name = key.replace(f'{task}_', '')
                print(f"  {metric_name:12s}: {value:.4f}")
    
    # Overall average
    if 'avg_dice' in metrics:
        print(f"\n{'OVERALL':}")
        print(f"  {'Avg Dice':12s}: {metrics.get('avg_dice', 0):.4f}")
        print(f"  {'Avg IoU':12s}: {metrics.get('avg_iou', 0):.4f}")
        print(f"  {'Avg PQ':12s}: {metrics.get('avg_pq', 0):.4f}")
    
    print(f"{'='*60}\n")


def compute_overall_metrics(metrics: Dict) -> Dict:
    """
    Compute overall averages across all tasks
    
    Args:
        metrics: Dictionary with task-specific metrics
    
    Returns:
        Dictionary with overall averages added
    """
    metrics = metrics.copy()
    
    # Average Dice across all 4 tasks
    dice_scores = [
        metrics.get('he_nuclei_dice', 0),
        metrics.get('he_cell_dice', 0),
        metrics.get('mif_nuclei_dice', 0),
        metrics.get('mif_cell_dice', 0)
    ]
    metrics['avg_dice'] = np.mean(dice_scores)
    
    # Average IoU
    iou_scores = [
        metrics.get('he_nuclei_iou', 0),
        metrics.get('he_cell_iou', 0),
        metrics.get('mif_nuclei_iou', 0),
        metrics.get('mif_cell_iou', 0)
    ]
    metrics['avg_iou'] = np.mean(iou_scores)
    
    # Average PQ
    pq_scores = [
        metrics.get('he_nuclei_pq', 0),
        metrics.get('he_cell_pq', 0),
        metrics.get('mif_nuclei_pq', 0),
        metrics.get('mif_cell_pq', 0)
    ]
    metrics['avg_pq'] = np.mean(pq_scores)
    
    # Average AJI
    aji_scores = [
        metrics.get('he_nuclei_aji', 0),
        metrics.get('he_cell_aji', 0),
        metrics.get('mif_nuclei_aji', 0),
        metrics.get('mif_cell_aji', 0)
    ]
    metrics['avg_aji'] = np.mean(aji_scores)
    
    return metrics
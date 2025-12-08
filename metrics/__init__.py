"""
Metrics package for cell/nuclei segmentation evaluation
Includes binary segmentation metrics and instance segmentation metrics
"""

from .segmentation import (
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
    f1_score,
    pixel_accuracy,
    batch_dice_coefficient,
    batch_iou_score
)

from .instance import (
    panoptic_quality,
    detection_quality,
    segmentation_quality,
    aggregated_jaccard_index,
    aggregated_jaccard_index_plus,
    get_fast_pq,
    get_fast_aji,
    get_fast_aji_plus,
    batch_panoptic_quality,
    batch_aji
)

from .utils import (
    compute_all_metrics,
    compute_batch_metrics,
    aggregate_metrics,
    print_metrics,
    compute_overall_metrics
)

__all__ = [
    # Binary segmentation metrics
    'dice_coefficient',
    'iou_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'pixel_accuracy',
    'batch_dice_coefficient',
    'batch_iou_score',
    
    # Instance segmentation metrics
    'panoptic_quality',
    'detection_quality',
    'segmentation_quality',
    'aggregated_jaccard_index',
    'aggregated_jaccard_index_plus',
    'get_fast_pq',
    'get_fast_aji',
    'get_fast_aji_plus',
    'batch_panoptic_quality',
    'batch_aji',
    
    # Utility functions
    'compute_all_metrics',
    'compute_batch_metrics',
    'aggregate_metrics',
    'print_metrics',
    'compute_overall_metrics',
]

__version__ = '1.0.0'
import numpy as np
from scipy import ndimage
import torch

def iou_score(pred, target, smooth=1e-5, threshold=0.2):
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_object_based_metrics(true_mask, pred_mask, distance_threshold=12):
    def get_centroids(mask):
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        labeled, num_objects = ndimage.label(mask_np)
        if num_objects == 0:
            return np.array([])
        centroids = ndimage.center_of_mass(mask_np, labeled, range(1, num_objects+1))
        return np.array(centroids)

    true_centroids = get_centroids(true_mask)
    pred_centroids = get_centroids(pred_mask)

    if len(true_centroids) == 0 and len(pred_centroids) == 0:
        return 1.0, 1.0, 1.0  # Perfect score if both are empty
    elif len(true_centroids) == 0:
        return 0.0, 0.0, 0.0  # All false positives
    elif len(pred_centroids) == 0:
        return 0.0, 0.0, 0.0  # All false negatives

    matched = set()
    tp = 0

    for pred_centroid in pred_centroids:
        distances = np.sqrt(((true_centroids - pred_centroid[np.newaxis, :]) ** 2).sum(axis=1))
        if distances.size > 0 and np.min(distances) <= distance_threshold:
            match_idx = np.argmin(distances)
            if match_idx not in matched:
                tp += 1
                matched.add(match_idx)

    fp = len(pred_centroids) - tp
    fn = len(true_centroids) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
"""
Binary segmentation metrics for pixel-level evaluation
Used for evaluating binary masks (nuclei/cell presence)
"""

import torch
import numpy as np


def dice_coefficient(pred, target, smooth=1e-5):
    """
    Dice Similarity Coefficient (DSC) / F1 Score
    
    Args:
        pred: Predicted binary mask (B, H, W) or (B, 1, H, W), values in [0, 1]
        target: Ground truth binary mask (B, H, W) or (B, 1, H, W), values in [0, 1]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score (scalar)
    """
    # Flatten spatial dimensions
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, smooth=1e-5):
    """
    Intersection over Union (IoU) / Jaccard Index
    
    Args:
        pred: Predicted binary mask (B, H, W) or (B, 1, H, W)
        target: Ground truth binary mask (B, H, W) or (B, 1, H, W)
        smooth: Smoothing factor
    
    Returns:
        IoU score (scalar)
    """
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def precision_score(pred, target, smooth=1e-5):
    """
    Precision = TP / (TP + FP)
    
    Args:
        pred: Predicted binary mask (B, H, W) or (B, 1, H, W)
        target: Ground truth binary mask (B, H, W) or (B, 1, H, W)
        smooth: Smoothing factor
    
    Returns:
        Precision score (scalar)
    """
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    
    return precision.item()


def recall_score(pred, target, smooth=1e-5):
    """
    Recall = TP / (TP + FN)
    
    Args:
        pred: Predicted binary mask (B, H, W) or (B, 1, H, W)
        target: Ground truth binary mask (B, H, W) or (B, 1, H, W)
        smooth: Smoothing factor
    
    Returns:
        Recall score (scalar)
    """
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    recall = (true_positive + smooth) / (actual_positive + smooth)
    
    return recall.item()


def f1_score(pred, target, smooth=1e-5):
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    Note: This is equivalent to Dice coefficient
    
    Args:
        pred: Predicted binary mask (B, H, W) or (B, 1, H, W)
        target: Ground truth binary mask (B, H, W) or (B, 1, H, W)
        smooth: Smoothing factor
    
    Returns:
        F1 score (scalar)
    """
    prec = precision_score(pred, target, smooth)
    rec = recall_score(pred, target, smooth)
    
    f1 = (2 * prec * rec) / (prec + rec + smooth)
    
    return f1


def pixel_accuracy(pred, target):
    """
    Pixel Accuracy = Correct Pixels / Total Pixels
    
    Args:
        pred: Predicted binary mask (B, H, W) or (B, 1, H, W), values in [0, 1]
        target: Ground truth binary mask (B, H, W) or (B, 1, H, W), values in [0, 1]
    
    Returns:
        Pixel accuracy (scalar)
    """
    # Binarize predictions (threshold at 0.5)
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    pred_binary = pred_binary.contiguous().view(-1)
    target_binary = target_binary.contiguous().view(-1)
    
    correct = (pred_binary == target_binary).sum().float()
    total = pred_binary.numel()
    
    accuracy = correct / total
    
    return accuracy.item()


def batch_dice_coefficient(pred, target, smooth=1e-5):
    """
    Compute Dice coefficient for each sample in batch separately
    
    Args:
        pred: Predicted masks (B, H, W) or (B, 1, H, W)
        target: Ground truth masks (B, H, W) or (B, 1, H, W)
        smooth: Smoothing factor
    
    Returns:
        List of Dice scores for each sample
    """
    batch_size = pred.shape[0]
    dice_scores = []
    
    for i in range(batch_size):
        dice = dice_coefficient(pred[i:i+1], target[i:i+1], smooth)
        dice_scores.append(dice)
    
    return dice_scores


def batch_iou_score(pred, target, smooth=1e-5):
    """
    Compute IoU for each sample in batch separately
    
    Args:
        pred: Predicted masks (B, H, W) or (B, 1, H, W)
        target: Ground truth masks (B, H, W) or (B, 1, H, W)
        smooth: Smoothing factor
    
    Returns:
        List of IoU scores for each sample
    """
    batch_size = pred.shape[0]
    iou_scores = []
    
    for i in range(batch_size):
        iou = iou_score(pred[i:i+1], target[i:i+1], smooth)
        iou_scores.append(iou)
    
    return iou_scores
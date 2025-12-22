"""
Utility functions for Vitamin-P models

Contains preprocessing, helper functions, and common utilities used across models.
"""

import torch
import torch.nn.functional as F


class SimplePreprocessing:
    """
    Image preprocessing and augmentation utilities
    
    Provides:
    - Percentile normalization
    - Color augmentations (channel shuffle, grayscale, invert)
    - Brightness and contrast jittering
    """
    
    def __init__(self):
        pass
    
    def percentile_normalize(self, img):
        """
        Normalize image using 1st and 99th percentiles
        
        Robust to outliers compared to min-max normalization.
        
        Args:
            img (torch.Tensor): Image tensor (C, H, W) or (B, C, H, W)
        
        Returns:
            torch.Tensor: Normalized image in range [0, 1]
        """
        if img.dim() == 4:
            # Batch of images
            B = img.shape[0]
            normalized = []
            for i in range(B):
                single_img = img[i]
                p1 = torch.quantile(single_img, 0.01)
                p99 = torch.quantile(single_img, 0.99)
                norm_img = (single_img - p1) / (p99 - p1 + 1e-8)
                normalized.append(torch.clamp(norm_img, 0, 1))
            return torch.stack(normalized, dim=0)
        else:
            # Single image
            p1 = torch.quantile(img, 0.01)
            p99 = torch.quantile(img, 0.99)
            img = (img - p1) / (p99 - p1 + 1e-8)
            return torch.clamp(img, 0, 1)
    
    def random_channel_shuffle(self, img, prob=0.3):
        """
        Randomly shuffle RGB channels
        
        Args:
            img (torch.Tensor): Image tensor (C, H, W)
            prob (float): Probability of applying augmentation
        
        Returns:
            torch.Tensor: Augmented image
        """
        if torch.rand(1).item() < prob:
            perm = torch.randperm(3)
            img = img[perm, :, :]
        return img
    
    def random_grayscale(self, img, prob=0.1):
        """
        Randomly convert to grayscale
        
        Args:
            img (torch.Tensor): Image tensor (C, H, W)
            prob (float): Probability of applying augmentation
        
        Returns:
            torch.Tensor: Augmented image
        """
        if torch.rand(1).item() < prob:
            gray = img.mean(dim=0, keepdim=True)
            img = gray.repeat(3, 1, 1)
        return img
    
    def random_invert(self, img, prob=0.25):
        """
        Randomly invert image intensities
        
        Args:
            img (torch.Tensor): Image tensor (C, H, W)
            prob (float): Probability of applying augmentation
        
        Returns:
            torch.Tensor: Augmented image
        """
        if torch.rand(1).item() < prob:
            img = 1 - img
        return img
    
    def brightness_contrast_jitter(self, img, brightness_std=0.2):
        """
        Apply random brightness and contrast jittering
        
        Args:
            img (torch.Tensor): Image tensor (C, H, W)
            brightness_std (float): Standard deviation for brightness jitter
        
        Returns:
            torch.Tensor: Augmented image
        """
        C = img.shape[0]
        
        # Brightness jitter
        brightness = torch.randn(C, 1, 1, device=img.device) * brightness_std
        img = img + brightness
        
        # Contrast jitter
        contrast_factor = torch.rand(C, 1, 1, device=img.device) * 4 - 2
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - mean) * (2 ** contrast_factor) + mean
        
        return torch.clamp(img, 0, 1)
    
    def apply_color_augmentations(self, img):
        """
        Apply all color augmentations in sequence
        
        Args:
            img (torch.Tensor): Image tensor (C, H, W)
        
        Returns:
            torch.Tensor: Augmented image
        """
        img = self.random_channel_shuffle(img, prob=0.3)
        img = self.random_grayscale(img, prob=0.1)
        img = self.random_invert(img, prob=0.25)
        img = self.brightness_contrast_jitter(img, brightness_std=0.2)
        return img


# Input preparation functions
def prepare_he_input(he_img):
    """
    Prepare H&E image for model input (no modification needed)
    
    Args:
        he_img (torch.Tensor): H&E image (B, 3, H, W)
    
    Returns:
        torch.Tensor: Prepared image (B, 3, H, W)
    """
    return he_img


def prepare_mif_input(mif_img):
    """
    Prepare MIF image for model input (add zero channel to make 3-channel)
    
    MIF images are 2-channel, but models expect 3-channel input.
    Add a zero-filled third channel.
    
    Args:
        mif_img (torch.Tensor): MIF image (B, 2, H, W)
    
    Returns:
        torch.Tensor: Prepared image (B, 3, H, W)
    """
    B, C, H, W = mif_img.shape
    zero_channel = torch.zeros(B, 1, H, W, device=mif_img.device, dtype=mif_img.dtype)
    return torch.cat([mif_img, zero_channel], dim=1)


# Metric computation functions
def compute_dice(pred, target):
    """
    Compute Dice coefficient (F1 score for binary segmentation)
    
    Args:
        pred (torch.Tensor): Predicted binary mask (any shape)
        target (torch.Tensor): Ground truth binary mask (same shape as pred)
    
    Returns:
        float: Dice coefficient in range [0, 1]
    """
    pred = pred.view(-1)
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    
    return dice.item() if isinstance(dice, torch.Tensor) else dice


def compute_iou(pred, target):
    """
    Compute Intersection over Union (IoU / Jaccard index)
    
    Args:
        pred (torch.Tensor): Predicted binary mask (any shape)
        target (torch.Tensor): Ground truth binary mask (same shape as pred)
    
    Returns:
        float: IoU score in range [0, 1]
    """
    pred = pred.view(-1)
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    return iou.item() if isinstance(iou, torch.Tensor) else iou


# Image resizing utilities
def resize_to_dinov2(img, target_size=518):
    """
    Resize image to DINOv2 input size (518x518)
    
    Args:
        img (torch.Tensor): Image tensor (B, C, H, W)
        target_size (int): Target size (default: 518)
    
    Returns:
        torch.Tensor: Resized image (B, C, target_size, target_size)
    """
    if img.shape[2] != target_size or img.shape[3] != target_size:
        img = F.interpolate(img, size=(target_size, target_size), 
                          mode='bilinear', align_corners=False)
    return img


def resize_to_output(img, target_size=512):
    """
    Resize to final output size (512x512)
    
    Args:
        img (torch.Tensor): Image tensor (B, C, H, W)
        target_size (int): Target size (default: 512)
    
    Returns:
        torch.Tensor: Resized image (B, C, target_size, target_size)
    """
    if img.shape[2] != target_size or img.shape[3] != target_size:
        img = F.interpolate(img, size=(target_size, target_size), 
                          mode='bilinear', align_corners=False)
    return img


# Model utilities
def count_parameters(model):
    """
    Count total and trainable parameters in a model
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        dict: {'total': int, 'trainable': int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable
    }


def freeze_backbone(model):
    """
    Freeze backbone parameters (for transfer learning)
    
    Args:
        model: Model with 'backbone' or similar attribute
    
    Returns:
        model: Model with frozen backbone
    """
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif hasattr(model, 'he_backbone'):
        for param in model.he_backbone.parameters():
            param.requires_grad = False
    elif hasattr(model, 'mif_backbone'):
        for param in model.mif_backbone.parameters():
            param.requires_grad = False
    
    return model


def unfreeze_backbone(model):
    """
    Unfreeze backbone parameters
    
    Args:
        model: Model with 'backbone' or similar attribute
    
    Returns:
        model: Model with unfrozen backbone
    """
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = True
    elif hasattr(model, 'he_backbone'):
        for param in model.he_backbone.parameters():
            param.requires_grad = True
    elif hasattr(model, 'mif_backbone'):
        for param in model.mif_backbone.parameters():
            param.requires_grad = True
    
    return model
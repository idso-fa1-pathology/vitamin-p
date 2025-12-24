"""
Utility functions for preprocessing and metrics
"""

import torch


class SimplePreprocessing:
    """
    Image preprocessing and augmentation utilities
    """
    def __init__(self):
        pass
    
    def percentile_normalize(self, img):
        """Normalize image to [0, 1] using 1st and 99th percentiles"""
        if img.dim() == 4:
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
            p1 = torch.quantile(img, 0.01)
            p99 = torch.quantile(img, 0.99)
            img = (img - p1) / (p99 - p1 + 1e-8)
            return torch.clamp(img, 0, 1)
    
    def random_channel_shuffle(self, img, prob=0.3):
        """Randomly shuffle color channels"""
        if torch.rand(1).item() < prob:
            perm = torch.randperm(3)
            img = img[perm, :, :]
        return img
    
    def random_grayscale(self, img, prob=0.1):
        """Randomly convert to grayscale"""
        if torch.rand(1).item() < prob:
            gray = img.mean(dim=0, keepdim=True)
            img = gray.repeat(3, 1, 1)
        return img
    
    def random_invert(self, img, prob=0.25):
        """Randomly invert image colors"""
        if torch.rand(1).item() < prob:
            img = 1 - img
        return img
    
    def brightness_contrast_jitter(self, img, brightness_std=0.2):
        """Apply random brightness and contrast jitter"""
        C = img.shape[0]
        brightness = torch.randn(C, 1, 1, device=img.device) * brightness_std
        img = img + brightness
        contrast_factor = torch.rand(C, 1, 1, device=img.device) * 4 - 2
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - mean) * (2 ** contrast_factor) + mean
        return torch.clamp(img, 0, 1)
    
    def apply_color_augmentations(self, img):
        """Apply all color augmentations"""
        img = self.random_channel_shuffle(img, prob=0.3)
        img = self.random_grayscale(img, prob=0.1)
        img = self.random_invert(img, prob=0.25)
        img = self.brightness_contrast_jitter(img, brightness_std=0.2)
        return img


def compute_dice(pred, target):
    """
    Compute Dice coefficient
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
    
    Returns:
        Dice score
    """
    pred = pred.view(-1)
    target = target.view(-1).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)


def prepare_he_input(he_img):
    """
    Prepare H&E input (pass through as-is)
    
    Args:
        he_img: (B, 3, H, W) H&E image
    
    Returns:
        (B, 3, H, W) H&E image
    """
    return he_img


def prepare_mif_input(mif_img):
    """
    Prepare MIF input by adding zero channel (2ch → 3ch)
    
    Args:
        mif_img: (B, 2, H, W) MIF image
    
    Returns:
        (B, 3, H, W) MIF image with zero channel
    """
    B, C, H, W = mif_img.shape
    zero_channel = torch.zeros(B, 1, H, W, device=mif_img.device, dtype=mif_img.dtype)
    return torch.cat([mif_img, zero_channel], dim=1)
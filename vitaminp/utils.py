"""
Utility functions for preprocessing and metrics
"""

import torch

class SimplePreprocessing:
    """
    Image preprocessing and augmentation utilities
    Optimized for stability in medical imaging
    """
    def __init__(self):
        pass
    
    def percentile_normalize(self, img):
        """
        Normalize image to [0, 1] using 1st and 99th percentiles
        Handles batches (4D) or single images (3D)
        """
        # Add epsilon to prevent div-by-zero on black placeholder images
        eps = 1e-8
        
        if img.dim() == 4:
            B = img.shape[0]
            normalized = []
            for i in range(B):
                single_img = img[i]
                # Efficient quantile calculation
                # Note: float32 cast is sometimes needed for quantile on older PyTorch versions
                p1 = torch.quantile(single_img.float(), 0.01)
                p99 = torch.quantile(single_img.float(), 0.99)
                
                norm_img = (single_img - p1) / (p99 - p1 + eps)
                normalized.append(torch.clamp(norm_img, 0, 1))
            return torch.stack(normalized, dim=0)
        else:
            p1 = torch.quantile(img.float(), 0.01)
            p99 = torch.quantile(img.float(), 0.99)
            img = (img - p1) / (p99 - p1 + eps)
            return torch.clamp(img, 0, 1)
    
    def random_channel_shuffle(self, img, prob=0.3):
        """Randomly shuffle color channels"""
        if img.shape[0] < 2: return img  # Skip if single channel
        
        if torch.rand(1).item() < prob:
            perm = torch.randperm(img.shape[0])
            img = img[perm, :, :]
        return img
    
    def random_grayscale(self, img, prob=0.1):
        """Randomly convert to grayscale"""
        if torch.rand(1).item() < prob:
            # Keep dim 0 for broadcasting
            gray = img.mean(dim=0, keepdim=True)
            img = gray.repeat(img.shape[0], 1, 1)
        return img
    
    def random_invert(self, img, prob=0.2):
        """Randomly invert image colors"""
        if torch.rand(1).item() < prob:
            img = 1.0 - img
        return img
    
    def brightness_contrast_jitter(self, img, prob=0.5, brightness_std=0.1, contrast_range=(0.75, 1.25)):
        """
        Apply random brightness and contrast jitter
        Args:
            prob: Probability of applying the transform
            brightness_std: Standard deviation of brightness noise
            contrast_range: (min, max) multiplier for contrast
        """
        if torch.rand(1).item() >= prob:
            return img
            
        C = img.shape[0]
        device = img.device
        
        # 1. Brightness (Additive)
        brightness = torch.randn(C, 1, 1, device=device) * brightness_std
        img = img + brightness
        
        # 2. Contrast (Multiplicative centered on mean)
        # Log-uniform sampling for balanced contrast scaling
        min_c, max_c = contrast_range
        contrast_log = torch.rand(C, 1, 1, device=device) * (torch.log(torch.tensor(max_c)) - torch.log(torch.tensor(min_c))) + torch.log(torch.tensor(min_c))
        contrast_factor = torch.exp(contrast_log)
        
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - mean) * contrast_factor + mean
        
        return torch.clamp(img, 0, 1)
    
    # def apply_color_augmentations(self, img):
    #     """Apply chain of color augmentations"""
    #     # Shuffle channels (channel independent learning)
    #     img = self.random_channel_shuffle(img, prob=0.3)
        
    #     # Grayscale (texture focus)
    #     img = self.random_grayscale(img, prob=0.1)
        
    #     # Invert (stain invariance / cross-modality alignment)
    #     img = self.random_invert(img, prob=0.15)
        
    #     # Jitter (robustness) - FIXED: Now has probability check inside
    #     img = self.brightness_contrast_jitter(img, prob=0.5, brightness_std=0.15)
        
    #     return img

    def apply_color_augmentations(self, img):
            """
            Apply chain of color augmentations.
            Smartly handles H&E vs MIF to maximize 'Shape Learning'.
            """
            # 1. DETECT MODALITY
            # MIF has a zero-filled 3rd channel (index 2)
            is_mif = False
            if img.dim() == 3: 
                if img.shape[0] == 3 and img[2, :, :].max() == 0: is_mif = True
            elif img.dim() == 4:
                if img.shape[1] == 3 and img[0, 2, :, :].max() == 0: is_mif = True

            # 2. APPLY AUGMENTATIONS
            
            # A. Channel Shuffle -> ONLY for MIF
            # Why? MIF markers (DAPI, CK, CD45) are arbitrary channels. 
            # Shuffling prevents the model from memorizing "Tumor is always Green".
            # We DO NOT shuffle H&E because Hematoxylin (Nuclei) and Eosin (Cytoplasm) 
            # have a fixed chemical relationship. Breaking this confuses the encoder.
            if is_mif:
                img = self.random_channel_shuffle(img, prob=0.3)
            
            # B. Grayscale -> APPLY TO BOTH
            # Why? This is the ultimate "Shape Not Color" teacher.
            # It forces the model to segment nuclei based on boundaries, not just "purple pixels".
            img = self.random_grayscale(img, prob=0.2) # Increased prob slightly
            
            # C. Invert -> APPLY TO BOTH (CRITICAL for Agnostic Learning)
            # Why? H&E is Brightfield (Light Background). MIF is Darkfield (Black Background).
            # Inverting H&E makes the background black and nuclei light.
            # This bridges the domain gap, making H&E look exactly like MIF to the model.
            img = self.random_invert(img, prob=0.2) 
            
            # D. Jitter -> APPLY TO BOTH
            # Standard robustness against scanner differences.
            img = self.brightness_contrast_jitter(img, prob=0.5, brightness_std=0.15)
            
            return img

def compute_dice(pred, target):
    """
    Compute Dice coefficient for binary tensors
    
    Args:
        pred: Predicted binary mask (0 or 1)
        target: Ground truth binary mask (0 or 1)
    """
    # Ensure inputs are flat floats
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    
    # Epsilon for numerical stability
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)


def prepare_he_input(he_img):
    """
    Prepare H&E input (pass through as-is)
    Args:
        he_img: (B, 3, H, W) H&E image
    """
    return he_img


def prepare_mif_input(mif_img):
    """
    Prepare MIF input by adding zero channel (2ch -> 3ch)
    Handles both single images (C,H,W) and batches (B,C,H,W)
    
    Args:
        mif_img: (B, 2, H, W) or (2, H, W) MIF image
    
    Returns:
        MIF image with appended zero channel
    """
    if mif_img.dim() == 4:
        B, C, H, W = mif_img.shape
        zero_channel = torch.zeros(B, 1, H, W, device=mif_img.device, dtype=mif_img.dtype)
        return torch.cat([mif_img, zero_channel], dim=1)
    else:
        C, H, W = mif_img.shape
        zero_channel = torch.zeros(1, H, W, device=mif_img.device, dtype=mif_img.dtype)
        return torch.cat([mif_img, zero_channel], dim=0)
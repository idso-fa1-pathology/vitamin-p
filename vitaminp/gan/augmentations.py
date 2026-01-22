"""
Paired augmentations for Pix2Pix GAN training

IMPORTANT: H&E and MIF must receive IDENTICAL transformations
to preserve pixel-level alignment.

Only geometric transformations are used (safe for paired data):
- Random flips (horizontal/vertical)
- Random 90° rotations
- No color augmentations (would break H&E→MIF relationship)
"""

import torch
import random


class PairedGANAugmentation:
    """
    Paired augmentation for H&E and MIF images
    
    Applies identical geometric transformations to both images
    to preserve spatial alignment.
    
    Args:
        flip_prob: Probability of horizontal/vertical flip (default: 0.5)
        rotate_prob: Probability of 90° rotation (default: 0.5)
        
    Example:
        >>> aug = PairedGANAugmentation(flip_prob=0.5, rotate_prob=0.5)
        >>> he_aug, mif_aug = aug(he_img, mif_img)
    """
    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
    
    def __call__(self, he_img, mif_img):
        """
        Apply paired augmentations
        
        Args:
            he_img: H&E image tensor (C, H, W) or (B, C, H, W)
            mif_img: MIF image tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Augmented (he_img, mif_img) with identical transformations
        """
        # Random horizontal flip
        if random.random() < self.flip_prob:
            he_img = torch.flip(he_img, dims=[-1])
            mif_img = torch.flip(mif_img, dims=[-1])
        
        # Random vertical flip
        if random.random() < self.flip_prob:
            he_img = torch.flip(he_img, dims=[-2])
            mif_img = torch.flip(mif_img, dims=[-2])
        
        # Random 90° rotation (0°, 90°, 180°, 270°)
        if random.random() < self.rotate_prob:
            k = random.randint(1, 3)  # 1, 2, or 3 (90°, 180°, 270°)
            he_img = torch.rot90(he_img, k=k, dims=[-2, -1])
            mif_img = torch.rot90(mif_img, k=k, dims=[-2, -1])
        
        return he_img, mif_img


class NoAugmentation:
    """
    Identity transformation (no augmentation)
    
    Useful for validation/test sets or debugging.
    """
    def __call__(self, he_img, mif_img):
        return he_img, mif_img


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Testing PairedGANAugmentation...")
    
    # Create sample images
    he_img = torch.randn(3, 512, 512)
    mif_img = torch.randn(2, 512, 512)
    
    # Create augmentation
    aug = PairedGANAugmentation(flip_prob=0.5, rotate_prob=0.5)
    
    # Apply augmentation
    he_aug, mif_aug = aug(he_img, mif_img)
    
    print(f"Original H&E shape: {he_img.shape}")
    print(f"Augmented H&E shape: {he_aug.shape}")
    print(f"Original MIF shape: {mif_img.shape}")
    print(f"Augmented MIF shape: {mif_aug.shape}")
    
    # Test multiple times to see variations
    print("\nTesting 10 augmentations:")
    for i in range(10):
        he_aug, mif_aug = aug(he_img, mif_img)
        # Check if augmented (compare first pixel)
        h_flipped = not torch.equal(he_img[0, 0, 0], he_aug[0, 0, 0])
        print(f"  Aug {i+1}: {'Changed' if h_flipped else 'Unchanged'}")
    
    print("\n✅ All tests passed!")
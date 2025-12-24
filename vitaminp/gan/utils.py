"""
Utility functions for GAN training
"""

import torch


class GANPreprocessing:
    """
    Preprocessing utilities for GAN training
    
    Example:
        >>> preprocessor = GANPreprocessing()
        >>> he_img = torch.randn(2, 3, 512, 512)
        >>> 
        >>> # Normalize to [0, 1]
        >>> normalized = preprocessor.percentile_normalize(he_img)
        >>> 
        >>> # Scale to [-1, 1] for GAN
        >>> gan_input = preprocessor.to_gan_range(normalized)
        >>> 
        >>> # Convert back to [0, 1]
        >>> output = preprocessor.from_gan_range(gan_input)
    """
    
    def percentile_normalize(self, img):
        """
        Percentile normalization (1st to 99th percentile)
        
        Args:
            img: Tensor of shape (B, C, H, W) or (C, H, W)
        
        Returns:
            Normalized tensor in range [0, 1]
        """
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
    
    def to_gan_range(self, img):
        """
        Convert from [0, 1] to [-1, 1] for GAN input
        
        Args:
            img: Tensor in range [0, 1]
        
        Returns:
            Tensor in range [-1, 1]
        """
        return img * 2 - 1
    
    def from_gan_range(self, img):
        """
        Convert from [-1, 1] to [0, 1]
        
        Args:
            img: Tensor in range [-1, 1]
        
        Returns:
            Tensor in range [0, 1]
        """
        return (img + 1) / 2
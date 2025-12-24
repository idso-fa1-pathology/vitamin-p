"""
Loss functions for Pix2Pix GAN training
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    GAN loss for generator and discriminator training
    
    Uses BCEWithLogitsLoss for stable training.
    
    Example:
        >>> criterion = GANLoss()
        >>> pred = torch.randn(2, 1, 30, 30)  # Discriminator output
        >>> 
        >>> # Generator loss (wants discriminator to predict real)
        >>> loss_g = criterion(pred, target_is_real=True)
        >>> 
        >>> # Discriminator loss for real images
        >>> loss_d_real = criterion(pred, target_is_real=True)
        >>> 
        >>> # Discriminator loss for fake images
        >>> loss_d_fake = criterion(pred, target_is_real=False)
    """
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, prediction, target_is_real):
        """
        Calculate GAN loss
        
        Args:
            prediction: Discriminator output (logits)
            target_is_real: If True, target is real (1), else fake (0)
        
        Returns:
            Loss value
        """
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)
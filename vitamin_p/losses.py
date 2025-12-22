"""
Loss functions for Vitamin-P models

Includes:
- DiceFocalLoss: Combined Dice + Focal loss for segmentation
- HVLoss: Combined MSE + MSGE loss for HoVer-Net style horizontal-vertical maps
- MSGELossMaps: Mean Squared Gradient Error for HV map gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    """
    Combined Dice Loss + Focal Loss for binary segmentation
    
    Dice Loss: Measures overlap between prediction and ground truth
    Focal Loss: Focuses on hard examples by down-weighting easy examples
    
    Args:
        alpha (float): Weighting factor for focal loss (default: 1)
        gamma (float): Focusing parameter for focal loss (default: 2)
        smooth (float): Smoothing factor to avoid division by zero (default: 1)
    
    Returns:
        torch.Tensor: Combined loss value (scalar)
    """
    
    def __init__(self, alpha=1, gamma=2, smooth=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted probabilities (B, 1, H, W) in range [0, 1]
            target (torch.Tensor): Ground truth binary mask (B, 1, H, W) in {0, 1}
        
        Returns:
            torch.Tensor: Combined Dice + Focal loss
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Dice Loss
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Focal Loss
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        return dice_loss + focal_loss


class MSGELossMaps(nn.Module):
    """
    Mean Squared Gradient Error for HV (Horizontal-Vertical) maps
    
    Computes the gradient error between predicted and target HV maps
    using Sobel-like kernels. This helps maintain smooth gradients
    for better cell boundary separation.
    
    Used as part of HVLoss in HoVer-Net architecture.
    """
    
    def __init__(self):
        super().__init__()
    
    def get_sobel_kernel(self, size: int, device: str):
        """
        Generate Sobel-like gradient kernels
        
        Args:
            size (int): Kernel size (must be odd)
            device (str): Device to place kernels on
        
        Returns:
            tuple: (kernel_h, kernel_v) for horizontal and vertical gradients
        """
        assert size % 2 == 1, "Kernel size must be odd"
        
        h_range = torch.arange(
            -size // 2 + 1, size // 2 + 1, 
            dtype=torch.float32, device=device, requires_grad=False
        )
        v_range = torch.arange(
            -size // 2 + 1, size // 2 + 1, 
            dtype=torch.float32, device=device, requires_grad=False
        )
        
        h, v = torch.meshgrid(h_range, v_range, indexing="ij")
        
        # Sobel-like gradient kernels
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        
        return kernel_h, kernel_v
    
    def get_gradient_hv(self, hv: torch.Tensor, device: str):
        """
        Compute gradients of HV maps
        
        Args:
            hv (torch.Tensor): HV map of shape (B, H, W, 2)
            device (str): Device
        
        Returns:
            torch.Tensor: Gradient maps of shape (B, H, W, 2)
        """
        kernel_h, kernel_v = self.get_sobel_kernel(5, device=device)
        kernel_h = kernel_h.view(1, 1, 5, 5)
        kernel_v = kernel_v.view(1, 1, 5, 5)
        
        # Extract horizontal and vertical channels
        h_ch = hv[..., 0].unsqueeze(1)  # (B, 1, H, W)
        v_ch = hv[..., 1].unsqueeze(1)  # (B, 1, H, W)
        
        # Compute gradients
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        
        # Combine gradients
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()
        
        return dhv
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, 
                focus: torch.Tensor, device: str):
        """
        Compute MSGE loss
        
        Args:
            input (torch.Tensor): Predicted HV map (B, 2, H, W)
            target (torch.Tensor): Ground truth HV map (B, 2, H, W)
            focus (torch.Tensor): Focus mask (B, 1, H, W) - where to compute loss
            device (str): Device
        
        Returns:
            torch.Tensor: MSGE loss value
        """
        # Convert to (B, H, W, C) format
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        focus = focus.permute(0, 2, 3, 1)
        
        # Prepare focus mask for both H and V channels
        focus = focus[..., 0]
        focus = (focus[..., None]).float()
        focus = torch.cat([focus, focus], axis=-1).to(device)
        
        # Compute gradients
        true_grad = self.get_gradient_hv(target, device)
        pred_grad = self.get_gradient_hv(input, device)
        
        # Compute squared error weighted by focus mask
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        
        return loss


class HVLoss(nn.Module):
    """
    Combined MSE + MSGE loss for HV (Horizontal-Vertical) maps
    
    HoVer-Net style loss that combines:
    1. MSE: Direct pixel-wise distance prediction error
    2. MSGE: Gradient error for better cell boundary separation
    
    Both losses are equally weighted as in the original HoVer-Net paper.
    
    Args:
        None
    
    Returns:
        torch.Tensor: Combined HV loss (MSE + MSGE)
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.msge = MSGELossMaps()
    
    def forward(self, pred, target, focus, device):
        """
        Compute HV loss
        
        Args:
            pred (torch.Tensor): Predicted HV map (B, 2, H, W)
            target (torch.Tensor): Ground truth HV map (B, 2, H, W)
            focus (torch.Tensor): Focus mask (B, 1, H, W) - nuclei/cell mask
            device (str): Device for computation
        
        Returns:
            torch.Tensor: Combined MSE + MSGE loss
        """
        # MSE: Direct distance prediction error
        mse_loss = self.mse(pred, target)
        
        # MSGE: Gradient error (for cell separation)
        msge_loss = self.msge(pred, target, focus, device)
        
        # Combine with equal weight (as in HoVer-Net)
        return mse_loss + msge_loss


# Convenience function for getting loss functions
def get_segmentation_loss(alpha=1, gamma=2, smooth=1):
    """
    Get segmentation loss (Dice + Focal)
    
    Args:
        alpha (float): Focal loss weight
        gamma (float): Focal loss focusing parameter
        smooth (float): Dice loss smoothing
    
    Returns:
        DiceFocalLoss: Configured loss function
    """
    return DiceFocalLoss(alpha=alpha, gamma=gamma, smooth=smooth)


def get_hv_loss():
    """
    Get HV map loss (MSE + MSGE)
    
    Returns:
        HVLoss: HV loss function
    """
    return HVLoss()
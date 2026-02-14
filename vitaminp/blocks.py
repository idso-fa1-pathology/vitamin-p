"""
Building blocks for Vitamin-P models

Contains reusable convolutional blocks and decoder components used across all models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer - NEW ADDITION
    
    Channel attention mechanism that recalibrates channel-wise features.
    Improves feature representation with minimal parameter overhead.
    
    Args:
        channel (int): Number of input channels
        reduction (int): Reduction ratio for bottleneck (default: 16)
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)


class ConvBlock(nn.Module):
    """
    Double Convolutional Block with GroupNorm, GELU, and SE attention
    
    Standard U-Net style block: Conv -> GN -> GELU -> Conv -> GN -> GELU -> SE
    Optional dropout for regularization.
    
    SOTA Updates:
    - GroupNorm instead of BatchNorm for stability with small batches
    - GELU activation (matches DINOv2 internal activations)
    - Squeeze-and-Excitation for channel attention
    
    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
        dropout_rate (float): Dropout probability (default: 0.0, no dropout)
    
    Returns:
        torch.Tensor: Output feature map after two convolutions + SE attention
    """
    
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),  # Changed from ReLU
        ]
        
        # Add dropout if specified
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU()  # Changed from ReLU
        ])
        
        self.conv = nn.Sequential(*layers)
        self.se = SELayer(out_ch)  # NEW: Channel attention
    
    def forward(self, x):
        x = self.conv(x)
        return self.se(x)  # Apply SE attention


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    
    Captures multi-scale context by probing features at different dilation rates.
    Crucial for SOTA segmentation to understand objects of different sizes.
    
    Updated: GELU activation to match DINOv2
    """
    def __init__(self, in_ch, out_ch, rates=[6, 12, 18]):
        super().__init__()
        self.modules_list = nn.ModuleList()
        
        # 1x1 conv
        self.modules_list.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch), 
            nn.GELU()))  # Changed from ReLU
        
        # 3x3 convs with dilation
        for rate in rates:
            self.modules_list.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(8, out_ch), 
                nn.GELU()))  # Changed from ReLU
        
        # Global Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch), 
            nn.GELU())  # Changed from ReLU

        # Project concatenated results
        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * out_ch + 2 * out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch), 
            nn.GELU(),  # Changed from ReLU
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.modules_list:
            res.append(conv(x))
        
        pool = self.global_pool(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(pool)
        
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with skip connections and residual pathway
    
    NEW: Added residual connection for improved gradient flow
    """
    def __init__(self, in_ch, skip_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout_rate)
        
        # Residual projection: 1x1 conv if dimensions don't match
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, skip):
        # Save input for residual
        identity = self.residual(x)
        
        # Standard decoder operations
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        # Add residual connection
        identity_upsampled = F.interpolate(identity, size=x.shape[2:], mode='bilinear', align_corners=True)
        return x + identity_upsampled


class SegmentationHead(nn.Module):
    """
    Segmentation + HV Map prediction head with CoordConv
    
    Updated for SOTA: 
    1. Adds CoordConv (concatenates x,y coordinates to input)
    2. Uses GroupNorm
    3. GELU activation
    """
    
    def __init__(self, in_ch, hidden_ch=32):
        super().__init__()
        
        # Input channels + 2 for the (x, y) coordinate grids
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch + 2, hidden_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_ch),
            nn.GELU(),  # Changed from ReLU
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.GELU(),  # Changed from ReLU
            nn.Conv2d(hidden_ch, 3, 1)  # 1 for seg + 2 for HV
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Decoder features (B, in_ch, H, W)
        """
        # --- FIXED COORDCONV LOGIC ---
        B, _, H, W = x.shape
        
        # Create X-coordinate grid (repeat H times along height dim)
        xx_channel = torch.arange(W, device=x.device).view(1, 1, 1, W).repeat(B, 1, H, 1).float() / (W - 1)
        
        # Create Y-coordinate grid (repeat W times along width dim)
        yy_channel = torch.arange(H, device=x.device).view(1, 1, H, 1).repeat(B, 1, 1, W).float() / (H - 1)
        
        # Normalize to [-1, 1]
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        
        # Concatenate coordinates to input
        x = torch.cat([x, xx_channel, yy_channel], dim=1)
        # -------------------------------------

        x = self.refine(x)
        x = self.head(x)
        
        # Split into segmentation and HV map
        seg_out = torch.sigmoid(x[:, 0:1])  # Segmentation: [0, 1]
        hv_out = torch.tanh(x[:, 1:3])      # HV map: [-1, 1]
        
        return seg_out, hv_out


class AuxiliaryHead(nn.Module):
    """
    Auxiliary head for deep supervision - NEW ADDITION
    
    Simple 1x1 convolution for auxiliary segmentation outputs
    during training to improve gradient flow to earlier layers.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_ch // 2),
            nn.GELU(),
            nn.Conv2d(in_ch // 2, 1, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.head(x))


class ProjectionLayer(nn.Module):
    """
    1x1 Convolution for channel dimension projection
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.proj(x)


class FusionLayer(nn.Module):
    """
    Feature fusion layer for multi-modal inputs
    
    Updated: GELU activation
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU()  # Changed from ReLU
        )
    
    def forward(self, *features):
        x = torch.cat(features, dim=1)
        return self.fusion(x)


# Utility function to get decoder dimensions based on model size
def get_decoder_dims(model_size):
    decoder_configs = {
        'small': [384, 256, 128, 64],
        'base':  [768, 384, 192, 96],
        'large': [1024, 512, 256, 128],
        'giant': [1536, 768, 384, 192]
    }
    
    if model_size not in decoder_configs:
        raise ValueError(f"model_size must be one of {list(decoder_configs.keys())}")
    
    return decoder_configs[model_size]
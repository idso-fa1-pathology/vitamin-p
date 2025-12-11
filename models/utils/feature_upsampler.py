"""
Feature Upsampler for Vision Transformers
Handles upsampling ViT features to match decoder expectations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureUpsampler(nn.Module):
    """
    Upsample features from ViT to match CNN decoder expectations
    
    ViT features are typically at lower spatial resolution than CNN features.
    This module handles the upsampling and optional channel adjustment.
    """
    
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='bilinear'):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            scale_factor: Upsampling factor
            mode: Upsampling mode ('bilinear', 'nearest', 'bicubic')
        """
        super().__init__()
        
        self.scale_factor = scale_factor
        self.mode = mode
        
        # Channel adjustment if needed
        if in_channels != out_channels:
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.channel_adjust = nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            x: (B, C_out, H*scale, W*scale)
        """
        # Adjust channels first
        x = self.channel_adjust(x)
        
        # Upsample spatially
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=False if self.mode == 'bilinear' else None
            )
        
        return x


class AdaptiveFeatureUpsampler(nn.Module):
    """
    Adaptive upsampler that can handle varying input sizes
    More sophisticated than simple interpolation
    """
    
    def __init__(self, in_channels, out_channels, target_size=None):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels  
            target_size: (H, W) target spatial size, or None for dynamic
        """
        super().__init__()
        
        self.target_size = target_size
        
        # Learnable upsampling with conv
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, target_size=None):
        """
        Args:
            x: (B, C_in, H, W)
            target_size: Optional (H, W) to override default
        Returns:
            x: (B, C_out, H_target, W_target)
        """
        # Determine target size
        size = target_size if target_size is not None else self.target_size
        
        # Upsample first
        if size is not None and (x.shape[2] != size[0] or x.shape[3] != size[1]):
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        # Apply conv
        x = self.upsample_conv(x)
        
        return x


class MultiScaleFeatureFusion(nn.Module):
    """
    Fuse features from multiple scales
    Useful when combining ViT and CNN features
    """
    
    def __init__(self, channels_list, out_channels):
        """
        Args:
            channels_list: List of input channel dimensions
            out_channels: Output channel dimension
        """
        super().__init__()
        
        # Create 1x1 conv for each input scale
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False)
            for ch in channels_list
        ])
        
        self.fusion_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list, target_size):
        """
        Args:
            features_list: List of features at different scales
            target_size: (H, W) target size for all features
        Returns:
            fused: (B, out_channels, H, W) fused features
        """
        # Resize all features to target size and adjust channels
        resized = []
        for feat, conv in zip(features_list, self.scale_convs):
            # Adjust channels
            feat = conv(feat)
            
            # Resize to target
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            
            resized.append(feat)
        
        # Sum all features
        fused = torch.stack(resized, dim=0).sum(dim=0)
        
        # Apply fusion conv
        fused = self.fusion_conv(fused)
        
        return fused


def resize_feature_map(feature, target_size, mode='bilinear'):
    """
    Utility function to resize a feature map
    
    Args:
        feature: (B, C, H, W) feature tensor
        target_size: (H_target, W_target) or int (for square)
        mode: interpolation mode
        
    Returns:
        resized: (B, C, H_target, W_target)
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    if feature.shape[2:] == target_size:
        return feature
    
    return F.interpolate(
        feature,
        size=target_size,
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )


def match_feature_size(feature, reference):
    """
    Resize feature to match reference spatial size
    
    Args:
        feature: (B, C1, H1, W1) feature to resize
        reference: (B, C2, H2, W2) reference feature
        
    Returns:
        resized: (B, C1, H2, W2)
    """
    if feature.shape[2:] == reference.shape[2:]:
        return feature
    
    return resize_feature_map(feature, reference.shape[2:])
"""
Building blocks for Vitamin-P models

Contains reusable convolutional blocks and decoder components used across all models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double Convolutional Block with BatchNorm and ReLU
    
    Standard U-Net style block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Optional dropout for regularization.
    
    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
        dropout_rate (float): Dropout probability (default: 0.0, no dropout)
    
    Returns:
        torch.Tensor: Output feature map after two convolutions
    """
    
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        
        # Add dropout if specified
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with skip connections
    
    Upsamples feature map and concatenates with skip connection from encoder,
    then applies ConvBlock.
    
    Args:
        in_ch (int): Number of input channels (from previous decoder layer)
        skip_ch (int): Number of skip connection channels (from encoder)
        out_ch (int): Number of output channels
        dropout_rate (float): Dropout probability (default: 0.0)
    
    Returns:
        torch.Tensor: Decoded feature map
    """
    
    def __init__(self, in_ch, skip_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout_rate)
    
    def forward(self, x, skip):
        """
        Args:
            x (torch.Tensor): Input from previous decoder layer
            skip (torch.Tensor): Skip connection from encoder
        
        Returns:
            torch.Tensor: Decoded features
        """
        # Upsample to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv(x)
        
        return x


class SegmentationHead(nn.Module):
    """
    Segmentation + HV Map prediction head
    
    Takes decoder features and produces:
    1. Segmentation mask (1 channel, sigmoid activation)
    2. HV map (2 channels, tanh activation)
    
    Args:
        in_ch (int): Number of input channels
        hidden_ch (int): Number of hidden channels (default: 32)
    
    Returns:
        tuple: (seg_mask, hv_map)
            - seg_mask: (B, 1, H, W) in range [0, 1]
            - hv_map: (B, 2, H, W) in range [-1, 1]
    """
    
    def __init__(self, in_ch, hidden_ch=32):
        super().__init__()
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 3, 1)  # 1 for seg + 2 for HV
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Decoder features (B, in_ch, H, W)
        
        Returns:
            tuple: (seg_mask, hv_map)
        """
        x = self.refine(x)
        x = self.head(x)
        
        # Split into segmentation and HV map
        seg_out = torch.sigmoid(x[:, 0:1])  # Segmentation: [0, 1]
        hv_out = torch.tanh(x[:, 1:3])      # HV map: [-1, 1]
        
        return seg_out, hv_out


class ProjectionLayer(nn.Module):
    """
    1x1 Convolution for channel dimension projection
    
    Used to project encoder features to decoder channel dimensions
    for skip connections.
    
    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    
    Returns:
        torch.Tensor: Projected features
    """
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.proj(x)


class FusionLayer(nn.Module):
    """
    Feature fusion layer for multi-modal inputs
    
    Concatenates features from multiple modalities and fuses them
    with 1x1 convolution followed by BatchNorm and ReLU.
    
    Used in dual-encoder architectures to combine H&E and MIF features.
    
    Args:
        in_ch (int): Total input channels (sum of all modalities)
        out_ch (int): Number of output channels
    
    Returns:
        torch.Tensor: Fused features
    """
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, *features):
        """
        Args:
            *features: Variable number of feature tensors to fuse
        
        Returns:
            torch.Tensor: Fused features
        """
        x = torch.cat(features, dim=1)
        return self.fusion(x)


# Utility function to get decoder dimensions based on model size
def get_decoder_dims(model_size):
    """
    Get decoder channel dimensions for a given model size
    
    Args:
        model_size (str): 'small', 'base', 'large', or 'giant'
    
    Returns:
        list: Decoder channel dimensions [d1, d2, d3, d4]
    """
    decoder_configs = {
        'small': [384, 256, 128, 64],
        'base':  [768, 384, 192, 96],
        'large': [1024, 512, 256, 128],
        'giant': [1536, 768, 384, 192]
    }
    
    if model_size not in decoder_configs:
        raise ValueError(f"model_size must be one of {list(decoder_configs.keys())}")
    
    return decoder_configs[model_size]
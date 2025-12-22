"""
DINOv2 Backbone for Vitamin-P models

Provides a flexible Vision Transformer backbone that can handle:
- 3-channel H&E images
- 2-channel MIF images
- Any custom number of input channels

Extracts hierarchical features at 4 different depths for U-Net decoder skip connections.
"""

import torch
import torch.nn as nn
import timm


class DINOv2Backbone(nn.Module):
    """
    DINOv2 Vision Transformer Backbone
    
    Features:
    - Supports variable input channels (default: 3 for RGB/H&E, 2 for MIF)
    - Extracts features at 4 hierarchical levels
    - Pre-trained weights from DINOv2
    - Dynamic image size support
    
    Args:
        model_size (str): Model variant - 'small', 'base', 'large', or 'giant'
        in_channels (int): Number of input channels (default: 3)
    
    Returns:
        dict: Feature maps at 4 different scales
            Keys: 'layer_5', 'layer_11', 'layer_17', 'layer_23' (for base model)
            Values: Feature tensors of shape (B, embed_dim, H, W)
    """
    
    def __init__(self, model_size='base', in_channels=3):
        super().__init__()
        
        # Model configurations for different DINOv2 variants
        model_configs = {
            'small': {'name': 'vit_small_patch14_dinov2.lvd142m', 'embed_dim': 384},
            'base':  {'name': 'vit_base_patch14_dinov2.lvd142m',  'embed_dim': 768},
            'large': {'name': 'vit_large_patch14_dinov2.lvd142m', 'embed_dim': 1024},
            'giant': {'name': 'vit_giant_patch14_dinov2.lvd142m', 'embed_dim': 1536}
        }
        
        if model_size not in model_configs:
            raise ValueError(f"model_size must be one of {list(model_configs.keys())}")
        
        config = model_configs[model_size]
        self.model_size = model_size
        self.embed_dim = config['embed_dim']
        self.patch_size = 14
        self.in_channels = in_channels
        
        # Load pre-trained DINOv2 model
        self.dinov2 = timm.create_model(
            config['name'],
            pretrained=True,
            dynamic_img_size=True
        )
        
        # Adapt input layer if not using 3 channels
        if in_channels != 3:
            self._adapt_input_layer(in_channels)
        
        # Determine feature extraction layers (4 levels like ResNet)
        self.num_layers = len(self.dinov2.blocks)
        self.feature_layers = [
            self.num_layers // 4,      # Early features (~25% depth)
            self.num_layers // 2,      # Mid features (~50% depth)
            3 * self.num_layers // 4,  # Late features (~75% depth)
            self.num_layers - 1        # Deep features (100% depth)
        ]
    
    def _adapt_input_layer(self, in_channels):
        """
        Adapt the first convolutional layer for custom input channels.
        
        For in_channels < 3: Use first N channels from pretrained weights
        For in_channels > 3: Copy RGB weights + initialize extra channels with average
        """
        original_conv = self.dinov2.patch_embed.proj
        
        # Create new conv layer with desired input channels
        self.dinov2.patch_embed.proj = nn.Conv2d(
            in_channels, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights intelligently from pretrained model
        with torch.no_grad():
            if in_channels < 3:
                # Use subset of pretrained weights
                self.dinov2.patch_embed.proj.weight[:, :in_channels] = \
                    original_conv.weight[:, :in_channels]
            else:
                # Copy RGB weights
                self.dinov2.patch_embed.proj.weight[:, :3] = original_conv.weight
                
                # Initialize extra channels with average of RGB
                avg_weight = original_conv.weight.mean(dim=1, keepdim=True)
                self.dinov2.patch_embed.proj.weight[:, 3:] = \
                    avg_weight.repeat(1, in_channels - 3, 1, 1)
    
    def forward(self, x):
        """
        Forward pass through DINOv2 backbone
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
        
        Returns:
            dict: Feature maps at 4 scales
                {
                    'layer_N': tensor of shape (B, embed_dim, h, w)
                    where h, w depend on input size and patch size
                }
        """
        B = x.shape[0]
        
        # 1. Patch Embedding
        x = self.dinov2.patch_embed(x)
        
        # 2. Add Position Embedding (includes CLS token)
        x = self.dinov2._pos_embed(x)
        
        features = {}
        
        # 3. Iterate through transformer blocks and extract features
        for i, blk in enumerate(self.dinov2.blocks):
            x = blk(x)
            
            if i in self.feature_layers:
                # Remove CLS token (first token)
                patch_tokens = x[:, 1:]
                
                # Reshape to spatial feature map
                # Assumes square feature maps
                h = w = int(patch_tokens.shape[1] ** 0.5)
                feat = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)
                
                features[f'layer_{i}'] = feat
        
        return features
    
    def get_feature_dims(self):
        """
        Get the embedding dimension for this backbone variant
        
        Returns:
            int: Feature dimension (384/768/1024/1536 for small/base/large/giant)
        """
        return self.embed_dim


# Convenience functions for creating backbones
def dinov2_small(in_channels=3):
    """Create DINOv2-Small backbone"""
    return DINOv2Backbone(model_size='small', in_channels=in_channels)


def dinov2_base(in_channels=3):
    """Create DINOv2-Base backbone"""
    return DINOv2Backbone(model_size='base', in_channels=in_channels)


def dinov2_large(in_channels=3):
    """Create DINOv2-Large backbone"""
    return DINOv2Backbone(model_size='large', in_channels=in_channels)


def dinov2_giant(in_channels=3):
    """Create DINOv2-Giant backbone"""
    return DINOv2Backbone(model_size='giant', in_channels=in_channels)
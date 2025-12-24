"""
DINOv2 Backbone for feature extraction
"""

import torch
import torch.nn as nn
import timm


class DINOv2Backbone(nn.Module):
    """
    DINOv2 Vision Transformer backbone for hierarchical feature extraction
    
    Args:
        model_size: One of 'small', 'base', 'large', 'giant'
        in_channels: Number of input channels (3 for H&E, 2 for MIF)
    """
    def __init__(self, model_size='base', in_channels=3):
        super().__init__()
        
        model_configs = {
            'small': {'name': 'vit_small_patch14_dinov2.lvd142m', 'embed_dim': 384},
            'base':  {'name': 'vit_base_patch14_dinov2.lvd142m',  'embed_dim': 768},
            'large': {'name': 'vit_large_patch14_dinov2.lvd142m', 'embed_dim': 1024},
            'giant': {'name': 'vit_giant_patch14_dinov2.lvd142m', 'embed_dim': 1536}
        }
        
        config = model_configs[model_size]
        self.embed_dim = config['embed_dim']
        self.patch_size = 14
        self.in_channels = in_channels
        
        self.dinov2 = timm.create_model(
            config['name'],
            pretrained=True,
            dynamic_img_size=True
        )
        
        # Replace first conv if not 3 channels
        if in_channels != 3:
            original_conv = self.dinov2.patch_embed.proj
            self.dinov2.patch_embed.proj = nn.Conv2d(
                in_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # Initialize new conv with pretrained weights (average across channels)
            with torch.no_grad():
                if in_channels < 3:
                    self.dinov2.patch_embed.proj.weight[:, :in_channels] = \
                        original_conv.weight[:, :in_channels]
                else:
                    self.dinov2.patch_embed.proj.weight[:, :3] = original_conv.weight
                    # Initialize extra channels with average of RGB
                    avg_weight = original_conv.weight.mean(dim=1, keepdim=True)
                    self.dinov2.patch_embed.proj.weight[:, 3:] = \
                        avg_weight.repeat(1, in_channels - 3, 1, 1)
        
        self.num_layers = len(self.dinov2.blocks)
        
        # Extract features at 4 different depths (like ResNet's 4 stages)
        self.feature_layers = [
            self.num_layers // 4,      # ~Layer 6  (early)
            self.num_layers // 2,      # ~Layer 12 (mid)
            3 * self.num_layers // 4,  # ~Layer 18 (late)
            self.num_layers - 1        # ~Layer 23 (deep)
        ]
    
    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patch Embedding
        x = self.dinov2.patch_embed(x)
        
        # 2. Add Position Embedding (includes CLS token)
        x = self.dinov2._pos_embed(x)
        
        features = {}
        
        # 3. Iterate blocks and extract features
        for i, blk in enumerate(self.dinov2.blocks):
            x = blk(x)
            
            if i in self.feature_layers:
                # Remove CLS token (index 0)
                patch_tokens = x[:, 1:]
                
                # Reshape to spatial feature map
                h = w = int(patch_tokens.shape[1] ** 0.5)
                feat = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)
                features[f'layer_{i}'] = feat
        
        return features
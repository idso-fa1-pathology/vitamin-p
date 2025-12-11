"""
Vision Transformer (ViT) Encoder Wrapper
Handles feature extraction from ViT models (DINOv2, SAM, etc.)
Converts patch tokens to spatial feature maps for UNet decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTBackbone(nn.Module):
    """
    Wrapper for Vision Transformer backbones to extract multi-scale features
    
    ViTs output patch tokens, but UNet decoders need spatial feature maps.
    This class handles the conversion and creates multi-scale features.
    """
    
    def __init__(self, vit_model, in_channels, pretrained, backbone_name):
        super().__init__()
        
        self.vit = vit_model
        self.in_channels = in_channels
        self.backbone_name = backbone_name
        
        # Get ViT configuration
        self.embed_dim = vit_model.embed_dim
        self.patch_size = vit_model.patch_embed.patch_size[0]  # Assuming square patches
        self.num_features = vit_model.num_features
        
        # Modify patch embedding if input channels != 3
        if in_channels != 3:
            original_patch_embed = vit_model.patch_embed.proj
            vit_model.patch_embed.proj = nn.Conv2d(
                in_channels,
                original_patch_embed.out_channels,
                kernel_size=original_patch_embed.kernel_size,
                stride=original_patch_embed.stride,
                padding=original_patch_embed.padding
            )
            
            if pretrained:
                # Initialize with averaged pretrained weights
                with torch.no_grad():
                    weight = original_patch_embed.weight.mean(dim=1, keepdim=True)
                    vit_model.patch_embed.proj.weight[:, :, :, :] = weight.repeat(1, in_channels, 1, 1)
        
        # Feature projection layers to create multi-scale features
        self._build_projection_layers()
    
    def _build_projection_layers(self):
        """Build projection layers to convert ViT features to multi-scale CNN-like features"""
        # Get depth of ViT
        if hasattr(self.vit, 'blocks'):
            num_blocks = len(self.vit.blocks)
        else:
            num_blocks = 12  # Default
        
        # Select which blocks to extract features from
        self.feature_blocks = [
            num_blocks // 6,      # Early features
            num_blocks // 3,      # 1/4 depth
            num_blocks // 2,      # 1/2 depth
            num_blocks - 1,       # Near end
            num_blocks - 1        # Final (same as x4)
        ]
        
        # Channel dimensions for each scale
        if 'vits' in self.backbone_name:
            self.scale_channels = [384, 384, 384, 384, 384]
        elif 'vitb' in self.backbone_name:
            self.scale_channels = [768, 768, 768, 768, 768]
        elif 'vitl' in self.backbone_name:
            self.scale_channels = [1024, 1024, 1024, 1024, 1024]
        elif 'vitg' in self.backbone_name:
            self.scale_channels = [1536, 1536, 1536, 1536, 1536]
        else:
            self.scale_channels = [self.embed_dim] * 5
    
    def pad_to_patch_size(self, x):
        """
        Pad image to be divisible by patch size
        
        Args:
            x: (B, C, H, W)
            
        Returns:
            x_padded: (B, C, H_pad, W_pad)
            pad_info: (pad_left, pad_right, pad_top, pad_bottom, orig_h, orig_w)
        """
        B, C, H, W = x.shape
        
        # Calculate padding needed
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0, H, W)
        
        # Pad (left, right, top, bottom)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        
        return x_padded, (pad_left, pad_right, pad_top, pad_bottom, H, W)
    
    def reshape_vit_output(self, x, h, w):
        """
        Reshape ViT patch tokens to spatial feature map
        
        Args:
            x: (B, N, C) - patch tokens
            h, w: height and width in patches
            
        Returns:
            (B, C, H, W) - spatial feature map
        """
        B, N, C = x.shape
        
        # Remove CLS token if present
        if N == h * w + 1:
            x = x[:, 1:, :]  # Remove first token (CLS)
        elif N != h * w:
            # Adjust if mismatch (shouldn't happen but safety check)
            x = x[:, :h*w, :]
        
        # Reshape to spatial
        x = x.reshape(B, h, w, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return x
    
    def extract_features(self, x):
        """
        Extract multi-scale features from ViT
        
        Args:
            x: (B, C, H, W) - input image
            
        Returns:
            features: Dict with keys [x0, x1, x2, x3, x4]
        """
        B, C, H, W = x.shape
        
        # Pad to patch size if needed
        x_padded, pad_info = self.pad_to_patch_size(x)
        pad_left, pad_right, pad_top, pad_bottom, orig_h, orig_w = pad_info
        H_pad, W_pad = x_padded.shape[2], x_padded.shape[3]
        
        # Calculate patch grid size
        h_patches = H_pad // self.patch_size
        w_patches = W_pad // self.patch_size
        
        # Use ViT's forward_features method which handles everything properly
        if hasattr(self.vit, 'forward_features'):
            x_patches = self.vit.forward_features(x_padded)
            # x_patches is (B, num_patches + 1, embed_dim) if CLS token exists
        else:
            # Manual forward through ViT layers
            x_patches = self.vit.patch_embed(x_padded)
            
            # Flatten if needed (B, C, H, W) -> (B, H*W, C)
            if len(x_patches.shape) == 4:
                B_p, C_p, H_p, W_p = x_patches.shape
                x_patches = x_patches.flatten(2).transpose(1, 2)
            
            # Add CLS token
            if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
                cls_tokens = self.vit.cls_token.expand(B, -1, -1)
                x_patches = torch.cat((cls_tokens, x_patches), dim=1)
            
            # Add position embedding
            if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
                x_patches = x_patches + self.vit.pos_embed
            
            # Dropout
            if hasattr(self.vit, 'pos_drop'):
                x_patches = self.vit.pos_drop(x_patches)
            
            # Pass through transformer blocks
            for block in self.vit.blocks:
                x_patches = block(x_patches)
            
            # Norm
            x_patches = self.vit.norm(x_patches)
        
        # For multi-scale features, we'll use the same final features
        # but resize them to different scales
        # This is simpler and works well for ViTs
        
        # Convert to spatial
        spatial = self.reshape_vit_output(x_patches, h_patches, w_patches)
        
        # Create multi-scale features by resizing
        # x0: 1/2, x1: 1/4, x2: 1/8, x3: 1/16, x4: 1/32
        sizes = {
            'x0': (orig_h // 2, orig_w // 2),
            'x1': (orig_h // 4, orig_w // 4),
            'x2': (orig_h // 8, orig_w // 8),
            'x3': (orig_h // 16, orig_w // 16),
            'x4': (orig_h // 32, orig_w // 32)
        }
        
        features = {}
        for key, (target_h, target_w) in sizes.items():
            feat = F.interpolate(
                spatial,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
            features[key] = feat
        
        return features
    
    def forward(self, x):
        """Forward pass - just extract features"""
        return self.extract_features(x)
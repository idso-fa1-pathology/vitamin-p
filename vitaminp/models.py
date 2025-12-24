"""
Vitamin-P: DINOv2 U-Net model architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv2Backbone
from .blocks import ConvBlock


class VitaminPDual(nn.Module):
    """
    Vitamin-P Dual: Dual-Encoder DINOv2 U-Net for H&E + MIF pathology images
    
    Architecture:
    - H&E Encoder (3ch): DINOv2 layers 0-1 (separate)
    - MIF Encoder (2ch): DINOv2 layers 0-1 (separate)
    - Fusion: Concatenate layer_1 features
    - Shared Encoder: DINOv2 layers 2-3 (shared)
    - 4 Decoders: HE nuclei, HE cell, MIF nuclei, MIF cell
    
    Args:
        model_size: One of 'small', 'base', 'large', 'giant'
        dropout_rate: Dropout probability for regularization
        freeze_backbone: Whether to freeze DINOv2 weights
    
    Example:
        >>> model = VitaminPDual(model_size='base', dropout_rate=0.3)
        >>> he_img = torch.randn(2, 3, 512, 512)
        >>> mif_img = torch.randn(2, 2, 512, 512)
        >>> outputs = model(he_img, mif_img)
    """
    def __init__(self, model_size='base', dropout_rate=0.3, freeze_backbone=False):
        super().__init__()
        
        self.model_size = model_size
        self.dropout_rate = dropout_rate
        
        # ========== H&E ENCODER (layers 0-1) ==========
        print(f"Building H&E encoder with DINOv2-{model_size}")
        self.he_backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        embed_dim = self.he_backbone.embed_dim
        
        # ========== MIF ENCODER (layers 0-1) ==========
        print(f"Building MIF encoder with DINOv2-{model_size}")
        self.mif_backbone = DINOv2Backbone(model_size=model_size, in_channels=2)
        
        # ========== SHARED ENCODER (layers 2-3) ==========
        print(f"Building shared encoder with DINOv2-{model_size}")
        self.shared_backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        
        if freeze_backbone:
            for param in self.he_backbone.parameters():
                param.requires_grad = False
            for param in self.mif_backbone.parameters():
                param.requires_grad = False
            for param in self.shared_backbone.parameters():
                param.requires_grad = False
        
        # ========== FUSION LAYER ==========
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Dropout layers
        self.dropout_high = nn.Dropout2d(dropout_rate * 1.5)
        self.dropout_mid = nn.Dropout2d(dropout_rate)
        
        # Decoder channel configuration
        if model_size == 'small':
            dec_dims = [384, 256, 128, 64]
        elif model_size == 'base':
            dec_dims = [768, 384, 192, 96]
        elif model_size == 'large':
            dec_dims = [1024, 512, 256, 128]
        elif model_size == 'giant':
            dec_dims = [1536, 768, 384, 192]
        
        self.dec_dims = dec_dims
        self.embed_dim = embed_dim
        
        # Projection layers for skip connections
        self.proj3 = nn.Conv2d(embed_dim, dec_dims[0], 1)
        self.proj2 = nn.Conv2d(embed_dim, dec_dims[1], 1)
        self.proj1 = nn.Conv2d(embed_dim, dec_dims[2], 1)
        self.proj0 = nn.Conv2d(embed_dim, dec_dims[3], 1)
        
        # ========== DECODERS ==========
        self._build_decoders()
        
        print(f"✓ VitaminPDual initialized with {model_size} backbone")
        print(f"  Embed dim: {embed_dim} | Decoder dims: {dec_dims}")
    
    def _build_decoders(self):
        """Build 4 separate decoder branches"""
        dec_dims = self.dec_dims
        dropout_rate = self.dropout_rate
        
        # ========== HE NUCLEI DECODER ==========
        self.dec1_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dropout_rate * 1.2)
        self.dec2_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dropout_rate)
        self.dec3_he_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dropout_rate * 0.8)
        self.dec4_he_nuclei = ConvBlock(dec_dims[2] + dec_dims[3] * 2, dec_dims[3], dropout_rate * 0.5)
        self.final_he_nuclei = ConvBlock(dec_dims[3], 32, dropout_rate * 0.3)
        self.head_he_nuclei = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
        # ========== HE CELL DECODER ==========
        self.dec1_he_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dropout_rate * 1.2)
        self.dec2_he_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dropout_rate)
        self.dec3_he_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dropout_rate * 0.8)
        self.dec4_he_cell = ConvBlock(dec_dims[2] + dec_dims[3] * 2, dec_dims[3], dropout_rate * 0.5)
        self.final_he_cell = ConvBlock(dec_dims[3], 32, dropout_rate * 0.3)
        self.head_he_cell = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
        # ========== MIF NUCLEI DECODER ==========
        self.dec1_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dropout_rate * 1.2)
        self.dec2_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dropout_rate)
        self.dec3_mif_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dropout_rate * 0.8)
        self.dec4_mif_nuclei = ConvBlock(dec_dims[2] + dec_dims[3] * 2, dec_dims[3], dropout_rate * 0.5)
        self.final_mif_nuclei = ConvBlock(dec_dims[3], 32, dropout_rate * 0.3)
        self.head_mif_nuclei = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
        # ========== MIF CELL DECODER ==========
        self.dec1_mif_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dropout_rate * 1.2)
        self.dec2_mif_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dropout_rate)
        self.dec3_mif_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dropout_rate * 0.8)
        self.dec4_mif_cell = ConvBlock(dec_dims[2] + dec_dims[3] * 2, dec_dims[3], dropout_rate * 0.5)
        self.final_mif_cell = ConvBlock(dec_dims[3], 32, dropout_rate * 0.3)
        self.head_mif_cell = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
    
    def decode_branch(self, x3, x2, fused_x1, x1, x0_he, x0_mif, branch_name):
        """Generic decoder branch with skip connections"""
        dec1 = getattr(self, f'dec1_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
        final_conv = getattr(self, f'final_{branch_name}')
        head = getattr(self, f'head_{branch_name}')
        
        # Upsample and decode
        d = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj3(x2)], dim=1)
        d = dec1(d)
        
        d = F.interpolate(d, size=fused_x1.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj2(fused_x1)], dim=1)
        d = dec2(d)
        
        d = F.interpolate(d, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj1(x1)], dim=1)
        d = dec3(d)
        
        # Combine early features from both modalities
        target_h = x1.shape[2] * 2
        d = F.interpolate(d, size=(target_h, target_h), mode='bilinear', align_corners=False)
        x0_he_proj = F.interpolate(self.proj0(x0_he), size=(target_h, target_h), mode='bilinear', align_corners=False)
        x0_mif_proj = F.interpolate(self.proj0(x0_mif), size=(target_h, target_h), mode='bilinear', align_corners=False)
        x0_combined = torch.cat([x0_he_proj, x0_mif_proj], dim=1)
        
        d = torch.cat([d, x0_combined], dim=1)
        d = dec4(d)
        
        feat = final_conv(d)
        
        # Final upsampling to 512x512
        if feat.shape[2] != 512 or feat.shape[3] != 512:
            feat = F.interpolate(feat, size=(512, 512), mode='bilinear', align_corners=False)
        
        out = head(feat)
        
        seg_out = torch.sigmoid(out[:, 0:1])
        hv_out = torch.tanh(out[:, 1:3])
        
        return seg_out, hv_out
    
    def forward(self, he_img, mif_img):
        """
        Forward pass
        
        Args:
            he_img: (B, 3, H, W) - H&E image
            mif_img: (B, 2, H, W) - MIF image
        
        Returns:
            Dictionary with segmentation and HV outputs for all 4 branches
        """
        # Resize inputs to 518x518 for DINOv2
        if he_img.shape[2] != 518 or he_img.shape[3] != 518:
            he_img = F.interpolate(he_img, size=(518, 518), mode='bilinear', align_corners=False)
        if mif_img.shape[2] != 518 or mif_img.shape[3] != 518:
            mif_img = F.interpolate(mif_img, size=(518, 518), mode='bilinear', align_corners=False)
        
        # ========== H&E ENCODER (layers 0-1) ==========
        he_features = self.he_backbone(he_img)
        he_layer_keys = sorted([k for k in he_features.keys()])
        he_x0 = he_features[he_layer_keys[0]]
        he_x1 = he_features[he_layer_keys[1]]
        
        # ========== MIF ENCODER (layers 0-1) ==========
        mif_features = self.mif_backbone(mif_img)
        mif_layer_keys = sorted([k for k in mif_features.keys()])
        mif_x0 = mif_features[mif_layer_keys[0]]
        mif_x1 = mif_features[mif_layer_keys[1]]
        
        # ========== FUSION at layer 1 ==========
        fused_x1 = self.fusion_conv(torch.cat([he_x1, mif_x1], dim=1))
        
        # ========== SHARED ENCODER (layers 2-3) ==========
        shared_features = self.shared_backbone(he_img)
        shared_layer_keys = sorted([k for k in shared_features.keys()])
        x2 = shared_features[shared_layer_keys[2]]
        x3 = shared_features[shared_layer_keys[3]]
        
        if self.training:
            x2 = self.dropout_mid(x2)
            x3 = self.dropout_high(x3)
        
        # ========== DECODERS ==========
        he_nuclei_seg, he_nuclei_hv = self.decode_branch(x3, x2, fused_x1, he_x1, he_x0, mif_x0, 'he_nuclei')
        he_cell_seg, he_cell_hv = self.decode_branch(x3, x2, fused_x1, he_x1, he_x0, mif_x0, 'he_cell')
        mif_nuclei_seg, mif_nuclei_hv = self.decode_branch(x3, x2, fused_x1, mif_x1, he_x0, mif_x0, 'mif_nuclei')
        mif_cell_seg, mif_cell_hv = self.decode_branch(x3, x2, fused_x1, mif_x1, he_x0, mif_x0, 'mif_cell')
        
        return {
            'he_nuclei_seg': he_nuclei_seg,
            'he_nuclei_hv': he_nuclei_hv,
            'he_cell_seg': he_cell_seg,
            'he_cell_hv': he_cell_hv,
            'mif_nuclei_seg': mif_nuclei_seg,
            'mif_nuclei_hv': mif_nuclei_hv,
            'mif_cell_seg': mif_cell_seg,
            'mif_cell_hv': mif_cell_hv,
        }


class VitaminPFlex(nn.Module):
    """
    Vitamin-P Flex: Single-Encoder with 4 Separate Decoders
    
    Architecture:
    - Shared Encoder: Single DINOv2 backbone for both H&E and MIF
    - 4 Separate Decoders: HE nuclei, HE cell, MIF nuclei, MIF cell
    - Training: Random modality selection per sample
    
    Args:
        model_size: One of 'small', 'base', 'large', 'giant'
        freeze_backbone: Whether to freeze DINOv2 weights
    
    Example:
        >>> model = VitaminPFlex(model_size='base')
        >>> # Can accept either H&E (3ch) or MIF (2ch+1zero = 3ch)
        >>> img = torch.randn(2, 3, 512, 512)
        >>> outputs = model(img)
    """
    def __init__(self, model_size='base', freeze_backbone=False):
        super().__init__()
        
        self.model_size = model_size
        self.backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Decoder channel configuration
        if model_size == 'small':
            dec_dims = [384, 256, 128, 64]
        elif model_size == 'base':
            dec_dims = [768, 384, 192, 96]
        elif model_size == 'large':
            dec_dims = [1024, 512, 256, 128]
        elif model_size == 'giant':
            dec_dims = [1536, 768, 384, 192]
        
        self.dec_dims = dec_dims
        self.embed_dim = embed_dim
        
        # Projection layers for skip connections
        self.proj3 = nn.Conv2d(embed_dim, dec_dims[0], 1)
        self.proj2 = nn.Conv2d(embed_dim, dec_dims[1], 1)
        self.proj1 = nn.Conv2d(embed_dim, dec_dims[2], 1)
        self.proj0 = nn.Conv2d(embed_dim, dec_dims[3], 1)
        
        # Build decoders
        self._build_decoders()
        
        print(f"✓ VitaminPFlex initialized with {model_size} backbone")
        print(f"  Architecture: Shared Encoder → 4 Separate Decoders")
        print(f"  Embed dim: {embed_dim} | Decoder dims: {dec_dims}")
    
    def _build_decoders(self):
        """Build 4 separate decoder branches"""
        dec_dims = self.dec_dims
        
        # ========== HE NUCLEI DECODER ==========
        self.dec1_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_he_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_he_nuclei = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.final_he_nuclei = ConvBlock(dec_dims[3], 32)
        self.head_he_nuclei = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
        # ========== HE CELL DECODER ==========
        self.dec1_he_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_he_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_he_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_he_cell = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.final_he_cell = ConvBlock(dec_dims[3], 32)
        self.head_he_cell = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
        # ========== MIF NUCLEI DECODER ==========
        self.dec1_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_mif_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_mif_nuclei = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.final_mif_nuclei = ConvBlock(dec_dims[3], 32)
        self.head_mif_nuclei = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
        # ========== MIF CELL DECODER ==========
        self.dec1_mif_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_mif_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_mif_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_mif_cell = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.final_mif_cell = ConvBlock(dec_dims[3], 32)
        self.head_mif_cell = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
    
    def decode_branch(self, e5, e4, e3, e2, branch_name):
        """Generic decoder branch with skip connections"""
        dec1 = getattr(self, f'dec1_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
        final_conv = getattr(self, f'final_{branch_name}')
        head = getattr(self, f'head_{branch_name}')
        
        # Decoder path with skip connections
        d = F.interpolate(e5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj3(e4)], dim=1)
        d = dec1(d)
        
        d = F.interpolate(d, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj2(e3)], dim=1)
        d = dec2(d)
        
        d = F.interpolate(d, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj1(e2)], dim=1)
        d = dec3(d)
        
        target_h = e2.shape[2] * 2
        d = F.interpolate(d, size=(target_h, target_h), mode='bilinear', align_corners=False)
        e2_up = F.interpolate(self.proj0(e2), size=(target_h, target_h), mode='bilinear', align_corners=False)
        d = torch.cat([d, e2_up], dim=1)
        d = dec4(d)
        
        feat = final_conv(d)
        
        # Final upsampling to 512x512
        if feat.shape[2] != 512 or feat.shape[3] != 512:
            feat = F.interpolate(feat, size=(512, 512), mode='bilinear', align_corners=False)
        
        out = head(feat)
        
        seg_out = torch.sigmoid(out[:, 0:1])
        hv_out = torch.tanh(out[:, 1:3])
        
        return seg_out, hv_out
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, 3, H, W) - Can be H&E or MIF (with zero channel)
        
        Returns:
            Dictionary with segmentation and HV outputs for all 4 branches
        """
        B, C, H, W = x.shape
        
        # Resize to 518x518 for DINOv2
        if H != 518 or W != 518:
            x = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)
        
        # ========== SHARED ENCODER ==========
        features = self.backbone(x)
        layer_keys = sorted([k for k in features.keys()])
        
        e5 = features[layer_keys[3]]  # Deepest
        e4 = features[layer_keys[2]]  # Deep
        e3 = features[layer_keys[1]]  # Mid
        e2 = features[layer_keys[0]]  # Early
        
        # ========== 4 SEPARATE DECODERS ==========
        he_nuclei_seg, he_nuclei_hv = self.decode_branch(e5, e4, e3, e2, 'he_nuclei')
        he_cell_seg, he_cell_hv = self.decode_branch(e5, e4, e3, e2, 'he_cell')
        mif_nuclei_seg, mif_nuclei_hv = self.decode_branch(e5, e4, e3, e2, 'mif_nuclei')
        mif_cell_seg, mif_cell_hv = self.decode_branch(e5, e4, e3, e2, 'mif_cell')
        
        return {
            'he_nuclei_seg': he_nuclei_seg,
            'he_nuclei_hv': he_nuclei_hv,
            'he_cell_seg': he_cell_seg,
            'he_cell_hv': he_cell_hv,
            'mif_nuclei_seg': mif_nuclei_seg,
            'mif_nuclei_hv': mif_nuclei_hv,
            'mif_cell_seg': mif_cell_seg,
            'mif_cell_hv': mif_cell_hv,
        }
    
class VitaminPBaselineHE(nn.Module):
    """
    Vitamin-P Baseline H&E: Single-Encoder for H&E-only with 2 Decoders
    
    Architecture:
    - H&E Encoder (3ch): Single DINOv2 backbone
    - 2 Decoders: HE nuclei, HE cell
    - Baseline model for comparison with Dual and Flex
    
    Args:
        model_size: One of 'small', 'base', 'large', 'giant'
        dropout_rate: Dropout probability for regularization
        freeze_backbone: Whether to freeze DINOv2 weights
    
    Example:
        >>> model = VitaminPBaselineHE(model_size='base')
        >>> he_img = torch.randn(2, 3, 512, 512)
        >>> outputs = model(he_img)
    """
    def __init__(self, model_size='base', dropout_rate=0.3, freeze_backbone=False):
        super().__init__()
        
        self.model_size = model_size
        self.dropout_rate = dropout_rate
        
        # ========== H&E ENCODER ==========
        print(f"Building H&E Baseline encoder with DINOv2-{model_size}")
        self.backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Dropout layers
        self.dropout_high = nn.Dropout2d(dropout_rate * 1.5)
        self.dropout_mid = nn.Dropout2d(dropout_rate)
        
        # Decoder channel configuration
        if model_size == 'small':
            dec_dims = [384, 256, 128, 64]
        elif model_size == 'base':
            dec_dims = [768, 384, 192, 96]
        elif model_size == 'large':
            dec_dims = [1024, 512, 256, 128]
        elif model_size == 'giant':
            dec_dims = [1536, 768, 384, 192]
        
        self.dec_dims = dec_dims
        self.embed_dim = embed_dim
        
        # Projection layers for skip connections
        self.proj3 = nn.Conv2d(embed_dim, dec_dims[0], 1)
        self.proj2 = nn.Conv2d(embed_dim, dec_dims[1], 1)
        self.proj1 = nn.Conv2d(embed_dim, dec_dims[2], 1)
        self.proj0 = nn.Conv2d(embed_dim, dec_dims[3], 1)
        
        # ========== DECODERS ==========
        self._build_decoders()
        
        print(f"✓ VitaminPBaselineHE initialized with {model_size} backbone")
        print(f"  Embed dim: {embed_dim} | Decoder dims: {dec_dims}")
    
    def _build_decoders(self):
        """Build 2 decoder branches for H&E nuclei and cell"""
        dec_dims = self.dec_dims
        dropout_rate = self.dropout_rate
        
        # ========== HE NUCLEI DECODER ==========
        self.dec1_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dropout_rate * 1.2)
        self.dec2_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dropout_rate)
        self.dec3_he_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dropout_rate * 0.8)
        self.dec4_he_nuclei = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3], dropout_rate * 0.5)
        self.final_he_nuclei = ConvBlock(dec_dims[3], 32, dropout_rate * 0.3)
        self.head_he_nuclei = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
        # ========== HE CELL DECODER ==========
        self.dec1_he_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dropout_rate * 1.2)
        self.dec2_he_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dropout_rate)
        self.dec3_he_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dropout_rate * 0.8)
        self.dec4_he_cell = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3], dropout_rate * 0.5)
        self.final_he_cell = ConvBlock(dec_dims[3], 32, dropout_rate * 0.3)
        self.head_he_cell = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
    
    def decode_branch(self, x3, x2, x1, x0, branch_name):
        """Generic decoder branch with skip connections"""
        dec1 = getattr(self, f'dec1_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
        final_conv = getattr(self, f'final_{branch_name}')
        head = getattr(self, f'head_{branch_name}')
        
        # Upsample and decode
        d = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj3(x2)], dim=1)
        d = dec1(d)
        
        d = F.interpolate(d, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj2(x1)], dim=1)
        d = dec2(d)
        
        d = F.interpolate(d, size=x0.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj1(x0)], dim=1)
        d = dec3(d)
        
        target_h = x0.shape[2] * 2
        d = F.interpolate(d, size=(target_h, target_h), mode='bilinear', align_corners=False)
        x0_proj = F.interpolate(self.proj0(x0), size=(target_h, target_h), mode='bilinear', align_corners=False)
        
        d = torch.cat([d, x0_proj], dim=1)
        d = dec4(d)
        
        feat = final_conv(d)
        
        # Final upsampling to 512x512
        if feat.shape[2] != 512 or feat.shape[3] != 512:
            feat = F.interpolate(feat, size=(512, 512), mode='bilinear', align_corners=False)
        
        out = head(feat)
        
        seg_out = torch.sigmoid(out[:, 0:1])
        hv_out = torch.tanh(out[:, 1:3])
        
        return seg_out, hv_out
    
    def forward(self, he_img):
        """
        Forward pass
        
        Args:
            he_img: (B, 3, H, W) - H&E image
        
        Returns:
            Dictionary with segmentation and HV outputs for HE nuclei and HE cell
        """
        # Resize inputs to 518x518 for DINOv2
        if he_img.shape[2] != 518 or he_img.shape[3] != 518:
            he_img = F.interpolate(he_img, size=(518, 518), mode='bilinear', align_corners=False)
        
        # ========== H&E ENCODER ==========
        features = self.backbone(he_img)
        layer_keys = sorted([k for k in features.keys()])
        x0 = features[layer_keys[0]]
        x1 = features[layer_keys[1]]
        x2 = features[layer_keys[2]]
        x3 = features[layer_keys[3]]
        
        if self.training:
            x2 = self.dropout_mid(x2)
            x3 = self.dropout_high(x3)
        
        # ========== DECODERS ==========
        he_nuclei_seg, he_nuclei_hv = self.decode_branch(x3, x2, x1, x0, 'he_nuclei')
        he_cell_seg, he_cell_hv = self.decode_branch(x3, x2, x1, x0, 'he_cell')
        
        return {
            'he_nuclei_seg': he_nuclei_seg,
            'he_nuclei_hv': he_nuclei_hv,
            'he_cell_seg': he_cell_seg,
            'he_cell_hv': he_cell_hv,
        }


class VitaminPBaselineMIF(nn.Module):
    """
    Vitamin-P Baseline MIF: Single-Encoder for MIF-only with 2 Decoders
    
    Architecture:
    - MIF Encoder (2ch): Single DINOv2 backbone
    - 2 Decoders: MIF nuclei, MIF cell
    - Baseline model for comparison with Dual and Flex
    
    Args:
        model_size: One of 'small', 'base', 'large', 'giant'
        dropout_rate: Dropout probability for regularization
        freeze_backbone: Whether to freeze DINOv2 weights
    
    Example:
        >>> model = VitaminPBaselineMIF(model_size='base')
        >>> mif_img = torch.randn(2, 2, 512, 512)
        >>> outputs = model(mif_img)
    """
    def __init__(self, model_size='base', dropout_rate=0.3, freeze_backbone=False):
        super().__init__()
        
        self.model_size = model_size
        self.dropout_rate = dropout_rate
        
        # ========== MIF ENCODER ==========
        print(f"Building MIF Baseline encoder with DINOv2-{model_size}")
        self.backbone = DINOv2Backbone(model_size=model_size, in_channels=2)
        embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Dropout layers
        self.dropout_high = nn.Dropout2d(dropout_rate * 1.5)
        self.dropout_mid = nn.Dropout2d(dropout_rate)
        
        # Decoder channel configuration
        if model_size == 'small':
            dec_dims = [384, 256, 128, 64]
        elif model_size == 'base':
            dec_dims = [768, 384, 192, 96]
        elif model_size == 'large':
            dec_dims = [1024, 512, 256, 128]
        elif model_size == 'giant':
            dec_dims = [1536, 768, 384, 192]
        
        self.dec_dims = dec_dims
        self.embed_dim = embed_dim
        
        # Projection layers for skip connections
        self.proj3 = nn.Conv2d(embed_dim, dec_dims[0], 1)
        self.proj2 = nn.Conv2d(embed_dim, dec_dims[1], 1)
        self.proj1 = nn.Conv2d(embed_dim, dec_dims[2], 1)
        self.proj0 = nn.Conv2d(embed_dim, dec_dims[3], 1)
        
        # ========== DECODERS ==========
        self._build_decoders()
        
        print(f"✓ VitaminPBaselineMIF initialized with {model_size} backbone")
        print(f"  Embed dim: {embed_dim} | Decoder dims: {dec_dims}")
    
    def _build_decoders(self):
        """Build 2 decoder branches for MIF nuclei and cell"""
        dec_dims = self.dec_dims
        dropout_rate = self.dropout_rate
        
        # ========== MIF NUCLEI DECODER ==========
        self.dec1_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dropout_rate * 1.2)
        self.dec2_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dropout_rate)
        self.dec3_mif_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dropout_rate * 0.8)
        self.dec4_mif_nuclei = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3], dropout_rate * 0.5)
        self.final_mif_nuclei = ConvBlock(dec_dims[3], 32, dropout_rate * 0.3)
        self.head_mif_nuclei = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
        # ========== MIF CELL DECODER ==========
        self.dec1_mif_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dropout_rate * 1.2)
        self.dec2_mif_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dropout_rate)
        self.dec3_mif_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dropout_rate * 0.8)
        self.dec4_mif_cell = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3], dropout_rate * 0.5)
        self.final_mif_cell = ConvBlock(dec_dims[3], 32, dropout_rate * 0.3)
        self.head_mif_cell = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
    
    def decode_branch(self, x3, x2, x1, x0, branch_name):
        """Generic decoder branch with skip connections"""
        dec1 = getattr(self, f'dec1_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
        final_conv = getattr(self, f'final_{branch_name}')
        head = getattr(self, f'head_{branch_name}')
        
        # Upsample and decode
        d = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj3(x2)], dim=1)
        d = dec1(d)
        
        d = F.interpolate(d, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj2(x1)], dim=1)
        d = dec2(d)
        
        d = F.interpolate(d, size=x0.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, self.proj1(x0)], dim=1)
        d = dec3(d)
        
        target_h = x0.shape[2] * 2
        d = F.interpolate(d, size=(target_h, target_h), mode='bilinear', align_corners=False)
        x0_proj = F.interpolate(self.proj0(x0), size=(target_h, target_h), mode='bilinear', align_corners=False)
        
        d = torch.cat([d, x0_proj], dim=1)
        d = dec4(d)
        
        feat = final_conv(d)
        
        # Final upsampling to 512x512
        if feat.shape[2] != 512 or feat.shape[3] != 512:
            feat = F.interpolate(feat, size=(512, 512), mode='bilinear', align_corners=False)
        
        out = head(feat)
        
        seg_out = torch.sigmoid(out[:, 0:1])
        hv_out = torch.tanh(out[:, 1:3])
        
        return seg_out, hv_out
    
    def forward(self, mif_img):
        """
        Forward pass
        
        Args:
            mif_img: (B, 2, H, W) - MIF image
        
        Returns:
            Dictionary with segmentation and HV outputs for MIF nuclei and MIF cell
        """
        # Resize inputs to 518x518 for DINOv2
        if mif_img.shape[2] != 518 or mif_img.shape[3] != 518:
            mif_img = F.interpolate(mif_img, size=(518, 518), mode='bilinear', align_corners=False)
        
        # ========== MIF ENCODER ==========
        features = self.backbone(mif_img)
        layer_keys = sorted([k for k in features.keys()])
        x0 = features[layer_keys[0]]
        x1 = features[layer_keys[1]]
        x2 = features[layer_keys[2]]
        x3 = features[layer_keys[3]]
        
        if self.training:
            x2 = self.dropout_mid(x2)
            x3 = self.dropout_high(x3)
        
        # ========== DECODERS ==========
        mif_nuclei_seg, mif_nuclei_hv = self.decode_branch(x3, x2, x1, x0, 'mif_nuclei')
        mif_cell_seg, mif_cell_hv = self.decode_branch(x3, x2, x1, x0, 'mif_cell')
        
        return {
            'mif_nuclei_seg': mif_nuclei_seg,
            'mif_nuclei_hv': mif_nuclei_hv,
            'mif_cell_seg': mif_cell_seg,
            'mif_cell_hv': mif_cell_hv,
        }

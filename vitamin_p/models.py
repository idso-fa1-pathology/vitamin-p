"""
Vitamin-P Model Collection

Contains all Vitamin-P model variants:
1. VitaminPFlex - Flexible model with 4 separate decoders (shared encoder)
2. VitaminPDual - Dual encoder with mid-fusion
3. VitaminPHEBaseline - H&E only baseline (2 decoders)
4. VitaminPMIFBaseline - MIF only baseline (2 decoders)
5. VitaminPSynthetic - Synthetic MIF generation (placeholder for now)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv2Backbone
from .blocks import ConvBlock, SegmentationHead, get_decoder_dims, FusionLayer
from .utils import resize_to_dinov2, resize_to_output


# ============================================================================
# VITAMIN-P FLEX: Flexible model with 4 separate decoders
# ============================================================================

class VitaminPFlex(nn.Module):
    """
    Vitamin-P Flexible Model
    
    Architecture:
    - Single shared encoder (DINOv2)
    - 4 separate decoder branches:
        * HE Nuclei
        * HE Cell
        * MIF Nuclei
        * MIF Cell
    
    Can process either H&E or MIF images through the same encoder.
    
    Args:
        model_size (str): DINOv2 variant - 'small', 'base', 'large', 'giant'
        freeze_backbone (bool): Whether to freeze backbone weights
    """
    
    def __init__(self, model_size='base', freeze_backbone=False):
        super().__init__()
        
        self.model_size = model_size
        self.backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get decoder dimensions
        dec_dims = get_decoder_dims(model_size)
        self.dec_dims = dec_dims
        self.embed_dim = embed_dim
        
        # Projection layers for skip connections
        self.proj3 = nn.Conv2d(embed_dim, dec_dims[0], 1)
        self.proj2 = nn.Conv2d(embed_dim, dec_dims[1], 1)
        self.proj1 = nn.Conv2d(embed_dim, dec_dims[2], 1)
        self.proj0 = nn.Conv2d(embed_dim, dec_dims[3], 1)
        
        # Build all 4 decoder branches
        self._build_decoders()
        
        print(f"✓ VitaminPFlex initialized with {model_size} backbone")
        print(f"  Embed dim: {embed_dim} | Decoder dims: {dec_dims}")
    
    def _build_decoders(self):
        """Build 4 separate decoder branches"""
        dec_dims = self.dec_dims
        
        # HE NUCLEI DECODER
        self.dec1_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_he_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_he_nuclei = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.head_he_nuclei = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        # HE CELL DECODER
        self.dec1_he_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_he_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_he_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_he_cell = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.head_he_cell = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        # MIF NUCLEI DECODER
        self.dec1_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_mif_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_mif_nuclei = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.head_mif_nuclei = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        # MIF CELL DECODER
        self.dec1_mif_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_mif_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_mif_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_mif_cell = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.head_mif_cell = SegmentationHead(dec_dims[3], hidden_ch=32)
    
    def decode_branch(self, e5, e4, e3, e2, branch_name):
        """Generic decoder branch with skip connections"""
        dec1 = getattr(self, f'dec1_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
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
        
        # Head: produce segmentation + HV map
        seg_out, hv_out = head(d)
        
        # Final upsampling to 512x512
        seg_out = resize_to_output(seg_out, 512)
        hv_out = resize_to_output(hv_out, 512)
        
        return seg_out, hv_out
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image (B, 3, H, W) - can be H&E or MIF
        
        Returns:
            dict: Predictions for all 4 branches
        """
        # Resize to 518x518 for DINOv2
        x = resize_to_dinov2(x, 518)
        
        # Extract features
        features = self.backbone(x)
        layer_keys = sorted([k for k in features.keys()])
        
        e5 = features[layer_keys[3]]  # Deepest
        e4 = features[layer_keys[2]]
        e3 = features[layer_keys[1]]
        e2 = features[layer_keys[0]]  # Earliest
        
        # Decode all 4 branches
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


# ============================================================================
# VITAMIN-P DUAL: Dual encoder with mid-fusion
# ============================================================================

class VitaminPDual(nn.Module):
    """
    Vitamin-P Dual Encoder Model
    
    Architecture:
    - Separate H&E encoder (3ch, DINOv2 layers 0-1)
    - Separate MIF encoder (2ch, DINOv2 layers 0-1)
    - Mid-level fusion of H&E + MIF features
    - Shared encoder (DINOv2 layers 2-3)
    - 4 separate decoder branches
    
    Args:
        model_size (str): DINOv2 variant
        dropout_rate (float): Dropout probability
        freeze_backbone (bool): Whether to freeze backbone weights
    """
    
    def __init__(self, model_size='base', dropout_rate=0.3, freeze_backbone=False):
        super().__init__()
        
        self.model_size = model_size
        self.dropout_rate = dropout_rate
        
        # H&E encoder (3 channels)
        self.he_backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        embed_dim = self.he_backbone.embed_dim
        
        # MIF encoder (2 channels)
        self.mif_backbone = DINOv2Backbone(model_size=model_size, in_channels=2)
        
        # Shared encoder (3 channels)
        self.shared_backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        
        if freeze_backbone:
            for param in self.he_backbone.parameters():
                param.requires_grad = False
            for param in self.mif_backbone.parameters():
                param.requires_grad = False
            for param in self.shared_backbone.parameters():
                param.requires_grad = False
        
        # Fusion layer
        self.fusion_conv = FusionLayer(embed_dim * 2, embed_dim)
        
        # Dropout
        self.dropout_high = nn.Dropout2d(dropout_rate * 1.5)
        self.dropout_mid = nn.Dropout2d(dropout_rate)
        
        # Decoder dims
        dec_dims = get_decoder_dims(model_size)
        self.dec_dims = dec_dims
        self.embed_dim = embed_dim
        
        # Projections
        self.proj3 = nn.Conv2d(embed_dim, dec_dims[0], 1)
        self.proj2 = nn.Conv2d(embed_dim, dec_dims[1], 1)
        self.proj1 = nn.Conv2d(embed_dim, dec_dims[2], 1)
        self.proj0 = nn.Conv2d(embed_dim, dec_dims[3], 1)
        
        # Build decoders
        self._build_decoders()
        
        print(f"✓ VitaminPDual initialized with {model_size} backbone")
        print(f"  Architecture: Dual encoder → Fusion → Shared encoder → 4 decoders")
    
    def _build_decoders(self):
        """Build 4 separate decoder branches with dropout"""
        dec_dims = self.dec_dims
        dr = self.dropout_rate
        
        # HE NUCLEI
        self.dec1_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dr * 1.2)
        self.dec2_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dr)
        self.dec3_he_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dr * 0.8)
        self.dec4_he_nuclei = ConvBlock(dec_dims[2] + dec_dims[3] * 2, dec_dims[3], dr * 0.5)
        self.head_he_nuclei = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        # HE CELL
        self.dec1_he_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dr * 1.2)
        self.dec2_he_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dr)
        self.dec3_he_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dr * 0.8)
        self.dec4_he_cell = ConvBlock(dec_dims[2] + dec_dims[3] * 2, dec_dims[3], dr * 0.5)
        self.head_he_cell = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        # MIF NUCLEI
        self.dec1_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dr * 1.2)
        self.dec2_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dr)
        self.dec3_mif_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dr * 0.8)
        self.dec4_mif_nuclei = ConvBlock(dec_dims[2] + dec_dims[3] * 2, dec_dims[3], dr * 0.5)
        self.head_mif_nuclei = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        # MIF CELL
        self.dec1_mif_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0], dr * 1.2)
        self.dec2_mif_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1], dr)
        self.dec3_mif_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2], dr * 0.8)
        self.dec4_mif_cell = ConvBlock(dec_dims[2] + dec_dims[3] * 2, dec_dims[3], dr * 0.5)
        self.head_mif_cell = SegmentationHead(dec_dims[3], hidden_ch=32)
    
    def decode_branch(self, x3, x2, fused_x1, x1, x0_he, x0_mif, branch_name):
        """Decoder branch with early feature fusion"""
        dec1 = getattr(self, f'dec1_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
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
        
        # Head
        seg_out, hv_out = head(d)
        
        # Final upsampling
        seg_out = resize_to_output(seg_out, 512)
        hv_out = resize_to_output(hv_out, 512)
        
        return seg_out, hv_out
    
    def forward(self, he_img, mif_img):
        """
        Args:
            he_img (torch.Tensor): H&E image (B, 3, H, W)
            mif_img (torch.Tensor): MIF image (B, 2, H, W)
        
        Returns:
            dict: Predictions for all 4 branches
        """
        # Resize inputs
        he_img = resize_to_dinov2(he_img, 518)
        mif_img = resize_to_dinov2(mif_img, 518)
        
        # H&E encoder (layers 0-1)
        he_features = self.he_backbone(he_img)
        he_layer_keys = sorted([k for k in he_features.keys()])
        he_x0 = he_features[he_layer_keys[0]]
        he_x1 = he_features[he_layer_keys[1]]
        
        # MIF encoder (layers 0-1)
        mif_features = self.mif_backbone(mif_img)
        mif_layer_keys = sorted([k for k in mif_features.keys()])
        mif_x0 = mif_features[mif_layer_keys[0]]
        mif_x1 = mif_features[mif_layer_keys[1]]
        
        # Fusion at layer 1
        fused_x1 = self.fusion_conv(he_x1, mif_x1)
        
        # Shared encoder (layers 2-3)
        shared_features = self.shared_backbone(he_img)
        shared_layer_keys = sorted([k for k in shared_features.keys()])
        x2 = shared_features[shared_layer_keys[2]]
        x3 = shared_features[shared_layer_keys[3]]
        
        if self.training:
            x2 = self.dropout_mid(x2)
            x3 = self.dropout_high(x3)
        
        # Decode all branches
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


# ============================================================================
# VITAMIN-P HE BASELINE: H&E only with 2 decoders
# ============================================================================

class VitaminPHEBaseline(nn.Module):
    """
    Vitamin-P H&E Baseline Model
    
    Architecture:
    - Single encoder (DINOv2, 3ch for H&E)
    - 2 decoder branches:
        * HE Nuclei
        * HE Cell
    
    Args:
        model_size (str): DINOv2 variant
        freeze_backbone (bool): Whether to freeze backbone weights
    """
    
    def __init__(self, model_size='base', freeze_backbone=False):
        super().__init__()
        
        self.model_size = model_size
        self.backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        dec_dims = get_decoder_dims(model_size)
        self.dec_dims = dec_dims
        self.embed_dim = embed_dim
        
        # Projections
        self.proj3 = nn.Conv2d(embed_dim, dec_dims[0], 1)
        self.proj2 = nn.Conv2d(embed_dim, dec_dims[1], 1)
        self.proj1 = nn.Conv2d(embed_dim, dec_dims[2], 1)
        self.proj0 = nn.Conv2d(embed_dim, dec_dims[3], 1)
        
        # HE NUCLEI decoder
        self.dec1_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_he_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_he_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_he_nuclei = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.head_he_nuclei = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        # HE CELL decoder
        self.dec1_he_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_he_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_he_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_he_cell = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.head_he_cell = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        print(f"✓ VitaminPHEBaseline initialized with {model_size} backbone")
        print(f"  Architecture: H&E encoder → 2 decoders (Nuclei, Cell)")
    
    def decode_branch(self, e5, e4, e3, e2, branch_name):
        """Decoder branch"""
        dec1 = getattr(self, f'dec1_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
        head = getattr(self, f'head_{branch_name}')
        
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
        
        seg_out, hv_out = head(d)
        
        seg_out = resize_to_output(seg_out, 512)
        hv_out = resize_to_output(hv_out, 512)
        
        return seg_out, hv_out
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): H&E image (B, 3, H, W)
        
        Returns:
            dict: HE nuclei and cell predictions
        """
        x = resize_to_dinov2(x, 518)
        
        features = self.backbone(x)
        layer_keys = sorted([k for k in features.keys()])
        
        e5 = features[layer_keys[3]]
        e4 = features[layer_keys[2]]
        e3 = features[layer_keys[1]]
        e2 = features[layer_keys[0]]
        
        he_nuclei_seg, he_nuclei_hv = self.decode_branch(e5, e4, e3, e2, 'he_nuclei')
        he_cell_seg, he_cell_hv = self.decode_branch(e5, e4, e3, e2, 'he_cell')
        
        return {
            'he_nuclei_seg': he_nuclei_seg,
            'he_nuclei_hv': he_nuclei_hv,
            'he_cell_seg': he_cell_seg,
            'he_cell_hv': he_cell_hv,
        }


# ============================================================================
# VITAMIN-P MIF BASELINE: MIF only with 2 decoders
# ============================================================================

class VitaminPMIFBaseline(nn.Module):
    """
    Vitamin-P MIF Baseline Model
    
    Architecture:
    - Single encoder (DINOv2, 3ch - MIF 2ch + zero padding)
    - 2 decoder branches:
        * MIF Nuclei
        * MIF Cell
    
    Args:
        model_size (str): DINOv2 variant
        freeze_backbone (bool): Whether to freeze backbone weights
    """
    
    def __init__(self, model_size='base', freeze_backbone=False):
        super().__init__()
        
        self.model_size = model_size
        self.backbone = DINOv2Backbone(model_size=model_size, in_channels=3)
        embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        dec_dims = get_decoder_dims(model_size)
        self.dec_dims = dec_dims
        self.embed_dim = embed_dim
        
        # Projections
        self.proj3 = nn.Conv2d(embed_dim, dec_dims[0], 1)
        self.proj2 = nn.Conv2d(embed_dim, dec_dims[1], 1)
        self.proj1 = nn.Conv2d(embed_dim, dec_dims[2], 1)
        self.proj0 = nn.Conv2d(embed_dim, dec_dims[3], 1)
        
        # MIF NUCLEI decoder
        self.dec1_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_mif_nuclei = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_mif_nuclei = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_mif_nuclei = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.head_mif_nuclei = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        # MIF CELL decoder
        self.dec1_mif_cell = ConvBlock(dec_dims[0] + dec_dims[0], dec_dims[0])
        self.dec2_mif_cell = ConvBlock(dec_dims[0] + dec_dims[1], dec_dims[1])
        self.dec3_mif_cell = ConvBlock(dec_dims[1] + dec_dims[2], dec_dims[2])
        self.dec4_mif_cell = ConvBlock(dec_dims[2] + dec_dims[3], dec_dims[3])
        self.head_mif_cell = SegmentationHead(dec_dims[3], hidden_ch=32)
        
        print(f"✓ VitaminPMIFBaseline initialized with {model_size} backbone")
        print(f"  Architecture: MIF encoder → 2 decoders (Nuclei, Cell)")
    
    def decode_branch(self, e5, e4, e3, e2, branch_name):
        """Decoder branch"""
        dec1 = getattr(self, f'dec1_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
        head = getattr(self, f'head_{branch_name}')
        
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
        
        seg_out, hv_out = head(d)
        
        seg_out = resize_to_output(seg_out, 512)
        hv_out = resize_to_output(hv_out, 512)
        
        return seg_out, hv_out
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): MIF image (B, 3, H, W) - 2ch + zero padding
        
        Returns:
            dict: MIF nuclei and cell predictions
        """
        x = resize_to_dinov2(x, 518)
        
        features = self.backbone(x)
        layer_keys = sorted([k for k in features.keys()])
        
        e5 = features[layer_keys[3]]
        e4 = features[layer_keys[2]]
        e3 = features[layer_keys[1]]
        e2 = features[layer_keys[0]]
        
        mif_nuclei_seg, mif_nuclei_hv = self.decode_branch(e5, e4, e3, e2, 'mif_nuclei')
        mif_cell_seg, mif_cell_hv = self.decode_branch(e5, e4, e3, e2, 'mif_cell')
        
        return {
            'mif_nuclei_seg': mif_nuclei_seg,
            'mif_nuclei_hv': mif_nuclei_hv,
            'mif_cell_seg': mif_cell_seg,
            'mif_cell_hv': mif_cell_hv,
        }
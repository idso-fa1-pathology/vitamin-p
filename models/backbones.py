"""
Unified backbone interface for CNN and Vision Transformer architectures
Supports: ResNet, DINOv2, ConvNeXt
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BackboneBuilder:
    """Factory class to build different backbone architectures"""
    
    SUPPORTED_BACKBONES = {
        # ResNet family
        'resnet34', 'resnet50', 'resnet101',
        # DINOv2 family
        'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
        # ConvNeXt family  
        'convnext_tiny', 'convnext_small', 'convnext_base',
    }
    
    @staticmethod
    def build(backbone_name, pretrained=True, in_channels=3):
        """
        Build a backbone encoder
        
        Args:
            backbone_name: Name of the backbone architecture
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels (3 for HE, 2 for MIF)
            
        Returns:
            backbone: The backbone model with extract_features() method
            channels: List of channel dimensions at each level [c0, c1, c2, c3, c4]
        """
        if backbone_name not in BackboneBuilder.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                f"Supported: {BackboneBuilder.SUPPORTED_BACKBONES}"
            )
        
        # Route to appropriate builder
        if backbone_name.startswith('resnet'):
            return BackboneBuilder._build_resnet(backbone_name, pretrained, in_channels)
        elif backbone_name.startswith('dinov2'):
            return BackboneBuilder._build_dinov2(backbone_name, pretrained, in_channels)
        elif backbone_name.startswith('convnext'):
            return BackboneBuilder._build_convnext(backbone_name, pretrained, in_channels)
        else:
            raise NotImplementedError(f"Builder for {backbone_name} not implemented yet")
    
    @staticmethod
    def _build_resnet(backbone_name, pretrained, in_channels):
        """Build ResNet backbone"""
        if backbone_name == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            channels = [64, 64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            channels = [64, 256, 512, 1024, 2048]
        elif backbone_name == 'resnet101':
            resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown ResNet variant: {backbone_name}")
        
        # Wrap ResNet in our interface
        backbone = ResNetBackbone(resnet, in_channels, pretrained)
        return backbone, channels
    
    @staticmethod
    def _build_dinov2(backbone_name, pretrained, in_channels):
        """Build DINOv2 backbone"""
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for DINOv2. Install: pip install timm")
        
        # Map our names to timm names
        dinov2_map = {
            'dinov2_vits14': ('vit_small_patch14_dinov2.lvd142m', [384, 384, 384, 384, 384]),
            'dinov2_vitb14': ('vit_base_patch14_dinov2.lvd142m', [768, 768, 768, 768, 768]),
            'dinov2_vitl14': ('vit_large_patch14_dinov2.lvd142m', [1024, 1024, 1024, 1024, 1024]),
            'dinov2_vitg14': ('vit_giant_patch14_dinov2.lvd142m', [1536, 1536, 1536, 1536, 1536]),
        }
        
        if backbone_name not in dinov2_map:
            raise ValueError(f"Unknown DINOv2 variant: {backbone_name}")
        
        timm_name, channels = dinov2_map[backbone_name]
        
        # Create DINOv2 model with timm
        vit = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            dynamic_img_size=True  # Allow flexible input sizes
        )
        
        # Wrap ViT in our interface
        from .vit_encoder import ViTBackbone
        backbone = ViTBackbone(vit, in_channels, pretrained, backbone_name)
        return backbone, channels
    
    @staticmethod
    def _build_convnext(backbone_name, pretrained, in_channels):
        """Build ConvNeXt backbone"""
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for ConvNeXt. Install: pip install timm")
        
        # Map our names to timm names and channels
        convnext_map = {
            'convnext_tiny': ('convnext_tiny.fb_in22k_ft_in1k', [96, 192, 384, 768, 768]),
            'convnext_small': ('convnext_small.fb_in22k_ft_in1k', [96, 192, 384, 768, 768]),
            'convnext_base': ('convnext_base.fb_in22k_ft_in1k', [128, 256, 512, 1024, 1024]),
        }
        
        if backbone_name not in convnext_map:
            raise ValueError(f"Unknown ConvNeXt variant: {backbone_name}")
        
        timm_name, channels = convnext_map[backbone_name]
        
        # Create ConvNeXt model with timm
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        
        # Wrap in our interface
        backbone = ConvNeXtBackbone(model, in_channels, pretrained)
        return backbone, channels


class ResNetBackbone(nn.Module):
    """Wrapper for ResNet to provide unified interface"""
    
    def __init__(self, resnet, in_channels, pretrained):
        super().__init__()
        
        # Modify first conv if needed
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                # Initialize with pretrained weights (average RGB channels)
                with torch.no_grad():
                    weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
                    self.conv1.weight[:, :, :, :] = weight.repeat(1, in_channels, 1, 1)
        else:
            self.conv1 = resnet.conv1
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def extract_features(self, x):
        """
        Extract multi-scale features
        
        Returns:
            features: Dict with keys [x0, x1, x2, x3, x4]
        """
        x0 = self.relu(self.bn1(self.conv1(x)))  # /2
        x0_pool = self.maxpool(x0)  # /4
        x1 = self.layer1(x0_pool)  # /4
        x2 = self.layer2(x1)  # /8
        x3 = self.layer3(x2)  # /16
        x4 = self.layer4(x3)  # /32
        
        return {
            'x0': x0,      # 1/2 resolution
            'x1': x1,      # 1/4 resolution
            'x2': x2,      # 1/8 resolution
            'x3': x3,      # 1/16 resolution
            'x4': x4       # 1/32 resolution
        }


class ConvNeXtBackbone(nn.Module):
    """Wrapper for ConvNeXt to provide unified interface"""
    
    def __init__(self, model, in_channels, pretrained):
        super().__init__()
        self.model = model
        
        # Modify first conv if needed
        if in_channels != 3:
            first_conv = self.model.stem[0]
            self.model.stem[0] = nn.Conv2d(
                in_channels, first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            if pretrained:
                with torch.no_grad():
                    weight = first_conv.weight.mean(dim=1, keepdim=True)
                    self.model.stem[0].weight[:, :, :, :] = weight.repeat(1, in_channels, 1, 1)
    
    def extract_features(self, x):
        """Extract multi-scale features"""
        features = self.model(x)  # Returns list of 5 feature maps
        
        return {
            'x0': features[0],  # 1/2 resolution
            'x1': features[1],  # 1/4 resolution
            'x2': features[2],  # 1/8 resolution
            'x3': features[3],  # 1/16 resolution
            'x4': features[4]   # 1/32 resolution
        }
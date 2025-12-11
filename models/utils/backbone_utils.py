"""
Utility functions for backbone management and debugging
"""

import torch
from typing import Dict, List, Tuple


def get_backbone_info(backbone_name: str) -> Dict:
    """
    Get information about a backbone architecture
    
    Args:
        backbone_name: Name of the backbone
        
    Returns:
        info: Dictionary with backbone information
    """
    info = {
        'name': backbone_name,
        'family': None,
        'parameters': None,
        'speed': None,
        'channels': None,
        'recommended_for': None
    }
    
    # ResNet family
    if backbone_name == 'resnet34':
        info.update({
            'family': 'ResNet',
            'parameters': '21.8M',
            'speed': 'Fast',
            'channels': [64, 64, 128, 256, 512],
            'recommended_for': 'Baseline, fast inference'
        })
    elif backbone_name == 'resnet50':
        info.update({
            'family': 'ResNet',
            'parameters': '25.6M',
            'speed': 'Medium',
            'channels': [64, 256, 512, 1024, 2048],
            'recommended_for': 'Good balance of speed and accuracy'
        })
    elif backbone_name == 'resnet101':
        info.update({
            'family': 'ResNet',
            'parameters': '44.5M',
            'speed': 'Medium-Slow',
            'channels': [64, 256, 512, 1024, 2048],
            'recommended_for': 'Higher accuracy, more capacity'
        })
    
    # DINOv2 family
    elif backbone_name == 'dinov2_vits14':
        info.update({
            'family': 'DINOv2-ViT',
            'parameters': '22M',
            'speed': 'Medium-Fast',
            'channels': [384, 384, 384, 384, 384],
            'recommended_for': 'Cell segmentation, fast ViT option'
        })
    elif backbone_name == 'dinov2_vitb14':
        info.update({
            'family': 'DINOv2-ViT',
            'parameters': '86M',
            'speed': 'Medium',
            'channels': [768, 768, 768, 768, 768],
            'recommended_for': 'Cell segmentation, best balance (RECOMMENDED)'
        })
    elif backbone_name == 'dinov2_vitl14':
        info.update({
            'family': 'DINOv2-ViT',
            'parameters': '300M',
            'speed': 'Slow',
            'channels': [1024, 1024, 1024, 1024, 1024],
            'recommended_for': 'Maximum accuracy, research'
        })
    elif backbone_name == 'dinov2_vitg14':
        info.update({
            'family': 'DINOv2-ViT',
            'parameters': '1.1B',
            'speed': 'Very Slow',
            'channels': [1536, 1536, 1536, 1536, 1536],
            'recommended_for': 'Maximum accuracy, requires large GPU'
        })
    
    # ConvNeXt family
    elif backbone_name == 'convnext_tiny':
        info.update({
            'family': 'ConvNeXt',
            'parameters': '28M',
            'speed': 'Fast',
            'channels': [96, 192, 384, 768, 768],
            'recommended_for': 'Modern CNN, good speed'
        })
    elif backbone_name == 'convnext_small':
        info.update({
            'family': 'ConvNeXt',
            'parameters': '50M',
            'speed': 'Medium',
            'channels': [96, 192, 384, 768, 768],
            'recommended_for': 'Modern CNN, balanced'
        })
    elif backbone_name == 'convnext_base':
        info.update({
            'family': 'ConvNeXt',
            'parameters': '89M',
            'speed': 'Medium-Slow',
            'channels': [128, 256, 512, 1024, 1024],
            'recommended_for': 'Modern CNN, high accuracy'
        })
    
    return info


def print_backbone_summary(backbone_name: str):
    """
    Print a formatted summary of backbone information
    
    Args:
        backbone_name: Name of the backbone
    """
    info = get_backbone_info(backbone_name)
    
    print(f"\n{'='*60}")
    print(f"Backbone: {info['name']}")
    print(f"{'='*60}")
    print(f"Family:           {info['family']}")
    print(f"Parameters:       {info['parameters']}")
    print(f"Speed:            {info['speed']}")
    print(f"Channels:         {info['channels']}")
    print(f"Recommended for:  {info['recommended_for']}")
    print(f"{'='*60}\n")


def test_backbone_output(backbone, input_size=(256, 256), batch_size=2, device='cuda'):
    """
    Test backbone feature extraction and print output shapes
    
    Args:
        backbone: Backbone model with extract_features() method
        input_size: (H, W) input image size
        batch_size: Batch size for testing
        device: Device to run test on
        
    Returns:
        features: Dictionary of extracted features
    """
    backbone = backbone.to(device)
    backbone.eval()
    
    # Get input channels from backbone
    if hasattr(backbone, 'conv1'):
        in_channels = backbone.conv1.in_channels
    elif hasattr(backbone, 'model') and hasattr(backbone.model, 'stem'):
        in_channels = backbone.model.stem[0].in_channels
    elif hasattr(backbone, 'vit'):
        in_channels = backbone.vit.patch_embed.proj.in_channels
    else:
        in_channels = 3  # Default
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, input_size[0], input_size[1]).to(device)
    
    print(f"\n{'='*60}")
    print(f"Testing Backbone Feature Extraction")
    print(f"{'='*60}")
    print(f"Input shape: {x.shape}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Extract features
    with torch.no_grad():
        features = backbone.extract_features(x)
    
    # Print feature shapes
    print("Extracted Features:")
    print(f"{'Level':<10} {'Shape':<25} {'Resolution':<15} {'Channels':<10}")
    print(f"{'-'*60}")
    
    for i, (key, feat) in enumerate(features.items()):
        resolution = f"1/{2**(i+1)}"
        channels = feat.shape[1]
        shape_str = f"{tuple(feat.shape)}"
        print(f"{key:<10} {shape_str:<25} {resolution:<15} {channels:<10}")
    
    print(f"{'='*60}\n")
    
    return features


def compare_backbones(backbone_names: List[str], input_size=(256, 256)):
    """
    Compare multiple backbones side by side
    
    Args:
        backbone_names: List of backbone names to compare
        input_size: Input image size for testing
    """
    print(f"\n{'='*80}")
    print(f"Backbone Comparison")
    print(f"{'='*80}\n")
    
    print(f"{'Backbone':<20} {'Family':<15} {'Params':<12} {'Speed':<12} {'Channels':<30}")
    print(f"{'-'*80}")
    
    for name in backbone_names:
        info = get_backbone_info(name)
        channels_str = str(info['channels'][:3]) + '...' if info['channels'] else 'N/A'
        print(f"{name:<20} {info['family']:<15} {info['parameters']:<12} "
              f"{info['speed']:<12} {channels_str:<30}")
    
    print(f"{'='*80}\n")


def get_recommended_backbone(task='cell_segmentation', priority='balanced'):
    """
    Get recommended backbone based on task and priority
    
    Args:
        task: Type of task ('cell_segmentation', 'tissue_segmentation', etc.)
        priority: 'speed', 'accuracy', or 'balanced'
        
    Returns:
        backbone_name: Recommended backbone name
    """
    recommendations = {
        'cell_segmentation': {
            'speed': 'dinov2_vits14',
            'balanced': 'dinov2_vitb14',
            'accuracy': 'dinov2_vitl14'
        },
        'tissue_segmentation': {
            'speed': 'resnet34',
            'balanced': 'resnet50',
            'accuracy': 'convnext_base'
        },
        'general': {
            'speed': 'resnet34',
            'balanced': 'resnet50',
            'accuracy': 'resnet101'
        }
    }
    
    task = task if task in recommendations else 'general'
    recommended = recommendations[task][priority]
    
    print(f"\n{'='*60}")
    print(f"Recommendation for: {task} ({priority} priority)")
    print(f"Recommended backbone: {recommended}")
    print(f"{'='*60}")
    
    print_backbone_summary(recommended)
    
    return recommended


def calculate_model_size(model):
    """
    Calculate model size in terms of parameters and memory
    
    Args:
        model: PyTorch model
        
    Returns:
        info: Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'size_mb': size_all_mb,
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2
    }
    
    return info


def print_model_summary(model, model_name='Model'):
    """
    Print detailed model summary
    
    Args:
        model: PyTorch model
        model_name: Name to display
    """
    info = calculate_model_size(model)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(f"Total parameters:        {info['total_params']:,}")
    print(f"Trainable parameters:    {info['trainable_params']:,}")
    print(f"Non-trainable parameters: {info['non_trainable_params']:,}")
    print(f"Model size:              {info['size_mb']:.2f} MB")
    print(f"  - Parameters:          {info['param_size_mb']:.2f} MB")
    print(f"  - Buffers:             {info['buffer_size_mb']:.2f} MB")
    print(f"{'='*60}\n")
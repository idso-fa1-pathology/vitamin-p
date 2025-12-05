# utils.py
# Utility functions for post-processing visualization and analysis

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Optional
import cv2


def visualize_instance_map(
    instance_map: np.ndarray,
    image: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    title: str = "Instance Segmentation"
) -> plt.Figure:
    """Visualize instance segmentation map with random colors
    
    Args:
        instance_map (np.ndarray): Instance map with unique IDs
        image (np.ndarray, optional): Original image to overlay on
        alpha (float): Transparency for overlay
        title (str): Plot title
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Generate random colors for each instance
    num_instances = len(np.unique(instance_map)) - 1  # Exclude background
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, max(num_instances, 20)))
    
    # Create colored instance map
    colored_map = np.zeros((*instance_map.shape, 3))
    for idx, inst_id in enumerate(np.unique(instance_map)[1:]):  # Skip background
        mask = instance_map == inst_id
        colored_map[mask] = colors[idx % len(colors)][:3]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    if image is not None:
        ax.imshow(image)
        ax.imshow(colored_map, alpha=alpha)
    else:
        ax.imshow(colored_map)
    
    ax.set_title(f"{title} ({num_instances} instances)")
    ax.axis('off')
    
    return fig


def draw_instance_contours(
    image: np.ndarray,
    inst_info_dict: dict,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    draw_centroids: bool = True,
    centroid_color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """Draw instance contours and centroids on image
    
    Args:
        image (np.ndarray): Image to draw on
        inst_info_dict (dict): Instance information dictionary
        color (Tuple[int, int, int]): Contour color (BGR)
        thickness (int): Contour thickness
        draw_centroids (bool): Whether to draw centroids
        centroid_color (Tuple[int, int, int]): Centroid color (BGR)
        
    Returns:
        np.ndarray: Image with drawn contours
    """
    # Ensure image is in BGR format for OpenCV
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3 and image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    
    output = image.copy()
    
    for inst_id, inst_data in inst_info_dict.items():
        # Draw contour
        contour = inst_data['contour']
        if contour.shape[0] >= 3:
            cv2.drawContours(output, [contour], -1, color, thickness)
        
        # Draw centroid
        if draw_centroids:
            centroid = inst_data['centroid'].astype(int)
            cv2.circle(output, tuple(centroid), 3, centroid_color, -1)
    
    return output


def compare_before_after_postprocessing(
    image: np.ndarray,
    binary_pred: np.ndarray,
    instance_map: np.ndarray,
    inst_info_dict: dict,
    title_prefix: str = ""
) -> plt.Figure:
    """Compare binary prediction vs instance segmentation
    
    Args:
        image (np.ndarray): Original image
        binary_pred (np.ndarray): Binary prediction before post-processing
        instance_map (np.ndarray): Instance map after post-processing
        inst_info_dict (dict): Instance information
        title_prefix (str): Prefix for titles (e.g., "HE Nuclei")
        
    Returns:
        plt.Figure: Comparison figure
    """
    num_instances = len(inst_info_dict)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'{title_prefix} - Original Image')
    axes[0].axis('off')
    
    # Binary prediction
    axes[1].imshow(image)
    axes[1].imshow(binary_pred > 0.5, alpha=0.5, cmap='Reds')
    axes[1].set_title(f'{title_prefix} - Binary Prediction')
    axes[1].axis('off')
    
    # Instance segmentation
    colored_map = np.zeros((*instance_map.shape, 3))
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, max(num_instances, 20)))
    
    for idx, inst_id in enumerate(np.unique(instance_map)[1:]):
        mask = instance_map == inst_id
        colored_map[mask] = colors[idx % len(colors)][:3]
    
    axes[2].imshow(image)
    axes[2].imshow(colored_map, alpha=0.5)
    axes[2].set_title(f'{title_prefix} - Instances ({num_instances})')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def get_instance_statistics(inst_info_dict: dict) -> dict:
    """Calculate statistics from instance information
    
    Args:
        inst_info_dict (dict): Instance information dictionary
        
    Returns:
        dict: Statistics including count, mean area, etc.
    """
    if len(inst_info_dict) == 0:
        return {
            'count': 0,
            'mean_area': 0,
            'std_area': 0,
            'min_area': 0,
            'max_area': 0
        }
    
    areas = []
    for inst_id, inst_data in inst_info_dict.items():
        bbox = inst_data['bbox']
        height = bbox[1, 0] - bbox[0, 0]
        width = bbox[1, 1] - bbox[0, 1]
        areas.append(height * width)
    
    areas = np.array(areas)
    
    return {
        'count': len(inst_info_dict),
        'mean_area': float(np.mean(areas)),
        'std_area': float(np.std(areas)),
        'min_area': float(np.min(areas)),
        'max_area': float(np.max(areas))
    }


def overlay_instances_on_image(
    image: np.ndarray,
    instance_map: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'tab20'
) -> np.ndarray:
    """Create overlay of instances on original image
    
    Args:
        image (np.ndarray): Original image (H, W, C)
        instance_map (np.ndarray): Instance map (H, W)
        alpha (float): Transparency
        colormap (str): Matplotlib colormap name
        
    Returns:
        np.ndarray: Image with overlay
    """
    # Normalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Generate random colors
    num_instances = len(np.unique(instance_map)) - 1
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, max(num_instances, 20)))
    
    # Create colored instance map
    colored_map = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    for idx, inst_id in enumerate(np.unique(instance_map)[1:]):
        mask = instance_map == inst_id
        color = (colors[idx % len(colors)][:3] * 255).astype(np.uint8)
        colored_map[mask] = color
    
    # Blend images
    overlay = cv2.addWeighted(image, 1 - alpha, colored_map, alpha, 0)
    
    return overlay
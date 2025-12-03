"""
HV Map Generator for CRC Dataset
Generates horizontal-vertical gradient maps for instance segmentation
Uses PanNuke's superior method for better quality
"""

import numpy as np
from scipy.ndimage import center_of_mass
from typing import Tuple


def get_bounding_box(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding box coordinates for a binary mask
    
    Args:
        binary_mask: Binary mask (H, W)
    
    Returns:
        (y_min, y_max, x_min, x_max) or None if mask is empty
    """
    coords = np.where(binary_mask)
    if len(coords[0]) == 0:
        return None
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return y_min, y_max + 1, x_min, x_max + 1


def generate_hv_map_pannuke(inst_map: np.ndarray) -> np.ndarray:
    """
    Generate HV map using PanNuke's superior method
    
    This method provides better quality than simple centroid-based approaches:
    - Uses center of mass (more accurate than mean)
    - Separate normalization for positive/negative directions
    - Bounding box optimization for speed
    
    Args:
        inst_map: Instance map where each instance has unique integer ID (H, W)
                 Background should be 0
    
    Returns:
        HV map with shape (2, H, W)
        - Channel 0: Horizontal gradient (-1 to 1)
        - Channel 1: Vertical gradient (-1 to 1)
    """
    orig_inst_map = inst_map.copy()
    h, w = orig_inst_map.shape
    
    # Initialize HV maps
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)
    
    # Get list of instance IDs (excluding background=0)
    inst_list = list(np.unique(orig_inst_map))
    if 0 in inst_list:
        inst_list.remove(0)
    
    # Process each instance
    for inst_id in inst_list:
        # Get instance mask
        inst_mask = (orig_inst_map == inst_id).astype(np.uint8)
        
        # Get bounding box
        bbox = get_bounding_box(inst_mask)
        if bbox is None:
            continue
        
        y_min, y_max, x_min, x_max = bbox
        
        # Expand bounding box by 2 pixels (with bounds checking)
        y_min = max(0, y_min - 2)
        x_min = max(0, x_min - 2)
        y_max = min(h, y_max + 2)
        x_max = min(w, x_max + 2)
        
        # Extract instance region
        inst_crop = inst_mask[y_min:y_max, x_min:x_max]
        
        # Skip if too small
        if inst_crop.shape[0] < 2 or inst_crop.shape[1] < 2:
            continue
        
        # Calculate center of mass (more precise than centroid)
        com = center_of_mass(inst_crop)
        com_y = int(com[0] + 0.5)  # Round to nearest pixel
        com_x = int(com[1] + 0.5)
        
        # Create coordinate grids (1-indexed)
        x_range = np.arange(1, inst_crop.shape[1] + 1)
        y_range = np.arange(1, inst_crop.shape[0] + 1)
        
        # Shift to center of mass
        x_range = x_range - com_x
        y_range = y_range - com_y
        
        # Create 2D grids
        inst_x, inst_y = np.meshgrid(x_range, y_range)
        
        # Remove coordinates outside instance
        inst_x[inst_crop == 0] = 0
        inst_y[inst_crop == 0] = 0
        inst_x = inst_x.astype(np.float32)
        inst_y = inst_y.astype(np.float32)
        
        # PanNuke's SUPERIOR normalization:
        # Normalize negative and positive directions separately
        
        # Horizontal (X) normalization
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        
        # Vertical (Y) normalization
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])
        
        # Assign to full maps
        x_map[y_min:y_max, x_min:x_max][inst_crop > 0] = inst_x[inst_crop > 0]
        y_map[y_min:y_max, x_min:x_max][inst_crop > 0] = inst_y[inst_crop > 0]
    
    # Stack into (2, H, W) format
    hv_map = np.stack([x_map, y_map], axis=0)
    
    return hv_map


def generate_hv_map_simple(inst_map: np.ndarray) -> np.ndarray:
    """
    Generate HV map using simple centroid-based method (faster but lower quality)
    
    Args:
        inst_map: Instance map where each instance has unique integer ID (H, W)
    
    Returns:
        HV map with shape (2, H, W)
    """
    h, w = inst_map.shape
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)
    
    # Get instance IDs
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    for inst_id in inst_list:
        inst_mask = (inst_map == inst_id)
        
        if not inst_mask.any():
            continue
        
        # Calculate centroid
        y_centroid = y_coords[inst_mask].mean()
        x_centroid = x_coords[inst_mask].mean()
        
        # Calculate distances
        y_dist = y_coords[inst_mask] - y_centroid
        x_dist = x_coords[inst_mask] - x_centroid
        
        # Normalize
        max_dist = max(np.abs(y_dist).max(), np.abs(x_dist).max())
        if max_dist > 0:
            y_dist = y_dist / max_dist
            x_dist = x_dist / max_dist
        
        # Assign to maps
        y_map[inst_mask] = y_dist
        x_map[inst_mask] = x_dist
    
    hv_map = np.stack([x_map, y_map], axis=0)
    return hv_map


def batch_generate_hv_maps(masks_batch: np.ndarray, method: str = 'pannuke') -> np.ndarray:
    """
    Generate HV maps for a batch of instance masks
    
    Args:
        masks_batch: Batch of instance masks (B, H, W)
        method: 'pannuke' (recommended) or 'simple'
    
    Returns:
        Batch of HV maps (B, 2, H, W)
    """
    batch_size, h, w = masks_batch.shape
    hv_maps = np.zeros((batch_size, 2, h, w), dtype=np.float32)
    
    # Select generator function
    if method == 'pannuke':
        generator_fn = generate_hv_map_pannuke
    elif method == 'simple':
        generator_fn = generate_hv_map_simple
    else:
        raise ValueError(f"Unknown HV method: {method}. Use 'pannuke' or 'simple'")
    
    # Generate for each mask in batch
    for i in range(batch_size):
        hv_maps[i] = generator_fn(masks_batch[i])
    
    return hv_maps


def visualize_hv_map(hv_map: np.ndarray, save_path: str = None):
    """
    Visualize HV map for debugging (optional utility)
    
    Args:
        hv_map: HV map (2, H, W)
        save_path: Optional path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Horizontal gradient
        im0 = axes[0].imshow(hv_map[0], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title('Horizontal Gradient')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0])
        
        # Vertical gradient
        im1 = axes[1].imshow(hv_map[1], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Vertical Gradient')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Magnitude
        magnitude = np.sqrt(hv_map[0]**2 + hv_map[1]**2)
        im2 = axes[2].imshow(magnitude, cmap='viridis')
        axes[2].set_title('Gradient Magnitude')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("⚠️  matplotlib not available for visualization")


# Quick test
if __name__ == "__main__":
    # Create a simple test instance map
    test_map = np.zeros((100, 100), dtype=np.int32)
    
    # Add some circular instances
    from skimage.draw import disk
    
    # Instance 1
    rr, cc = disk((30, 30), 15)
    test_map[rr, cc] = 1
    
    # Instance 2
    rr, cc = disk((70, 70), 20)
    test_map[rr, cc] = 2
    
    # Generate HV map
    hv_map = generate_hv_map_pannuke(test_map)
    
    print(f"✅ HV map generated: {hv_map.shape}")
    print(f"   Horizontal range: [{hv_map[0].min():.3f}, {hv_map[0].max():.3f}]")
    print(f"   Vertical range: [{hv_map[1].min():.3f}, {hv_map[1].max():.3f}]")
    
    # Visualize if matplotlib available
    visualize_hv_map(hv_map, save_path="test_hv_map.png")
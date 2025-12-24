# utils.py
# Utility functions for post-processing visualization and analysis

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Optional
import cv2

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
from pathlib import Path
from glob import glob

# Configuration
base_dir = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/yasin-vitaminp/Xenium_public/5k'
output_dir = Path('./wsi_patches_output')
output_dir.mkdir(parents=True, exist_ok=True)

chunk_size = 1024
num_patches_per_tissue = 3  # Patches per tissue type
min_cell_count = 300  # Minimum cells required
max_attempts = 50  # Maximum attempts per patch to avoid infinite loops

# Find all OME-TIFF files
tissue_types = ['breast', 'cervical', 'lung', 'lymph_node', 'ovarian', 'prostate', 'skin']
wsi_files = []

for tissue in tissue_types:
    pattern = f"{base_dir}/{tissue}/*_registered.ome.tif"
    files = glob(pattern)
    if files:
        wsi_files.append((tissue, files[0]))
        print(f"Found {tissue}: {Path(files[0]).name}")

print(f"\n{'='*60}")
print(f"Found {len(wsi_files)} tissue types to process")
print('='*60)

# Process each tissue type
for tissue_name, wsi_path in wsi_files:
    print(f"\n{'='*60}")
    print(f"PROCESSING: {tissue_name.upper()}")
    print('='*60)
    
    # Create tissue-specific output directory
    tissue_output_dir = output_dir / tissue_name
    tissue_output_dir.mkdir(exist_ok=True)
    
    # Load WSI metadata
    with tifffile.TiffFile(wsi_path) as tif:
        page = tif.pages[0]
        wsi_shape = page.shape
        print(f"WSI shape: {wsi_shape}")
        
        max_y = wsi_shape[0] - chunk_size
        max_x = wsi_shape[1] - chunk_size
        np.random.seed(hash(tissue_name) % (2**32))  # Reproducible per tissue
    
    # Process patches - keep trying until we get num_patches_per_tissue good ones
    successful_patches = 0
    attempt = 0
    
    while successful_patches < num_patches_per_tissue and attempt < max_attempts * num_patches_per_tissue:
        attempt += 1
        
        # Generate random coordinates
        start_y = np.random.randint(0, max_y)
        start_x = np.random.randint(0, max_x)
        
        print(f"\n  Attempt {attempt} (Patch {successful_patches+1}/{num_patches_per_tissue}) at ({start_y}, {start_x})")
        
        # Load patch
        with tifffile.TiffFile(wsi_path) as tif:
            wsi_array = tif.pages[0].asarray()[start_y:start_y+chunk_size, start_x:start_x+chunk_size]
            
            # Ensure RGB uint8
            if wsi_array.ndim == 2:
                wsi_array = np.stack([wsi_array] * 3, axis=-1)
            elif wsi_array.shape[-1] > 3:
                wsi_array = wsi_array[..., :3]
            if wsi_array.dtype != np.uint8:
                wsi_array = (wsi_array * 255).astype(np.uint8) if wsi_array.max() <= 1 else wsi_array.astype(np.uint8)
        
        # Run inference
        results_nuclei = wsi_inference.process_wsi(wsi_array, f"{tissue_name}_patch_{successful_patches+1}_nuclei", "he", "nuclei")
        results_cells = wsi_inference.process_wsi(wsi_array, f"{tissue_name}_patch_{successful_patches+1}_cells", "he", "cell")
        
        num_cells = results_cells['num_cells']
        num_nuclei = results_nuclei['num_cells']
        
        print(f"  Nuclei: {num_nuclei}, Cells: {num_cells}")
        
        # Check if this patch meets our criteria
        if num_cells < min_cell_count:
            print(f"  ❌ Skipping: Only {num_cells} cells (need ≥{min_cell_count})")
            continue
        
        # This patch is good! Save it
        successful_patches += 1
        print(f"  ✅ Accepted: {num_cells} cells (≥{min_cell_count})")
        
        # Create overlay visualization
        vis = wsi_array.copy()
        
        # Draw cells (blue)
        for cell in results_cells['cells']:
            contour = np.array(cell['contour'], dtype=np.int32)
            if contour.shape[0] >= 3:
                cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2)
        
        # Draw nuclei (green)
        for cell in results_nuclei['cells']:
            contour = np.array(cell['contour'], dtype=np.int32)
            if contour.shape[0] >= 3:
                cv2.drawContours(vis, [contour], -1, (0, 255, 0), 1)
            cv2.circle(vis, centroid, 2, (255, 0, 0), -1)
        
        # Save figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(wsi_array)
        axes[0].set_title(f'{tissue_name.title()} - Patch {successful_patches}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(vis)
        axes[1].set_title(f'Cells: {num_cells} (Blue) | Nuclei: {num_nuclei} (Green)', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        save_path = tissue_output_dir / f'patch_{successful_patches:02d}_y{start_y}_x{start_x}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {save_path}")
    
    if successful_patches < num_patches_per_tissue:
        print(f"\n  ⚠️  WARNING: Only found {successful_patches}/{num_patches_per_tissue} patches with ≥{min_cell_count} cells after {attempt} attempts")

print(f"\n{'='*60}")
print(f"✅ ALL PROCESSING COMPLETE!")
print(f"Results saved to: {output_dir}")
print('='*60)
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
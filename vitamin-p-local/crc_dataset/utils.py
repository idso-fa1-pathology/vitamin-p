"""
Utility functions for CRC Dataset
Helper functions for validation, visualization, and debugging
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import psutil


def check_memory_available(required_gb: float = 10.0) -> bool:
    """
    Check if enough memory is available
    
    Args:
        required_gb: Required memory in GB
    
    Returns:
        True if enough memory available
    """
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    if available_gb < required_gb:
        print(f"⚠️  Low memory warning:")
        print(f"   Required: {required_gb:.2f} GB")
        print(f"   Available: {available_gb:.2f} GB")
        return False
    
    return True


def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    rss_gb = memory_info.rss / (1024 ** 3)
    vms_gb = memory_info.vms / (1024 ** 3)
    
    print(f"💾 Memory Usage:")
    print(f"   RSS (Resident Set Size): {rss_gb:.2f} GB")
    print(f"   VMS (Virtual Memory Size): {vms_gb:.2f} GB")
    
    # System memory
    vm = psutil.virtual_memory()
    print(f"   System Total: {vm.total / (1024**3):.2f} GB")
    print(f"   System Available: {vm.available / (1024**3):.2f} GB")
    print(f"   System Used: {vm.percent:.1f}%")


def validate_zarr_structure(zarr_base: str, samples: List[str], verbose: bool = True):
    """
    Validate that all samples have correct Zarr structure
    
    Args:
        zarr_base: Base directory for Zarr data
        samples: List of sample names to validate
        verbose: Print detailed information
    
    Returns:
        (valid_samples, invalid_samples)
    """
    import zarr
    
    valid_samples = []
    invalid_samples = []
    
    required_files = {
        'he': ['images.zarr', 'nuclei_masks.zarr', 'cell_masks.zarr'],
        'mif': ['images.zarr', 'nuclei_masks.zarr', 'cell_masks.zarr']
    }
    
    if verbose:
        print(f"🔍 Validating {len(samples)} samples...")
    
    for sample in samples:
        sample_valid = True
        errors = []
        
        # Check each modality
        for modality, files in required_files.items():
            for filename in files:
                filepath = f"{zarr_base}/{sample}/{modality}/{filename}"
                
                if not os.path.exists(filepath):
                    sample_valid = False
                    errors.append(f"Missing: {modality}/{filename}")
                else:
                    # Try to open to verify it's valid zarr
                    try:
                        z = zarr.open(filepath, 'r')
                        # Check shape is reasonable
                        if len(z.shape) < 2:
                            sample_valid = False
                            errors.append(f"Invalid shape: {modality}/{filename}")
                    except Exception as e:
                        sample_valid = False
                        errors.append(f"Cannot open: {modality}/{filename} - {e}")
        
        if sample_valid:
            valid_samples.append(sample)
            if verbose:
                print(f"  ✅ {sample}")
        else:
            invalid_samples.append(sample)
            if verbose:
                print(f"  ❌ {sample}")
                for error in errors:
                    print(f"     {error}")
    
    if verbose:
        print(f"\n📊 Validation Summary:")
        print(f"   Valid: {len(valid_samples)}/{len(samples)}")
        print(f"   Invalid: {len(invalid_samples)}/{len(samples)}")
    
    return valid_samples, invalid_samples


def print_batch_statistics(batch: Dict[str, torch.Tensor]):
    """
    Print statistics about a batch
    
    Args:
        batch: Batch dictionary from dataloader
    """
    print("="*80)
    print("BATCH STATISTICS")
    print("="*80)
    
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            print(f"\n{key}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
            print(f"  Mean: {tensor.float().mean().item():.4f}")
            print(f"  Std: {tensor.float().std().item():.4f}")
            
            # For masks, print unique values
            if 'mask' in key and len(tensor.shape) == 3:  # (B, H, W)
                unique_vals = torch.unique(tensor)
                print(f"  Unique values: {unique_vals.tolist()}")
    
    print("="*80)


def visualize_batch(batch: Dict[str, torch.Tensor], 
                   num_samples: int = 4,
                   save_path: Optional[str] = None):
    """
    Visualize a batch of data
    
    Args:
        batch: Batch dictionary from dataloader
        num_samples: Number of samples to visualize
        save_path: Optional path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("⚠️  matplotlib not available for visualization")
        return
    
    batch_size = batch['image'].shape[0]
    num_samples = min(num_samples, batch_size)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5 * num_samples))
    gs = GridSpec(num_samples, 10, figure=fig, hspace=0.3, wspace=0.3)
    
    for i in range(num_samples):
        # H&E image (first 3 channels)
        ax = fig.add_subplot(gs[i, 0])
        he_img = batch['he_image'][i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(np.clip(he_img, 0, 1))
        ax.set_title(f"Sample {i}: H&E")
        ax.axis('off')
        
        # MIF image (next 2 channels as RGB - duplicate first channel)
        ax = fig.add_subplot(gs[i, 1])
        mif_img = batch['mif_image'][i].cpu().numpy()
        # Convert 2-channel to RGB for visualization
        mif_rgb = np.zeros((mif_img.shape[1], mif_img.shape[2], 3))
        mif_rgb[:, :, 0] = mif_img[0]  # First channel -> Red
        mif_rgb[:, :, 1] = mif_img[1]  # Second channel -> Green
        ax.imshow(np.clip(mif_rgb, 0, 1))
        ax.set_title(f"Sample {i}: MIF")
        ax.axis('off')
        
        # HE nuclei mask
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(batch['he_nuclei_mask'][i].cpu().numpy(), cmap='gray')
        ax.set_title("HE Nuclei Mask")
        ax.axis('off')
        
        # HE cell mask
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(batch['he_cell_mask'][i].cpu().numpy(), cmap='gray')
        ax.set_title("HE Cell Mask")
        ax.axis('off')
        
        # MIF nuclei mask
        ax = fig.add_subplot(gs[i, 4])
        ax.imshow(batch['mif_nuclei_mask'][i].cpu().numpy(), cmap='gray')
        ax.set_title("MIF Nuclei Mask")
        ax.axis('off')
        
        # MIF cell mask
        ax = fig.add_subplot(gs[i, 5])
        ax.imshow(batch['mif_cell_mask'][i].cpu().numpy(), cmap='gray')
        ax.set_title("MIF Cell Mask")
        ax.axis('off')
        
        # HE nuclei HV (horizontal)
        ax = fig.add_subplot(gs[i, 6])
        ax.imshow(batch['he_nuclei_hv'][i, 0].cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title("HE Nuclei HV-H")
        ax.axis('off')
        
        # HE nuclei HV (vertical)
        ax = fig.add_subplot(gs[i, 7])
        ax.imshow(batch['he_nuclei_hv'][i, 1].cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title("HE Nuclei HV-V")
        ax.axis('off')
        
        # HE cell HV magnitude
        ax = fig.add_subplot(gs[i, 8])
        hv_mag = torch.sqrt(batch['he_cell_hv'][i, 0]**2 + batch['he_cell_hv'][i, 1]**2)
        ax.imshow(hv_mag.cpu().numpy(), cmap='viridis')
        ax.set_title("HE Cell HV Mag")
        ax.axis('off')
        
        # MIF nuclei HV magnitude
        ax = fig.add_subplot(gs[i, 9])
        hv_mag = torch.sqrt(batch['mif_nuclei_hv'][i, 0]**2 + batch['mif_nuclei_hv'][i, 1]**2)
        ax.imshow(hv_mag.cpu().numpy(), cmap='viridis')
        ax.set_title("MIF Nuclei HV Mag")
        ax.axis('off')
    
    plt.suptitle("Batch Visualization: Images, Masks, and HV Maps", fontsize=16, y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def count_instances_in_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
    """
    Count number of instances in each mask type for each sample in batch
    
    Args:
        batch: Batch dictionary
    
    Returns:
        Dictionary with instance counts per mask type
    """
    counts = {
        'he_nuclei': [],
        'he_cells': [],
        'mif_nuclei': [],
        'mif_cells': []
    }
    
    batch_size = batch['he_nuclei_instance'].shape[0]
    
    for i in range(batch_size):
        # Count unique non-zero values (each unique value is an instance)
        he_nuclei_instances = len(torch.unique(batch['he_nuclei_instance'][i])) - 1  # -1 for background
        he_cell_instances = len(torch.unique(batch['he_cell_instance'][i])) - 1
        mif_nuclei_instances = len(torch.unique(batch['mif_nuclei_instance'][i])) - 1
        mif_cell_instances = len(torch.unique(batch['mif_cell_instance'][i])) - 1
        
        counts['he_nuclei'].append(max(0, he_nuclei_instances))
        counts['he_cells'].append(max(0, he_cell_instances))
        counts['mif_nuclei'].append(max(0, mif_nuclei_instances))
        counts['mif_cells'].append(max(0, mif_cell_instances))
    
    return counts


def test_dataloader_speed(dataloader, num_batches: int = 10):
    """
    Test dataloader speed
    
    Args:
        dataloader: DataLoader to test
        num_batches: Number of batches to iterate
    """
    import time
    
    print(f"⏱️  Testing dataloader speed ({num_batches} batches)...")
    
    times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        start = time.time()
        # Simulate some processing
        _ = batch['image'].mean()
        end = time.time()
        
        times.append(end - start)
        
        if i == 0:
            print(f"   First batch: {times[0]:.3f}s (includes initialization)")
    
    if len(times) > 1:
        avg_time = np.mean(times[1:])  # Exclude first batch
        print(f"   Average time per batch: {avg_time:.3f}s")
        print(f"   Throughput: {1/avg_time:.1f} batches/sec")


def get_dataset_statistics(dataloader, max_batches: Optional[int] = None):
    """
    Compute statistics over entire dataset
    
    Args:
        dataloader: DataLoader to analyze
        max_batches: Maximum number of batches to process (None = all)
    
    Returns:
        Dictionary with statistics
    """
    print("📊 Computing dataset statistics...")
    
    stats = {
        'num_batches': 0,
        'num_samples': 0,
        'he_mean': [],
        'he_std': [],
        'mif_mean': [],
        'mif_std': [],
        'instance_counts': {
            'he_nuclei': [],
            'he_cells': [],
            'mif_nuclei': [],
            'mif_cells': []
        }
    }
    
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        
        stats['num_batches'] += 1
        stats['num_samples'] += batch['image'].shape[0]
        
        # Image statistics
        stats['he_mean'].append(batch['he_image'].mean(dim=(0, 2, 3)))
        stats['he_std'].append(batch['he_image'].std(dim=(0, 2, 3)))
        stats['mif_mean'].append(batch['mif_image'].mean(dim=(0, 2, 3)))
        stats['mif_std'].append(batch['mif_image'].std(dim=(0, 2, 3)))
        
        # Instance counts
        counts = count_instances_in_batch(batch)
        for key in counts:
            stats['instance_counts'][key].extend(counts[key])
    
    # Aggregate statistics
    stats['he_mean'] = torch.stack(stats['he_mean']).mean(dim=0).tolist()
    stats['he_std'] = torch.stack(stats['he_std']).mean(dim=0).tolist()
    stats['mif_mean'] = torch.stack(stats['mif_mean']).mean(dim=0).tolist()
    stats['mif_std'] = torch.stack(stats['mif_std']).mean(dim=0).tolist()
    
    # Instance count statistics
    for key in stats['instance_counts']:
        counts = stats['instance_counts'][key]
        stats['instance_counts'][key] = {
            'mean': np.mean(counts),
            'std': np.std(counts),
            'min': np.min(counts),
            'max': np.max(counts),
            'median': np.median(counts)
        }
    
    print(f"✅ Processed {stats['num_samples']} samples in {stats['num_batches']} batches")
    
    return stats


def print_statistics(stats: Dict):
    """Print dataset statistics in readable format"""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\n📊 Total Samples: {stats['num_samples']}")
    print(f"📦 Total Batches: {stats['num_batches']}")
    
    print(f"\n🎨 H&E Image Statistics:")
    print(f"   Mean (per channel): {[f'{x:.4f}' for x in stats['he_mean']]}")
    print(f"   Std (per channel):  {[f'{x:.4f}' for x in stats['he_std']]}")
    
    print(f"\n🔬 MIF Image Statistics:")
    print(f"   Mean (per channel): {[f'{x:.4f}' for x in stats['mif_mean']]}")
    print(f"   Std (per channel):  {[f'{x:.4f}' for x in stats['mif_std']]}")
    
    print(f"\n🎯 Instance Counts:")
    for mask_type, counts in stats['instance_counts'].items():
        print(f"   {mask_type}:")
        print(f"      Mean: {counts['mean']:.1f}")
        print(f"      Std:  {counts['std']:.1f}")
        print(f"      Range: [{counts['min']:.0f}, {counts['max']:.0f}]")
        print(f"      Median: {counts['median']:.1f}")
    
    print("="*80)
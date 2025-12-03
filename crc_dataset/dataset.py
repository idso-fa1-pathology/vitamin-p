"""
CRC Dataset - Main dataset class for loading Zarr data
Handles multi-modal imaging (H&E + MIF) with instance segmentation
"""

import os
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import psutil

from .hv_generator import batch_generate_hv_maps
from .augmentation import MedicalImageAugmentation


class CRCZarrDataset(Dataset):
    """
    CRC Dataset loader for Zarr format data
    
    Features:
    - Loads H&E (3 channels) + MIF (2 channels) images
    - Handles 4 mask types: HE nuclei, HE cells, MIF nuclei, MIF cells
    - Generates 4 HV maps using PanNuke method
    - Optional memory caching for speed
    - Data augmentation support
    """
    
    def __init__(self, samples: List[str], config, training: bool = True):
        """
        Initialize dataset
        
        Args:
            samples: List of sample names (e.g., ['CRC01', 'CRC02'])
            config: Config object with all settings
            training: Whether this is training set (enables augmentation)
        """
        self.samples = samples
        self.config = config
        self.training = training
        self.zarr_base = config.zarr_base
        
        # Build patch index (sample_name, patch_idx)
        self.patch_info = []
        self._build_patch_index()
        
        # Initialize augmentation
        if training and config.train_augment:
            aug_config = config.get_augmentation_config()
            self.augment = MedicalImageAugmentation(aug_config)
        else:
            self.augment = None
        
        if config.verbose:
            print(f"{'Training' if training else 'Validation/Test'} Dataset: "
                  f"{len(self.samples)} samples, {len(self.patch_info)} patches")
    
    def _build_patch_index(self):
        """Build index of all patches across samples"""
        if self.config.verbose:
            print(f"📊 Building patch index for {len(self.samples)} samples...")
        
        for sample in self.samples:
            he_path = f"{self.zarr_base}/{sample}/he/images.zarr"
            
            try:
                # Open zarr to get number of patches
                z = zarr.open(he_path, 'r')
                n_patches = z.shape[0]
                
                # Add all patches from this sample
                for patch_idx in range(n_patches):
                    self.patch_info.append((sample, patch_idx))
                
                if self.config.verbose:
                    print(f"  ✅ {sample}: {n_patches} patches")
                    
            except Exception as e:
                print(f"  ⚠️  {sample}: Could not load - {e}")
                continue
    
    def __len__(self):
        return len(self.patch_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patch with all modalities and masks
        
        Returns:
            Dictionary containing:
                'image': (5, H, W) - Combined H&E + MIF
                'he_image': (3, H, W) - H&E only
                'mif_image': (2, H, W) - MIF only
                'he_nuclei_mask': (H, W) - Binary mask
                'he_cell_mask': (H, W) - Binary mask
                'mif_nuclei_mask': (H, W) - Binary mask
                'mif_cell_mask': (H, W) - Binary mask
                'he_nuclei_hv': (2, H, W) - HV map
                'he_cell_hv': (2, H, W) - HV map
                'mif_nuclei_hv': (2, H, W) - HV map
                'mif_cell_hv': (2, H, W) - HV map
        """
        sample_name, patch_idx = self.patch_info[idx]
        
        # Load H&E data (3 channels)
        he_img = zarr.open(f"{self.zarr_base}/{sample_name}/he/images.zarr", 'r')[patch_idx]
        he_nuclei_mask = zarr.open(f"{self.zarr_base}/{sample_name}/he/nuclei_masks.zarr", 'r')[patch_idx]
        he_cell_mask = zarr.open(f"{self.zarr_base}/{sample_name}/he/cell_masks.zarr", 'r')[patch_idx]
        
        # Load MIF data (2 channels)
        mif_img = zarr.open(f"{self.zarr_base}/{sample_name}/mif/images.zarr", 'r')[patch_idx]
        mif_nuclei_mask = zarr.open(f"{self.zarr_base}/{sample_name}/mif/nuclei_masks.zarr", 'r')[patch_idx]
        mif_cell_mask = zarr.open(f"{self.zarr_base}/{sample_name}/mif/cell_masks.zarr", 'r')[patch_idx]
        
        # Convert to float32 and normalize
        he_img = he_img.astype(np.float32) / self.config.he_max_value
        mif_img = (mif_img.astype(np.float32) - mif_img.min()) / (mif_img.max() - mif_img.min() + 1e-8)
        
        # Convert masks to proper format
        he_nuclei_mask = he_nuclei_mask.astype(np.int64)
        he_cell_mask = he_cell_mask.astype(np.int64)
        mif_nuclei_mask = mif_nuclei_mask.astype(np.int64)
        mif_cell_mask = mif_cell_mask.astype(np.int64)
        
        # Generate HV maps if enabled
        if self.config.generate_hv_maps:
            he_nuclei_hv = batch_generate_hv_maps(
                he_nuclei_mask[np.newaxis, ...], 
                method=self.config.hv_method
            )[0]  # (2, H, W)
            
            he_cell_hv = batch_generate_hv_maps(
                he_cell_mask[np.newaxis, ...],
                method=self.config.hv_method
            )[0]
            
            mif_nuclei_hv = batch_generate_hv_maps(
                mif_nuclei_mask[np.newaxis, ...],
                method=self.config.hv_method
            )[0]
            
            mif_cell_hv = batch_generate_hv_maps(
                mif_cell_mask[np.newaxis, ...],
                method=self.config.hv_method
            )[0]
        else:
            # Create dummy HV maps
            h, w = he_img.shape[:2]
            he_nuclei_hv = np.zeros((2, h, w), dtype=np.float32)
            he_cell_hv = np.zeros((2, h, w), dtype=np.float32)
            mif_nuclei_hv = np.zeros((2, h, w), dtype=np.float32)
            mif_cell_hv = np.zeros((2, h, w), dtype=np.float32)
        
        # Convert to torch tensors
        he_img = torch.from_numpy(he_img).permute(2, 0, 1).float()  # (3, H, W)
        mif_img = torch.from_numpy(mif_img).permute(2, 0, 1).float()  # (2, H, W)
        
        he_nuclei_mask = torch.from_numpy(he_nuclei_mask).long()
        he_cell_mask = torch.from_numpy(he_cell_mask).long()
        mif_nuclei_mask = torch.from_numpy(mif_nuclei_mask).long()
        mif_cell_mask = torch.from_numpy(mif_cell_mask).long()
        
        he_nuclei_hv = torch.from_numpy(he_nuclei_hv).float()
        he_cell_hv = torch.from_numpy(he_cell_hv).float()
        mif_nuclei_hv = torch.from_numpy(mif_nuclei_hv).float()
        mif_cell_hv = torch.from_numpy(mif_cell_hv).float()
        
        # Apply augmentation if training
        if self.augment is not None:
            (he_img, mif_img, he_nuclei_mask, he_cell_mask,
             mif_nuclei_mask, mif_cell_mask, he_nuclei_hv, he_cell_hv,
             mif_nuclei_hv, mif_cell_hv) = self.augment(
                he_img, mif_img, he_nuclei_mask, he_cell_mask,
                mif_nuclei_mask, mif_cell_mask, he_nuclei_hv, he_cell_hv,
                mif_nuclei_hv, mif_cell_hv
            )
        
        # Convert masks to binary (instance maps -> binary masks)
        he_nuclei_binary = (he_nuclei_mask > 0).long()
        he_cell_binary = (he_cell_mask > 0).long()
        mif_nuclei_binary = (mif_nuclei_mask > 0).long()
        mif_cell_binary = (mif_cell_mask > 0).long()
        
        # Concatenate images
        combined_img = torch.cat([he_img, mif_img], dim=0)  # (5, H, W)
        
        return {
            'image': combined_img,  # (5, H, W) - Combined
            'he_image': he_img,  # (3, H, W)
            'mif_image': mif_img,  # (2, H, W)
            'he_nuclei_mask': he_nuclei_binary,  # (H, W)
            'he_cell_mask': he_cell_binary,  # (H, W)
            'mif_nuclei_mask': mif_nuclei_binary,  # (H, W)
            'mif_cell_mask': mif_cell_binary,  # (H, W)
            'he_nuclei_hv': he_nuclei_hv,  # (2, H, W)
            'he_cell_hv': he_cell_hv,  # (2, H, W)
            'mif_nuclei_hv': mif_nuclei_hv,  # (2, H, W)
            'mif_cell_hv': mif_cell_hv,  # (2, H, W)
            'he_nuclei_instance': he_nuclei_mask,  # (H, W) - Original instance IDs
            'he_cell_instance': he_cell_mask,
            'mif_nuclei_instance': mif_nuclei_mask,
            'mif_cell_instance': mif_cell_mask,
        }


class CRCCachedDataset(CRCZarrDataset):
    """
    Memory-cached version of CRC dataset
    Preloads all data to RAM for maximum speed
    """
    
    def __init__(self, samples: List[str], config, training: bool = True):
        """
        Initialize cached dataset
        
        Args:
            samples: List of sample names
            config: Config object
            training: Whether this is training set
        """
        self.cache = None
        self.cache_loaded = False
        
        # Try to load from cache file
        if config.use_cache and not config.force_regenerate_cache:
            cache_path = self._get_cache_path(config, samples, training)
            if os.path.exists(cache_path):
                print(f"📦 Loading from cache: {cache_path}")
                self._load_cache(cache_path)
                self.samples = samples
                self.config = config
                self.training = training
                self.zarr_base = config.zarr_base
                
                # Initialize augmentation
                if training and config.train_augment:
                    aug_config = config.get_augmentation_config()
                    self.augment = MedicalImageAugmentation(aug_config)
                else:
                    self.augment = None
                
                return
        
        # Initialize parent class
        super().__init__(samples, config, training)
        
        # Preload all data to memory
        if config.use_cache:
            print(f"🚀 Preloading {len(self.patch_info)} patches to memory...")
            self._preload_to_memory()
            
            # Save cache
            cache_path = self._get_cache_path(config, samples, training)
            print(f"💾 Saving cache to: {cache_path}")
            self._save_cache(cache_path)
    
    def _get_cache_path(self, config, samples, training):
        """Generate unique cache path based on config and samples"""
        import hashlib
        
        # Create unique identifier from samples
        samples_str = "_".join(sorted(samples))
        samples_hash = hashlib.md5(samples_str.encode()).hexdigest()[:8]
        
        split_name = "train" if training else "val"
        cache_filename = f"crc_cache_{split_name}_{samples_hash}.pkl"
        
        return os.path.join(config.cache_dir, cache_filename)
    
    def _preload_to_memory(self):
        """Preload all patches to memory"""
        self.cache = []
        
        for idx in tqdm(range(len(self.patch_info)), desc="Preloading"):
            # Get data using parent class method
            data = super().__getitem__(idx)
            self.cache.append(data)
        
        self.cache_loaded = True
        
        # Report memory usage
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
        print(f"✅ Preloaded {len(self.cache)} patches")
        print(f"   Memory usage: {memory_gb:.2f} GB")
    
    def _save_cache(self, cache_path):
        """Save cache to disk"""
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'cache': self.cache,
                'patch_info': self.patch_info
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        cache_size_gb = os.path.getsize(cache_path) / (1024 ** 3)
        print(f"💾 Cache saved: {cache_size_gb:.2f} GB")
    
    def _load_cache(self, cache_path):
        """Load cache from disk"""
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        self.cache = data['cache']
        self.patch_info = data['patch_info']
        self.cache_loaded = True
        
        print(f"📦 Loaded {len(self.cache)} patches from cache")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from cache"""
        if self.cache_loaded:
            data = self.cache[idx]
            
            # Apply augmentation if training (must be done per-access, not cached)
            if self.augment is not None:
                (he_img, mif_img, he_nuclei_mask, he_cell_mask,
                 mif_nuclei_mask, mif_cell_mask, he_nuclei_hv, he_cell_hv,
                 mif_nuclei_hv, mif_cell_hv) = self.augment(
                    data['he_image'], data['mif_image'],
                    data['he_nuclei_instance'], data['he_cell_instance'],
                    data['mif_nuclei_instance'], data['mif_cell_instance'],
                    data['he_nuclei_hv'], data['he_cell_hv'],
                    data['mif_nuclei_hv'], data['mif_cell_hv']
                )
                
                # Update data dict
                data = data.copy()
                data['he_image'] = he_img
                data['mif_image'] = mif_img
                data['image'] = torch.cat([he_img, mif_img], dim=0)
                data['he_nuclei_mask'] = (he_nuclei_mask > 0).long()
                data['he_cell_mask'] = (he_cell_mask > 0).long()
                data['mif_nuclei_mask'] = (mif_nuclei_mask > 0).long()
                data['mif_cell_mask'] = (mif_cell_mask > 0).long()
                data['he_nuclei_hv'] = he_nuclei_hv
                data['he_cell_hv'] = he_cell_hv
                data['mif_nuclei_hv'] = mif_nuclei_hv
                data['mif_cell_hv'] = mif_cell_hv
            
            return data
        else:
            # Fall back to parent class if cache not loaded
            return super().__getitem__(idx)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to batch dictionary outputs
    
    Args:
        batch: List of sample dictionaries
    
    Returns:
        Batched dictionary
    """
    # Stack all tensors
    batched = {}
    
    for key in batch[0].keys():
        batched[key] = torch.stack([item[key] for item in batch])
    
    return batched


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Config object with all settings
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Get splits
    train_samples, val_samples, test_samples = config.get_splits()
    
    if config.verbose:
        print(f"\n{'='*80}")
        print("CREATING DATALOADERS")
        print(f"{'='*80}")
        print(f"Strategy: {config.strategy}")
        print(f"Use Cache: {config.use_cache}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Num Workers: {config.num_workers}")
        print(f"{'='*80}\n")
    
    # Select dataset class based on strategy
    if config.strategy == 'memory' and config.use_cache:
        dataset_class = CRCCachedDataset
    else:
        dataset_class = CRCZarrDataset
    
    # Create datasets
    train_dataset = dataset_class(train_samples, config, training=True)
    val_dataset = dataset_class(val_samples, config, training=False)
    test_dataset = dataset_class(test_samples, config, training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    if config.verbose:
        print(f"✅ Dataloaders created:")
        print(f"   Train: {len(train_dataset)} patches ({len(train_loader)} batches)")
        print(f"   Val: {len(val_dataset)} patches ({len(val_loader)} batches)")
        print(f"   Test: {len(test_dataset)} patches ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader
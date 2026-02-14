"""
CRC Dataset - Main dataset class for loading Zarr data
Handles multi-modal imaging (H&E + MIF) with instance segmentation
Supports multiple cancer types (CRC, Xenium, TissueNet)
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
    Multi-cancer Dataset loader for Zarr format data
    
    Features:
    - Loads H&E (3 channels) + MIF (2 channels) images
    - Handles 4 mask types: HE nuclei, HE cells, MIF nuclei, MIF cells
    - Generates 4 HV maps using PanNuke method
    - Supports multiple cancer types (CRC, Xenium, TissueNet)
    - Optional memory caching for speed
    - Data augmentation support
    """
    
    def __init__(self, samples: List[str], config, training: bool = True):
        """
        Initialize dataset
        
        Args:
            samples: List of sample names (e.g., ['CRC01', 'breast', 'train'])
            config: Config object with all settings
            training: Whether this is training set (enables augmentation)
        """
        self.samples = samples
        self.config = config
        self.training = training
        self.zarr_bases = config.zarr_bases  # Dictionary of zarr bases
        
        # Build patch index (sample_name, sub_sample, patch_idx)
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
    
    def _get_zarr_path(self, sample_name: str) -> str:
        """
        Get zarr path for a given sample based on naming convention
        """
        # 1. TissueNet Splits (train, val, test)
        if sample_name in ['train', 'val', 'test']:
            return self.zarr_bases['tissuenet']
            
        # 2. CRC Samples
        elif sample_name.startswith('CRC'):
            return self.zarr_bases['crc']
            
        # 3. Xenium Samples (Default fallback)
        else:
            return self.zarr_bases.get('xenium', list(self.zarr_bases.values())[0])

    def _get_dataset_type(self, sample_name: str) -> str:
        """Helper to identify dataset source"""
        if sample_name in ['train', 'val', 'test']:
            return 'tissuenet'
        elif sample_name.startswith('CRC'):
            return 'crc'
        return 'xenium'
    
    def _build_patch_index(self):
            """Build index of all patches across samples"""
            if self.config.verbose:
                print(f"📊 Building patch index for {len(self.samples)} samples...")
            
            for sample in self.samples:
                zarr_path = self._get_zarr_path(sample)
                dataset_type = self._get_dataset_type(sample)
                
                # --- Case 1: TissueNet (Nested folder structure) ---
                if dataset_type == 'tissuenet':
                    # For TissueNet, 'sample' is the split name ('train', 'val', or 'test')
                    split_path = os.path.join(zarr_path, sample)
                    
                    if not os.path.exists(split_path):
                        # Fallback: Check if zarr_path ITSELF is the split path
                        if os.path.basename(zarr_path) == sample:
                            split_path = zarr_path
                        else:
                            print(f"  ⚠️  TissueNet split not found at: {split_path}")
                            continue
                    
                    # Get all sub-directories (tissuenet_mosaic_X or tissuenet_sample_X)
                    try:
                        sub_samples = sorted([
                            d for d in os.listdir(split_path) 
                            if os.path.isdir(os.path.join(split_path, d))
                        ])
                        
                        if not sub_samples:
                            print(f"  ⚠️  TissueNet: No sub-folders found in {split_path}")
                            continue

                        for sub in sub_samples:
                            # Patch index is always 0 because your script saves 1 patch per Zarr
                            self.patch_info.append((sample, sub, 0))
                            
                        if self.config.verbose:
                            print(f"  ✅ TissueNet {sample}: {len(sub_samples)} samples loaded")
                    except Exception as e:
                        print(f"  ⚠️  TissueNet error reading {split_path}: {e}")

                # --- Case 2: CRC / Xenium (Standard multi-patch Zarr) ---
                else:
                    # Based on your setup, check for the 'he' folder first
                    he_zarr_path = os.path.join(zarr_path, sample, 'he', 'images.zarr')
                    
                    if not os.path.exists(he_zarr_path):
                        # Fallback for Xenium which might not have 'he' subfolder
                        he_zarr_path = os.path.join(zarr_path, sample, 'images.zarr')

                    try:
                        if not os.path.exists(he_zarr_path):
                            print(f"  ⚠️  {sample}: Zarr not found at {he_zarr_path}")
                            continue

                        z = zarr.open(he_zarr_path, 'r')
                        n_patches = z.shape[0]
                        for patch_idx in range(n_patches):
                            # (sample_name, sub_sample_folder, patch_idx_in_zarr)
                            self.patch_info.append((sample, None, patch_idx))
                        
                        if self.config.verbose:
                            print(f"  ✅ {sample} ({dataset_type}): {n_patches} patches")
                    except Exception as e:
                        print(f"  ⚠️  {sample}: Could not load - {e}")

    def __len__(self):
        return len(self.patch_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            """
            Get a single patch with all modalities and masks
            """
            sample_name, sub_sample, patch_idx = self.patch_info[idx]
            dataset_type = self._get_dataset_type(sample_name)
            zarr_path = self._get_zarr_path(sample_name)
            
            # --- 1. Load Data ---
            try:
                if dataset_type == 'tissuenet':
                    # TissueNet Path: base/split/tissuenet_sample_X/mif/
                    base_folder = os.path.join(zarr_path, sample_name, sub_sample)
                    
                    # Load from 'mif' subfolder as per your conversion script
                    # Index [0] because each TissueNet Zarr contains exactly one stitched mosaic
                    mif_img = zarr.open(os.path.join(base_folder, 'mif/images.zarr'), 'r')[0]
                    mif_nuclei_mask = zarr.open(os.path.join(base_folder, 'mif/nuclei_masks.zarr'), 'r')[0]
                    mif_cell_mask = zarr.open(os.path.join(base_folder, 'mif/cell_masks.zarr'), 'r')[0]
                    
                    # Generate blank H&E placeholders matching TissueNet patch size
                    h, w = mif_img.shape[:2]
                    he_img = np.zeros((h, w, 3), dtype=np.uint8)
                    he_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                    he_cell_mask = np.zeros((h, w), dtype=np.int32)
                    
                else:
                    # Standard CRC / Xenium Path (Multi-patch Zarrs)
                    base_folder = os.path.join(zarr_path, sample_name)
                    
                    he_img = zarr.open(os.path.join(base_folder, 'he/images.zarr'), 'r')[patch_idx]
                    he_nuclei_mask = zarr.open(os.path.join(base_folder, 'he/nuclei_masks.zarr'), 'r')[patch_idx]
                    he_cell_mask = zarr.open(os.path.join(base_folder, 'he/cell_masks.zarr'), 'r')[patch_idx]
                    
                    mif_img = zarr.open(os.path.join(base_folder, 'mif/images.zarr'), 'r')[patch_idx]
                    mif_nuclei_mask = zarr.open(os.path.join(base_folder, 'mif/nuclei_masks.zarr'), 'r')[patch_idx]
                    mif_cell_mask = zarr.open(os.path.join(base_folder, 'mif/cell_masks.zarr'), 'r')[patch_idx]

            except Exception as e:
                if self.config.verbose:
                    print(f"❌ Error loading {dataset_type} ({sample_name}/{sub_sample}): {e}")
                # Skip corrupted/missing sample by picking another index
                return self.__getitem__(np.random.randint(0, len(self.patch_info)))

            # --- 2. Normalize ---
            # H&E: 0-255 -> 0.0-1.0
            he_img = he_img.astype(np.float32) / self.config.he_max_value
            
            # MIF: Using the max value from your conversion script (65535.0 for uint16)
            mif_img = mif_img.astype(np.float32) / self.config.mif_max_value
            mif_img = np.clip(mif_img, 0.0, 1.0) # Safety clip
            
            # --- 3. Cast Masks ---
            he_nuclei_mask = he_nuclei_mask.astype(np.int64)
            he_cell_mask = he_cell_mask.astype(np.int64)
            mif_nuclei_mask = mif_nuclei_mask.astype(np.int64)
            mif_cell_mask = mif_cell_mask.astype(np.int64)
            
            # --- 4. Generate HV Maps ---
            if self.config.generate_hv_maps:
                # batch_generate_hv_maps expects (Batch, H, W)
                he_n_hv = batch_generate_hv_maps(he_nuclei_mask[np.newaxis, ...], method=self.config.hv_method)[0]
                he_c_hv = batch_generate_hv_maps(he_cell_mask[np.newaxis, ...], method=self.config.hv_method)[0]
                mif_n_hv = batch_generate_hv_maps(mif_nuclei_mask[np.newaxis, ...], method=self.config.hv_method)[0]
                mif_c_hv = batch_generate_hv_maps(mif_cell_mask[np.newaxis, ...], method=self.config.hv_method)[0]
            else:
                h, w = mif_img.shape[:2]
                he_n_hv = np.zeros((2, h, w), dtype=np.float32)
                he_c_hv = np.zeros((2, h, w), dtype=np.float32)
                mif_n_hv = np.zeros((2, h, w), dtype=np.float32)
                mif_c_hv = np.zeros((2, h, w), dtype=np.float32)
            
            # --- 5. To Tensor ---
            he_img = torch.from_numpy(he_img).permute(2, 0, 1).float()    # (3, H, W)
            mif_img = torch.from_numpy(mif_img).permute(2, 0, 1).float()  # (2, H, W)
            
            he_nuclei_mask = torch.from_numpy(he_nuclei_mask).long()
            he_cell_mask = torch.from_numpy(he_cell_mask).long()
            mif_nuclei_mask = torch.from_numpy(mif_nuclei_mask).long()
            mif_cell_mask = torch.from_numpy(mif_cell_mask).long()
            
            he_n_hv = torch.from_numpy(he_n_hv).float()
            he_c_hv = torch.from_numpy(he_c_hv).float()
            mif_n_hv = torch.from_numpy(mif_n_hv).float()
            mif_c_hv = torch.from_numpy(mif_c_hv).float()
            
            # --- 6. Augmentation ---
            if self.augment is not None:
                (he_img, mif_img, he_nuclei_mask, he_cell_mask,
                mif_nuclei_mask, mif_cell_mask, he_n_hv, he_c_hv,
                mif_n_hv, mif_c_hv) = self.augment(
                    he_img, mif_img, he_nuclei_mask, he_cell_mask,
                    mif_nuclei_mask, mif_cell_mask, he_n_hv, he_c_hv,
                    mif_n_hv, mif_c_hv
                )
            
            # --- 7. Binary Masks ---
            he_n_bin = (he_nuclei_mask > 0).long()
            he_c_bin = (he_cell_mask > 0).long()
            mif_n_bin = (mif_nuclei_mask > 0).long()
            mif_c_bin = (mif_cell_mask > 0).long()
            
            # Combined Input: H&E (3) + MIF (2) = 5 Channels
            combined_img = torch.cat([he_img, mif_img], dim=0)
            
            return {
            'image': combined_img,
            'he_image': he_img,
            'mif_image': mif_img,
            'he_nuclei_mask': he_n_bin,
            'he_cell_mask': he_c_bin,
            'mif_nuclei_mask': mif_n_bin,
            'mif_cell_mask': mif_c_bin,
            'he_nuclei_hv': he_n_hv,
            'he_cell_hv': he_c_hv,
            'mif_nuclei_hv': mif_n_hv,
            'mif_cell_hv': mif_c_hv,
            'he_nuclei_instance': he_nuclei_mask,
            'he_cell_instance': he_cell_mask,
            'mif_nuclei_instance': mif_nuclei_mask,
            'mif_cell_instance': mif_cell_mask,
            'sample_name': sub_sample if sub_sample else sample_name,
            'patch_idx': patch_idx,
            'dataset_source': dataset_type 
        }

class CRCCachedDataset(CRCZarrDataset):
    """
    Memory-cached version of multi-cancer dataset
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
                self.zarr_bases = config.zarr_bases
                
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
        cache_filename = f"combined_cache_{split_name}_{samples_hash}.pkl"
        
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
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
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
            if self.augment is not None and self.training:
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
        if key in ['sample_name', 'dataset_source']:  # Don't stack strings
            batched[key] = [item[key] for item in batch]
        elif key == 'patch_idx':
            batched[key] = torch.tensor([item[key] for item in batch])
        else:
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
        
        # Helper to count types
        def count_types(samples):
            crc = sum(1 for s in samples if s.startswith('CRC'))
            tn = sum(1 for s in samples if s in ['train', 'val', 'test'])
            xen = len(samples) - crc - tn
            return crc, xen, tn

        c_tr, x_tr, t_tr = count_types(train_samples)
        print(f"\nTrain split: {c_tr} CRC + {x_tr} Xenium + {t_tr} TissueNet")
        
        c_val, x_val, t_val = count_types(val_samples)
        print(f"Val split:   {c_val} CRC + {x_val} Xenium + {t_val} TissueNet")
        
        c_te, x_te, t_te = count_types(test_samples)
        print(f"Test split:  {c_te} CRC + {x_te} Xenium + {t_te} TissueNet")
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
"""
Medical Image Augmentation for CRC Dataset
Applies consistent geometric and photometric augmentations
to multi-modal data (H&E + MIF) and their masks/HV maps
"""

import numpy as np
import cv2
import random
import torch
from typing import Tuple, Dict


class MedicalImageAugmentation:
    """
    Augmentation pipeline for medical imaging with multi-modal data
    
    Key features:
    - Geometric augmentations applied consistently to all modalities
    - Photometric augmentations applied selectively (H&E only for color)
    - HV maps correctly transformed with geometric changes
    - Stain augmentation for H&E pathology images
    - Random resize for magnification invariance
    """
    
    def __init__(self, config: Dict):
        """
        Initialize augmentation pipeline from config
        
        Args:
            config: Dictionary containing augmentation parameters
        """
        self.config = config
        self.enabled = config.get('train_augment', True)
        self.augment_prob = config.get('augment_prob', 0.5)
        
        # Geometric augmentation params
        geom = config.get('geometric', {})
        self.flip_prob = geom.get('flip_prob', 0.6)
        self.rotation_prob = geom.get('rotation_prob', 0.6)
        self.rotation_angles = geom.get('rotation_angles', [90, 180, 270])
        self.scale_prob = geom.get('scale_prob', 0.25)
        self.scale_range = tuple(geom.get('scale_range', [0.95, 1.05]))
        
        # ⭐ NEW: Random resize for magnification invariance
        self.random_resize_prob = geom.get('random_resize_prob', 0.0)
        self.resize_range = tuple(geom.get('resize_range', [0.5, 2.0]))
        
        # Photometric params (H&E only)
        photo = config.get('photometric', {})
        self.brightness_prob = photo.get('brightness_prob', 0.5)
        self.brightness_factor = photo.get('brightness_factor', 0.25)
        self.contrast_prob = photo.get('contrast_prob', 0.5)
        self.contrast_factor = photo.get('contrast_factor', 0.25)
        
        # Stain augmentation (H&E only)
        stain = config.get('stain', {})
        self.stain_prob = stain.get('stain_prob', 0.5)
        self.hue_shift_range = tuple(stain.get('hue_shift_range', [-0.05, 0.05]))
        self.saturation_range = tuple(stain.get('saturation_range', [0.85, 1.15]))
        
        # Noise and blur
        noise_blur = config.get('noise_blur', {})
        self.blur_prob = noise_blur.get('blur_prob', 0.4)
        self.blur_sigma_range = tuple(noise_blur.get('blur_sigma_range', [0.1, 1.0]))
        self.noise_prob = noise_blur.get('noise_prob', 0.3)
        self.noise_he_std_range = tuple(noise_blur.get('noise_he_std_range', [0.01, 0.05]))
        self.noise_mif_std_range = tuple(noise_blur.get('noise_mif_std_range', [0.005, 0.02]))
        
        # Cutout
        cutout = config.get('cutout', {})
        self.cutout_prob = cutout.get('cutout_prob', 0.3)
        self.cutout_size_range = tuple(cutout.get('cutout_size_range', [8, 32]))
    
    def __call__(self, he_image: torch.Tensor, mif_image: torch.Tensor,
                 he_nuclei_mask: torch.Tensor, he_cell_mask: torch.Tensor,
                 mif_nuclei_mask: torch.Tensor, mif_cell_mask: torch.Tensor,
                 he_nuclei_hv: torch.Tensor, he_cell_hv: torch.Tensor,
                 mif_nuclei_hv: torch.Tensor, mif_cell_hv: torch.Tensor) -> Tuple:
        """
        Apply augmentations to all data
        
        Args:
            he_image: (3, H, W) H&E image
            mif_image: (2, H, W) MIF image
            *_mask: (H, W) instance masks
            *_hv: (2, H, W) HV maps
        
        Returns:
            Tuple of augmented tensors in same order
        """
        # Skip if disabled or random check fails
        if not self.enabled or random.random() > self.augment_prob:
            return (he_image, mif_image, he_nuclei_mask, he_cell_mask,
                    mif_nuclei_mask, mif_cell_mask, he_nuclei_hv, he_cell_hv,
                    mif_nuclei_hv, mif_cell_hv)
        
        # Convert to numpy for augmentation (H, W, C)
        he_np = he_image.permute(1, 2, 0).numpy()  # (H, W, 3)
        mif_np = mif_image.permute(1, 2, 0).numpy()  # (H, W, 2)
        
        # Masks (H, W)
        he_nuclei_mask_np = he_nuclei_mask.numpy()
        he_cell_mask_np = he_cell_mask.numpy()
        mif_nuclei_mask_np = mif_nuclei_mask.numpy()
        mif_cell_mask_np = mif_cell_mask.numpy()
        
        # HV maps (H, W, 2)
        he_nuclei_hv_np = he_nuclei_hv.permute(1, 2, 0).numpy()
        he_cell_hv_np = he_cell_hv.permute(1, 2, 0).numpy()
        mif_nuclei_hv_np = mif_nuclei_hv.permute(1, 2, 0).numpy()
        mif_cell_hv_np = mif_cell_hv.permute(1, 2, 0).numpy()
        
        # ====================================================================
        # GEOMETRIC AUGMENTATIONS (applied to ALL data)
        # ====================================================================
        
        # Horizontal flip
        if random.random() < self.flip_prob:
            he_np = np.fliplr(he_np)
            mif_np = np.fliplr(mif_np)
            he_nuclei_mask_np = np.fliplr(he_nuclei_mask_np)
            he_cell_mask_np = np.fliplr(he_cell_mask_np)
            mif_nuclei_mask_np = np.fliplr(mif_nuclei_mask_np)
            mif_cell_mask_np = np.fliplr(mif_cell_mask_np)
            
            # Flip horizontal component of HV maps
            he_nuclei_hv_np = np.fliplr(he_nuclei_hv_np)
            he_nuclei_hv_np[:, :, 0] = -he_nuclei_hv_np[:, :, 0]
            
            he_cell_hv_np = np.fliplr(he_cell_hv_np)
            he_cell_hv_np[:, :, 0] = -he_cell_hv_np[:, :, 0]
            
            mif_nuclei_hv_np = np.fliplr(mif_nuclei_hv_np)
            mif_nuclei_hv_np[:, :, 0] = -mif_nuclei_hv_np[:, :, 0]
            
            mif_cell_hv_np = np.fliplr(mif_cell_hv_np)
            mif_cell_hv_np[:, :, 0] = -mif_cell_hv_np[:, :, 0]
        
        # Vertical flip
        if random.random() < self.flip_prob:
            he_np = np.flipud(he_np)
            mif_np = np.flipud(mif_np)
            he_nuclei_mask_np = np.flipud(he_nuclei_mask_np)
            he_cell_mask_np = np.flipud(he_cell_mask_np)
            mif_nuclei_mask_np = np.flipud(mif_nuclei_mask_np)
            mif_cell_mask_np = np.flipud(mif_cell_mask_np)
            
            # Flip vertical component of HV maps
            he_nuclei_hv_np = np.flipud(he_nuclei_hv_np)
            he_nuclei_hv_np[:, :, 1] = -he_nuclei_hv_np[:, :, 1]
            
            he_cell_hv_np = np.flipud(he_cell_hv_np)
            he_cell_hv_np[:, :, 1] = -he_cell_hv_np[:, :, 1]
            
            mif_nuclei_hv_np = np.flipud(mif_nuclei_hv_np)
            mif_nuclei_hv_np[:, :, 1] = -mif_nuclei_hv_np[:, :, 1]
            
            mif_cell_hv_np = np.flipud(mif_cell_hv_np)
            mif_cell_hv_np[:, :, 1] = -mif_cell_hv_np[:, :, 1]
        
        # Rotation (90, 180, 270 degrees only)
        if random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
            
            he_np = np.rot90(he_np, k)
            mif_np = np.rot90(mif_np, k)
            he_nuclei_mask_np = np.rot90(he_nuclei_mask_np, k)
            he_cell_mask_np = np.rot90(he_cell_mask_np, k)
            mif_nuclei_mask_np = np.rot90(mif_nuclei_mask_np, k)
            mif_cell_mask_np = np.rot90(mif_cell_mask_np, k)
            
            # Rotate HV components accordingly
            he_nuclei_hv_np = self._rotate_hv_map(he_nuclei_hv_np, k)
            he_cell_hv_np = self._rotate_hv_map(he_cell_hv_np, k)
            mif_nuclei_hv_np = self._rotate_hv_map(mif_nuclei_hv_np, k)
            mif_cell_hv_np = self._rotate_hv_map(mif_cell_hv_np, k)
        
        # Random scale
        if random.random() < self.scale_prob:
            scale_factor = random.uniform(*self.scale_range)
            if scale_factor != 1.0:
                (he_np, mif_np, he_nuclei_mask_np, he_cell_mask_np,
                 mif_nuclei_mask_np, mif_cell_mask_np, he_nuclei_hv_np,
                 he_cell_hv_np, mif_nuclei_hv_np, mif_cell_hv_np) = self._apply_scale(
                    he_np, mif_np, he_nuclei_mask_np, he_cell_mask_np,
                    mif_nuclei_mask_np, mif_cell_mask_np, he_nuclei_hv_np,
                    he_cell_hv_np, mif_nuclei_hv_np, mif_cell_hv_np, scale_factor
                )
        
        # ⭐ NEW: Random resize (magnification simulation)
        if random.random() < self.random_resize_prob:
            (he_np, mif_np, he_nuclei_mask_np, he_cell_mask_np,
             mif_nuclei_mask_np, mif_cell_mask_np, he_nuclei_hv_np,
             he_cell_hv_np, mif_nuclei_hv_np, mif_cell_hv_np) = self._apply_random_resize(
                he_np, mif_np, he_nuclei_mask_np, he_cell_mask_np,
                mif_nuclei_mask_np, mif_cell_mask_np, he_nuclei_hv_np,
                he_cell_hv_np, mif_nuclei_hv_np, mif_cell_hv_np
            )
        
        # ====================================================================
        # PHOTOMETRIC AUGMENTATIONS (H&E only)
        # ====================================================================
        
        # Stain augmentation (H&E pathology specific)
        if random.random() < self.stain_prob:
            he_np = self._apply_stain_augmentation(he_np)
        
        # Brightness
        if random.random() < self.brightness_prob:
            brightness_delta = random.uniform(-self.brightness_factor, self.brightness_factor)
            he_np = np.clip(he_np + brightness_delta, 0.0, 1.0)
        
        # Contrast
        if random.random() < self.contrast_prob:
            contrast_factor = random.uniform(1.0 - self.contrast_factor,
                                            1.0 + self.contrast_factor)
            he_np = np.clip(he_np * contrast_factor, 0.0, 1.0)
        
        # ====================================================================
        # NOISE AND BLUR (both modalities)
        # ====================================================================
        
        # Gaussian blur
        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma_range)
            for c in range(he_np.shape[2]):
                he_np[:, :, c] = cv2.GaussianBlur(he_np[:, :, c], (5, 5), sigma)
            for c in range(mif_np.shape[2]):
                mif_np[:, :, c] = cv2.GaussianBlur(mif_np[:, :, c], (5, 5), sigma)
        
        # Gaussian noise
        if random.random() < self.noise_prob:
            # H&E noise
            noise_std = random.uniform(*self.noise_he_std_range)
            he_noise = np.random.normal(0, noise_std, he_np.shape)
            he_np = np.clip(he_np + he_noise, 0.0, 1.0)
            
            # MIF noise (smaller magnitude)
            noise_std = random.uniform(*self.noise_mif_std_range)
            mif_noise = np.random.normal(0, noise_std, mif_np.shape)
            mif_np = np.clip(mif_np + mif_noise, 0.0, 1.0)
        
        # Convert back to tensors
        he_image = torch.from_numpy(he_np.copy()).permute(2, 0, 1).float()
        mif_image = torch.from_numpy(mif_np.copy()).permute(2, 0, 1).float()
        
        he_nuclei_mask = torch.from_numpy(he_nuclei_mask_np.copy()).long()
        he_cell_mask = torch.from_numpy(he_cell_mask_np.copy()).long()
        mif_nuclei_mask = torch.from_numpy(mif_nuclei_mask_np.copy()).long()
        mif_cell_mask = torch.from_numpy(mif_cell_mask_np.copy()).long()
        
        he_nuclei_hv = torch.from_numpy(he_nuclei_hv_np.copy()).permute(2, 0, 1).float()
        he_cell_hv = torch.from_numpy(he_cell_hv_np.copy()).permute(2, 0, 1).float()
        mif_nuclei_hv = torch.from_numpy(mif_nuclei_hv_np.copy()).permute(2, 0, 1).float()
        mif_cell_hv = torch.from_numpy(mif_cell_hv_np.copy()).permute(2, 0, 1).float()
        
        # ====================================================================
        # CUTOUT (applied to images only, after tensor conversion)
        # ====================================================================
        
        if random.random() < self.cutout_prob:
            h, w = he_image.shape[1], he_image.shape[2]
            cut_h = random.randint(*self.cutout_size_range)
            cut_w = random.randint(*self.cutout_size_range)
            y = random.randint(0, h - cut_h)
            x = random.randint(0, w - cut_w)
            
            he_image[:, y:y+cut_h, x:x+cut_w] = 0
            mif_image[:, y:y+cut_h, x:x+cut_w] = 0
        
        return (he_image, mif_image, he_nuclei_mask, he_cell_mask,
                mif_nuclei_mask, mif_cell_mask, he_nuclei_hv, he_cell_hv,
                mif_nuclei_hv, mif_cell_hv)
    
    def _rotate_hv_map(self, hv_map: np.ndarray, k: int) -> np.ndarray:
        """
        Rotate HV map and adjust components accordingly
        
        Args:
            hv_map: (H, W, 2) HV map
            k: Number of 90-degree rotations
        
        Returns:
            Rotated HV map
        """
        hv_map = np.rot90(hv_map, k)
        
        if k == 1:  # 90 degrees clockwise
            hv_temp = hv_map[:, :, 0].copy()
            hv_map[:, :, 0] = hv_map[:, :, 1]
            hv_map[:, :, 1] = -hv_temp
        elif k == 2:  # 180 degrees
            hv_map[:, :, 0] = -hv_map[:, :, 0]
            hv_map[:, :, 1] = -hv_map[:, :, 1]
        elif k == 3:  # 270 degrees clockwise
            hv_temp = hv_map[:, :, 0].copy()
            hv_map[:, :, 0] = -hv_map[:, :, 1]
            hv_map[:, :, 1] = hv_temp
        
        return hv_map
    
    def _apply_scale(self, he_np, mif_np, he_nuclei_mask_np, he_cell_mask_np,
                     mif_nuclei_mask_np, mif_cell_mask_np, he_nuclei_hv_np,
                     he_cell_hv_np, mif_nuclei_hv_np, mif_cell_hv_np, scale_factor):
        """Apply random scaling with center crop/pad back to original size"""
        h, w = he_np.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Resize all
        he_np = cv2.resize(he_np, (new_w, new_h))
        mif_np = cv2.resize(mif_np, (new_w, new_h))
        he_nuclei_mask_np = cv2.resize(he_nuclei_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        he_cell_mask_np = cv2.resize(he_cell_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mif_nuclei_mask_np = cv2.resize(mif_nuclei_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mif_cell_mask_np = cv2.resize(mif_cell_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        he_nuclei_hv_np = cv2.resize(he_nuclei_hv_np, (new_w, new_h))
        he_cell_hv_np = cv2.resize(he_cell_hv_np, (new_w, new_h))
        mif_nuclei_hv_np = cv2.resize(mif_nuclei_hv_np, (new_w, new_h))
        mif_cell_hv_np = cv2.resize(mif_cell_hv_np, (new_w, new_h))
        
        # Center crop or pad
        if scale_factor > 1.0:  # Crop
            start_y, start_x = (new_h - h) // 2, (new_w - w) // 2
            he_np = he_np[start_y:start_y+h, start_x:start_x+w]
            mif_np = mif_np[start_y:start_y+h, start_x:start_x+w]
            he_nuclei_mask_np = he_nuclei_mask_np[start_y:start_y+h, start_x:start_x+w]
            he_cell_mask_np = he_cell_mask_np[start_y:start_y+h, start_x:start_x+w]
            mif_nuclei_mask_np = mif_nuclei_mask_np[start_y:start_y+h, start_x:start_x+w]
            mif_cell_mask_np = mif_cell_mask_np[start_y:start_y+h, start_x:start_x+w]
            he_nuclei_hv_np = he_nuclei_hv_np[start_y:start_y+h, start_x:start_x+w]
            he_cell_hv_np = he_cell_hv_np[start_y:start_y+h, start_x:start_x+w]
            mif_nuclei_hv_np = mif_nuclei_hv_np[start_y:start_y+h, start_x:start_x+w]
            mif_cell_hv_np = mif_cell_hv_np[start_y:start_y+h, start_x:start_x+w]
        else:  # Pad
            pad_y, pad_x = (h - new_h) // 2, (w - new_w) // 2
            he_np = np.pad(he_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            mif_np = np.pad(mif_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            he_nuclei_mask_np = np.pad(he_nuclei_mask_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='reflect')
            he_cell_mask_np = np.pad(he_cell_mask_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='reflect')
            mif_nuclei_mask_np = np.pad(mif_nuclei_mask_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='reflect')
            mif_cell_mask_np = np.pad(mif_cell_mask_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='reflect')
            he_nuclei_hv_np = np.pad(he_nuclei_hv_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            he_cell_hv_np = np.pad(he_cell_hv_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            mif_nuclei_hv_np = np.pad(mif_nuclei_hv_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            mif_cell_hv_np = np.pad(mif_cell_hv_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
        
        return (he_np, mif_np, he_nuclei_mask_np, he_cell_mask_np,
                mif_nuclei_mask_np, mif_cell_mask_np, he_nuclei_hv_np,
                he_cell_hv_np, mif_nuclei_hv_np, mif_cell_hv_np)
    
    def _apply_random_resize(self, he_np, mif_np, he_nuclei_mask_np, he_cell_mask_np,
                             mif_nuclei_mask_np, mif_cell_mask_np, he_nuclei_hv_np,
                             he_cell_hv_np, mif_nuclei_hv_np, mif_cell_hv_np):
        """
        ⭐ NEW: Random resize augmentation - simulates different magnifications
        Resize to random scale, then crop/pad back to original size
        This is specifically designed to make the model magnification-agnostic
        """
        h, w = he_np.shape[:2]
        resize_factor = random.uniform(*self.resize_range)
        new_h, new_w = int(h * resize_factor), int(w * resize_factor)
        
        # Resize all data
        he_np = cv2.resize(he_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mif_np = cv2.resize(mif_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        he_nuclei_mask_np = cv2.resize(he_nuclei_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        he_cell_mask_np = cv2.resize(he_cell_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mif_nuclei_mask_np = cv2.resize(mif_nuclei_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mif_cell_mask_np = cv2.resize(mif_cell_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        he_nuclei_hv_np = cv2.resize(he_nuclei_hv_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        he_cell_hv_np = cv2.resize(he_cell_hv_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mif_nuclei_hv_np = cv2.resize(mif_nuclei_hv_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mif_cell_hv_np = cv2.resize(mif_cell_hv_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad back to original size
        if resize_factor > 1.0:  # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            he_np = he_np[start_y:start_y+h, start_x:start_x+w]
            mif_np = mif_np[start_y:start_y+h, start_x:start_x+w]
            he_nuclei_mask_np = he_nuclei_mask_np[start_y:start_y+h, start_x:start_x+w]
            he_cell_mask_np = he_cell_mask_np[start_y:start_y+h, start_x:start_x+w]
            mif_nuclei_mask_np = mif_nuclei_mask_np[start_y:start_y+h, start_x:start_x+w]
            mif_cell_mask_np = mif_cell_mask_np[start_y:start_y+h, start_x:start_x+w]
            he_nuclei_hv_np = he_nuclei_hv_np[start_y:start_y+h, start_x:start_x+w]
            he_cell_hv_np = he_cell_hv_np[start_y:start_y+h, start_x:start_x+w]
            mif_nuclei_hv_np = mif_nuclei_hv_np[start_y:start_y+h, start_x:start_x+w]
            mif_cell_hv_np = mif_cell_hv_np[start_y:start_y+h, start_x:start_x+w]
        else:  # Pad
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            he_np = np.pad(he_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            mif_np = np.pad(mif_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            # Change these lines (around line 370-375):
            he_nuclei_mask_np = np.pad(he_nuclei_mask_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='reflect')  # ← CHANGE
            he_cell_mask_np = np.pad(he_cell_mask_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='reflect')  # ← CHANGE
            mif_nuclei_mask_np = np.pad(mif_nuclei_mask_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='reflect')  # ← CHANGE
            mif_cell_mask_np = np.pad(mif_cell_mask_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)), mode='reflect')  # ← CHANGE
            he_nuclei_hv_np = np.pad(he_nuclei_hv_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            he_cell_hv_np = np.pad(he_cell_hv_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            mif_nuclei_hv_np = np.pad(mif_nuclei_hv_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
            mif_cell_hv_np = np.pad(mif_cell_hv_np, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), mode='reflect')
        
        return (he_np, mif_np, he_nuclei_mask_np, he_cell_mask_np,
                mif_nuclei_mask_np, mif_cell_mask_np, he_nuclei_hv_np,
                he_cell_hv_np, mif_nuclei_hv_np, mif_cell_hv_np)
    
    def _apply_stain_augmentation(self, he_np: np.ndarray) -> np.ndarray:
        """
        Apply H&E stain augmentation (hue shift + saturation)
        
        Args:
            he_np: (H, W, 3) H&E image in RGB
        
        Returns:
            Augmented image
        """
        # Convert to HSV
        he_hsv = cv2.cvtColor(he_np.astype(np.float32), cv2.COLOR_RGB2HSV)
        
        # Hue shift
        if random.random() < 0.6:
            hue_shift = random.uniform(*self.hue_shift_range)
            he_hsv[:, :, 0] = np.clip(he_hsv[:, :, 0] + hue_shift, 0, 1)
        
        # Saturation adjustment
        if random.random() < 0.5:
            saturation_factor = random.uniform(*self.saturation_range)
            he_hsv[:, :, 1] = np.clip(he_hsv[:, :, 1] * saturation_factor, 0, 1)
        
        # Convert back to RGB
        he_np = cv2.cvtColor(he_hsv, cv2.COLOR_HSV2RGB)
        
        return he_np
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WSI/Image loading utilities for multiple pathology formats."""

import openslide
import tifffile
from PIL import Image
import numpy as np
from pathlib import Path


class MultiFormatImageLoader:
    """Universal loader for pathology image formats
    
    Supports:
    - Whole slide formats: SVS, NDPI, MRXS, etc. (via OpenSlide)
    - OME-TIFF: Multi-page OME-TIFF files
    - Standard images: PNG, JPG, JPEG
    - Regular TIFF: Single/multi-page TIFF
    """
    
    OPENSLIDE_FORMATS = ['.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif']
    SIMPLE_FORMATS = ['.png', '.jpg', '.jpeg']
    OME_FORMATS = ['.ome.tif', '.ome.tiff']
    
    @staticmethod
    def load_image(image_path):
        """Load image from any supported format
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy.ndarray: RGB image (H, W, 3), uint8
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        ext = image_path.suffix.lower()
        
        # Check if OME-TIFF (check before regular TIFF)
        if image_path.name.lower().endswith(('.ome.tif', '.ome.tiff')):
            return MultiFormatImageLoader._load_ome_tiff(image_path)
        
        # Try OpenSlide for whole slide formats
        elif ext in MultiFormatImageLoader.OPENSLIDE_FORMATS:
            return MultiFormatImageLoader._load_openslide(image_path)
        
        # Simple formats (PNG, JPG)
        elif ext in MultiFormatImageLoader.SIMPLE_FORMATS:
            return MultiFormatImageLoader._load_simple(image_path)
        
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    @staticmethod
    def _load_openslide(path):
        """Load using OpenSlide (for SVS, TIFF, etc.)"""
        try:
            slide = openslide.OpenSlide(str(path))
            # Get level 0 (highest resolution)
            level = 0
            dimensions = slide.level_dimensions[level]
            
            # Read the region
            image = slide.read_region((0, 0), level, dimensions)
            image = np.array(image.convert('RGB'))
            
            slide.close()
            return image
        except Exception as e:
            print(f"⚠️  OpenSlide failed, trying tifffile: {e}")
            return MultiFormatImageLoader._load_tifffile(path)
    
    @staticmethod
    def _load_tifffile(path):
        """Load using tifffile (backup for TIFF)"""
        image = tifffile.imread(str(path))
        
        # Handle different TIFF formats
        if image.ndim == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        elif image.ndim == 3 and image.shape[0] < 10:  # (C, H, W) format
            image = np.transpose(image, (1, 2, 0))
            if image.shape[2] > 3:
                image = image[:, :, :3]
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
    
    @staticmethod
    def _load_ome_tiff(path):
        """Load OME-TIFF format"""
        with tifffile.TiffFile(str(path)) as tif:
            image = tif.asarray()
            
            # Handle multi-page OME-TIFF (take first page)
            if image.ndim == 4:  # (pages, H, W, C)
                image = image[0]
            elif image.ndim == 3 and image.shape[0] < 10:  # (C, H, W)
                image = np.transpose(image, (1, 2, 0))
            
            # Convert to RGB if needed
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            
            # Ensure uint8
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            return image
    
    @staticmethod
    def _load_simple(path):
        """Load simple formats (PNG, JPG)"""
        image = Image.open(str(path))
        image = np.array(image.convert('RGB'))
        return image
    
    @staticmethod
    def get_supported_formats():
        """Get list of all supported file extensions"""
        all_formats = (
            MultiFormatImageLoader.OPENSLIDE_FORMATS +
            MultiFormatImageLoader.SIMPLE_FORMATS +
            MultiFormatImageLoader.OME_FORMATS
        )
        return list(set(all_formats))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WSI/Image loading utilities for multiple pathology formats with streaming support."""

import openslide
import tifffile
from PIL import Image
import numpy as np
from pathlib import Path
from .channel_config import ChannelConfig


class MultiFormatImageLoader:
    """Universal loader for pathology image formats with streaming support
    
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
    def get_wsi_reader(image_path):
        """Get a WSI reader object for streaming tile access (NEW)
        
        Args:
            image_path: Path to WSI file
            
        Returns:
            WSIReader object with .read_region() method
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        ext = image_path.suffix.lower()
        
        # Try OpenSlide first for WSI formats
        if ext in MultiFormatImageLoader.OPENSLIDE_FORMATS:
            try:
                return OpenSlideReader(image_path)
            except:
                print(f"⚠️  OpenSlide failed, trying TiffReader")
                return TiffReader(image_path)
        
        # OME-TIFF
        elif image_path.name.lower().endswith(('.ome.tif', '.ome.tiff')):
            return TiffReader(image_path)
        
        # Simple formats - load fully (small images)
        elif ext in MultiFormatImageLoader.SIMPLE_FORMATS:
            return SimpleImageReader(image_path)
        
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    @staticmethod
    def load_image(image_path):
        """Load image from any supported format (LEGACY - loads full image)
        
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

    @staticmethod
    def load_mif_image(image_path, channel_config=None):
        """Load MIF image with channel selection (LEGACY - loads full image)
        
        Args:
            image_path: Path to MIF OME-TIFF file
            channel_config: ChannelConfig instance or None (uses first 2 channels)
            
        Returns:
            numpy.ndarray: 2-channel image (2, H, W), float32, normalized to 0-1
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load MIF with tifffile
        import tifffile
        mif_array = tifffile.imread(str(image_path))
        
        print(f"Loaded MIF: shape={mif_array.shape}, dtype={mif_array.dtype}")
        
        # Apply channel configuration
        if channel_config is not None:
            print(f"Applying channel config: {channel_config.get_description()}")
            mif_2ch = channel_config.select_channels(mif_array)
        else:
            # Default: use first 2 channels
            if mif_array.ndim == 3 and mif_array.shape[0] >= 2:
                mif_2ch = mif_array[:2, :, :]
                print(f"Using default: first 2 channels")
            else:
                raise ValueError(
                    f"Cannot auto-select channels from shape {mif_array.shape}. "
                    f"Please provide channel_config."
                )
        
        # Normalize to 0-1 (handle 16-bit)
        if mif_2ch.dtype == np.uint16:
            mif_2ch = mif_2ch.astype(np.float32) / 65535.0
        elif mif_2ch.dtype == np.uint8:
            mif_2ch = mif_2ch.astype(np.float32) / 255.0
        else:
            mif_2ch = mif_2ch.astype(np.float32)
        
        print(f"Output: shape={mif_2ch.shape}, dtype={mif_2ch.dtype}, range=[{mif_2ch.min():.3f}, {mif_2ch.max():.3f}]")
        
        return mif_2ch


# NEW: Reader classes for streaming tile access

class OpenSlideReader:
    """Stream tiles from OpenSlide-compatible WSI"""
    
    def __init__(self, path):
        self.slide = openslide.OpenSlide(str(path))
        self.dimensions = self.slide.level_dimensions[0]
        self.width, self.height = self.dimensions
        
    def read_region(self, location, size):
        """Read a region from the slide
        
        Args:
            location: (x, y) top-left corner
            size: (width, height) of region
            
        Returns:
            numpy array (H, W, 3), uint8, RGB
        """
        x, y = location
        w, h = size
        region = self.slide.read_region((x, y), 0, (w, h))
        return np.array(region.convert('RGB'))
    
    def close(self):
        self.slide.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class TiffReader:
    """Stream tiles from TIFF/OME-TIFF files"""
    
    def __init__(self, path):
        self.tif = tifffile.TiffFile(str(path))
        # Get first page/series
        if hasattr(self.tif, 'series'):
            self.page = self.tif.series[0]
            self.height, self.width = self.page.shape[:2]
        else:
            self.page = self.tif.pages[0]
            self.height, self.width = self.page.shape[:2]
        
    def read_region(self, location, size):
        """Read a region from the TIFF
        
        Args:
            location: (x, y) top-left corner
            size: (width, height) of region
            
        Returns:
            numpy array (H, W, 3), uint8, RGB
        """
        x, y = location
        w, h = size
        
        # Load full image and crop (tifffile doesn't support region reading directly)
        # For large TIFFs, this is still faster than loading full image upfront
        full_image = self.tif.asarray()
        
        # Handle channel-first format
        if full_image.ndim == 3 and full_image.shape[0] < 10:
            full_image = np.transpose(full_image, (1, 2, 0))
        
        # Crop region
        region = full_image[y:y+h, x:x+w]
        
        # Convert to RGB if needed
        if region.ndim == 2:
            region = np.stack([region] * 3, axis=-1)
        elif region.shape[2] > 3:
            region = region[:, :, :3]
        
        # Ensure uint8
        if region.dtype != np.uint8:
            if region.max() <= 1.0:
                region = (region * 255).astype(np.uint8)
            else:
                region = region.astype(np.uint8)
        
        return region
    
    def close(self):
        self.tif.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class SimpleImageReader:
    """Reader for simple image formats (loads once, streams tiles)"""
    
    def __init__(self, path):
        self.image = Image.open(str(path)).convert('RGB')
        self.width, self.height = self.image.size
        self._array = None
    
    def read_region(self, location, size):
        """Read a region from the image
        
        Args:
            location: (x, y) top-left corner
            size: (width, height) of region
            
        Returns:
            numpy array (H, W, 3), uint8, RGB
        """
        if self._array is None:
            self._array = np.array(self.image)
        
        x, y = location
        w, h = size
        return self._array[y:y+h, x:x+w]
    
    def close(self):
        if self.image:
            self.image.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
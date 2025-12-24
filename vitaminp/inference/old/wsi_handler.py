# -*- coding: utf-8 -*-
# WSI Handler for VitaminP Inference
# Handles WSI loading, metadata extraction, and tile generation
# Updated with multi-format support (OpenSlide + TiffSlide)

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field

# Multi-format WSI support
try:
    from openslide import OpenSlide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Warning: OpenSlide not available. Install with: pip install openslide-python")

try:
    from tiffslide import TiffSlide
    TIFFSLIDE_AVAILABLE = True
except ImportError:
    TIFFSLIDE_AVAILABLE = False
    print("Warning: TiffSlide not available. Install with: pip install tiffslide")

from PIL import Image


class SimpleImageSlide:
    """Wrapper to make simple images (PNG, JPEG) compatible with OpenSlide/TiffSlide API.
    
    This allows processing of simple non-pyramidal images using the same interface.
    """
    
    def __init__(self, filepath: str):
        """Load a simple image file.
        
        Args:
            filepath: Path to image file (PNG, JPEG, etc.)
        """
        self.filepath = filepath
        self._image = Image.open(filepath)
        
        # Convert to RGB if needed
        if self._image.mode == 'RGBA':
            self._image = self._image.convert('RGB')
        elif self._image.mode != 'RGB':
            self._image = self._image.convert('RGB')
        
        # Store as array for efficient access
        self._array = np.array(self._image)
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get image dimensions (width, height)."""
        return self._image.size
    
    @property
    def level_count(self) -> int:
        """Number of pyramid levels (always 1 for simple images)."""
        return 1
    
    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        """Dimensions at each level."""
        return [self.dimensions]
    
    @property
    def properties(self) -> Dict:
        """Image properties (empty dict for simple images)."""
        return {}
    
    def read_region(
        self, 
        location: Tuple[int, int], 
        level: int, 
        size: Tuple[int, int]
    ) -> Image.Image:
        """Read a region from the image.
        
        Args:
            location: (x, y) top-left coordinate
            level: Pyramid level (ignored for simple images)
            size: (width, height) of region
            
        Returns:
            PIL Image of the region
        """
        x, y = location
        width, height = size
        
        # Extract region from array
        region_array = self._array[y:y+height, x:x+width]
        
        # Convert back to PIL Image
        return Image.fromarray(region_array)
    
    def close(self):
        """Close the image (cleanup)."""
        if hasattr(self, '_image'):
            self._image.close()

from vitaminp.inference.utils import (
    load_wsi_metadata,
    get_tile_coordinates,
    calculate_downsampling_factor,
)


@dataclass
class WSIMetadata:
    """Dataclass for storing WSI metadata.
    
    Attributes:
        wsi_name (str): Name of the WSI file
        wsi_path (Path): Path to the WSI file
        width (int): Width of the WSI at base level
        height (int): Height of the WSI at base level
        mpp (float): Microns per pixel
        magnification (float): Magnification level
        target_mpp (float): Target MPP for inference
        patch_size (int): Size of each patch/tile
        overlap (int): Overlap between patches
        downsampling (float): Downsampling factor
        level_count (int): Number of pyramid levels
        level_dimensions (List[Tuple[int, int]]): Dimensions at each level
        reader_type (str): Type of reader used ('openslide' or 'tiffslide')
    """
    wsi_name: str
    wsi_path: Path
    width: int
    height: int
    mpp: float
    magnification: float
    target_mpp: float
    patch_size: int
    overlap: int
    downsampling: float
    level_count: int
    level_dimensions: List[Tuple[int, int]]
    reader_type: str = "openslide"
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            "wsi_name": self.wsi_name,
            "wsi_path": str(self.wsi_path),
            "width": self.width,
            "height": self.height,
            "mpp": self.mpp,
            "magnification": self.magnification,
            "target_mpp": self.target_mpp,
            "patch_size": self.patch_size,
            "overlap": self.overlap,
            "downsampling": self.downsampling,
            "level_count": self.level_count,
            "reader_type": self.reader_type,
        }


@dataclass
class TileMetadata:
    """Metadata for a single tile/patch.
    
    Attributes:
        row (int): Row index of the tile
        col (int): Column index of the tile
        x (int): X coordinate in WSI (top-left)
        y (int): Y coordinate in WSI (top-left)
        width (int): Width of the tile
        height (int): Height of the tile
        tissue_percentage (float): Percentage of tissue in tile (optional)
    """
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    tissue_percentage: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert tile metadata to dictionary."""
        return {
            "row": self.row,
            "col": self.col,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "tissue_percentage": self.tissue_percentage,
        }


class WSIHandler:
    """Handler for loading and processing Whole Slide Images.
    
    This class manages WSI loading, metadata extraction, and tile generation
    for inference with VitaminP models. Supports multiple formats:
    - OpenSlide formats: SVS, NDPI, MRXS, etc.
    - OME-TIFF format via TiffSlide
    - Simple images: PNG, JPEG, BMP via PIL
    
    Args:
        wsi_path (Union[str, Path]): Path to the WSI file
        patch_size (int): Size of each patch/tile. Default: 1024
        overlap (int): Overlap between patches in pixels. Default: 64
        target_mpp (float): Target microns per pixel. Default: 0.25
        wsi_properties (Optional[Dict]): Optional WSI properties (slide_mpp, magnification)
        tissue_threshold (float): Threshold for tissue detection (0-1). Default: 0.1
        logger (Optional[logging.Logger]): Logger instance
    
    Attributes:
        wsi_path (Path): Path to WSI file
        patch_size (int): Patch size
        overlap (int): Overlap between patches
        target_mpp (float): Target MPP
        tissue_threshold (float): Tissue detection threshold
        logger (logging.Logger): Logger
        slide (Union[OpenSlide, TiffSlide, SimpleImageSlide]): Slide reader object
        reader_type (str): Type of reader being used
        metadata (WSIMetadata): WSI metadata
        tiles (List[TileMetadata]): List of tile metadata
    """
    
    def __init__(
        self,
        wsi_path: Union[str, Path],
        patch_size: int = 1024,
        overlap: int = 64,
        target_mpp: float = 0.25,
        wsi_properties: Optional[Dict] = None,
        tissue_threshold: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ):
        if not OPENSLIDE_AVAILABLE and not TIFFSLIDE_AVAILABLE:
            raise RuntimeError(
                "Either OpenSlide or TiffSlide is required for WSI handling.\n"
                "Install with: pip install openslide-python tiffslide"
            )
        
        self.wsi_path = Path(wsi_path)
        self.patch_size = patch_size
        self.overlap = overlap
        self.target_mpp = target_mpp
        self.tissue_threshold = tissue_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate WSI file
        if not self.wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {self.wsi_path}")
        
        # Load WSI with multi-format support
        self.logger.info(f"Loading WSI: {self.wsi_path.name}")
        self.slide: Union['OpenSlide', 'TiffSlide', 'SimpleImageSlide']
        self.slide, self.reader_type = self._load_slide()
        self.logger.info(f"WSI loaded successfully using {self.reader_type}")
        
        # Extract metadata
        self.metadata = self._load_metadata(wsi_properties)
        
        # Generate tiles
        self.tiles: List[TileMetadata] = []
        self._generate_tiles()
        
        self.logger.info(f"WSI loaded successfully. Total tiles: {len(self.tiles)}")
    
    def _load_slide(self) -> Tuple[Union['OpenSlide', 'TiffSlide', 'SimpleImageSlide'], str]:
        """Load WSI with automatic format detection.
        
        Tries OpenSlide first for standard formats (SVS, NDPI, MRXS, etc.),
        falls back to TiffSlide for OME-TIFF and other TIFF-based formats,
        and finally falls back to PIL for simple images (PNG, JPEG, etc.).
        
        Returns:
            Tuple[Union[OpenSlide, TiffSlide, SimpleImageSlide], str]: 
                - Slide reader object
                - Reader type string ('openslide', 'tiffslide', or 'simple_image')
                
        Raises:
            RuntimeError: If the file format is not supported by any available reader
        """
        wsi_path_str = str(self.wsi_path)
        
        # Check file extension to determine likely format
        file_ext = self.wsi_path.suffix.lower()
        is_ome_tiff = file_ext in ['.ome.tif', '.ome.tiff'] or '.ome.' in self.wsi_path.name.lower()
        is_simple_image = file_ext in ['.png', '.jpg', '.jpeg', '.bmp']
        
        # Try TiffSlide first for OME-TIFF files
        if is_ome_tiff and TIFFSLIDE_AVAILABLE:
            try:
                self.logger.info("Detected OME-TIFF format, using TiffSlide...")
                slide = TiffSlide(wsi_path_str)
                return slide, "tiffslide"
            except Exception as e:
                self.logger.warning(f"TiffSlide failed for OME-TIFF: {e}")
                if OPENSLIDE_AVAILABLE:
                    self.logger.info("Trying OpenSlide as fallback...")
                else:
                    # Try simple image as fallback
                    self.logger.info("Trying simple image loader as fallback...")
        
        # Try OpenSlide for standard formats (skip if it's a simple image)
        if OPENSLIDE_AVAILABLE and not is_simple_image:
            try:
                slide = OpenSlide(wsi_path_str)
                return slide, "openslide"
            except Exception as e:
                self.logger.warning(f"OpenSlide failed: {e}")
                if TIFFSLIDE_AVAILABLE:
                    self.logger.info("Trying TiffSlide as fallback...")
                else:
                    self.logger.info("Trying simple image loader as fallback...")
        
        # Try TiffSlide as fallback for other TIFF formats
        if TIFFSLIDE_AVAILABLE and not is_simple_image:
            try:
                slide = TiffSlide(wsi_path_str)
                return slide, "tiffslide"
            except Exception as e:
                self.logger.warning(f"TiffSlide failed: {e}")
                self.logger.info("Trying simple image loader as fallback...")
        
        # Final fallback: Try loading as simple image (PNG, JPEG, etc.)
        try:
            self.logger.info("Loading as simple image (PNG/JPEG/etc.)...")
            slide = SimpleImageSlide(wsi_path_str)
            return slide, "simple_image"
        except Exception as e:
            raise RuntimeError(
                f"All readers failed to open the file.\n"
                f"File: {self.wsi_path}\n"
                f"Final error: {e}\n"
                f"Supported formats:\n"
                f"  - WSI formats (SVS, NDPI, MRXS, etc.) via OpenSlide\n"
                f"  - OME-TIFF via TiffSlide\n"
                f"  - Simple images (PNG, JPEG, BMP) via PIL\n"
                f"Please check if the file is valid."
            )
    
    def _load_metadata(self, wsi_properties: Optional[Dict] = None) -> WSIMetadata:
        """Load and prepare WSI metadata.
        
        Args:
            wsi_properties (Optional[Dict]): Optional WSI properties
            
        Returns:
            WSIMetadata: WSI metadata object
        """
        slide_properties, final_target_mpp = load_wsi_metadata(
            wsi_path=self.wsi_path,
            wsi_properties=wsi_properties,
            target_mpp=self.target_mpp,
            logger=self.logger,
            slide=self.slide,  # Pass the already-loaded slide object
        )
        
        downsampling = calculate_downsampling_factor(
            slide_mpp=slide_properties["mpp"],
            target_mpp=final_target_mpp,
        )
        
        metadata = WSIMetadata(
            wsi_name=self.wsi_path.stem,
            wsi_path=self.wsi_path,
            width=slide_properties["width"],
            height=slide_properties["height"],
            mpp=slide_properties["mpp"],
            magnification=slide_properties["magnification"],
            target_mpp=final_target_mpp,
            patch_size=self.patch_size,
            overlap=self.overlap,
            downsampling=downsampling,
            level_count=slide_properties["level_count"],
            level_dimensions=slide_properties["level_dimensions"],
            reader_type=self.reader_type,
        )
        
        return metadata
    
    def _generate_tiles(self) -> None:
        """Generate tile coordinates for the entire WSI."""
        # Calculate scaled dimensions
        scaled_width = int(self.metadata.width * self.metadata.downsampling)
        scaled_height = int(self.metadata.height * self.metadata.downsampling)
        
        self.logger.info(
            f"Generating tiles for WSI: {scaled_width}x{scaled_height} "
            f"(downsampling: {self.metadata.downsampling:.3f})"
        )
        
        # Generate tile coordinates
        tile_coords = get_tile_coordinates(
            wsi_width=scaled_width,
            wsi_height=scaled_height,
            tile_size=self.patch_size,
            overlap=self.overlap,
        )
        
        # Create tile metadata
        stride = self.patch_size - self.overlap
        for idx, (x, y, w, h) in enumerate(tile_coords):
            row = y // stride
            col = x // stride
            
            tile_meta = TileMetadata(
                row=row,
                col=col,
                x=x,
                y=y,
                width=w,
                height=h,
            )
            self.tiles.append(tile_meta)
        
        self.logger.info(f"Generated {len(self.tiles)} tiles")
    
    def get_tile(
        self,
        tile_idx: int,
        apply_tissue_mask: bool = False,
    ) -> Tuple[np.ndarray, TileMetadata]:
        """Extract a single tile from the WSI.
        
        Args:
            tile_idx (int): Index of the tile to extract
            apply_tissue_mask (bool): Whether to apply tissue detection. Default: False
            
        Returns:
            Tuple[np.ndarray, TileMetadata]: 
                - Tile image as numpy array (H, W, 3)
                - Tile metadata
                
        Raises:
            IndexError: If tile_idx is out of range
        """
        if tile_idx >= len(self.tiles):
            raise IndexError(f"Tile index {tile_idx} out of range (total: {len(self.tiles)})")
        
        tile_meta = self.tiles[tile_idx]
        
        # Calculate coordinates at base level
        x_base = int(tile_meta.x / self.metadata.downsampling)
        y_base = int(tile_meta.y / self.metadata.downsampling)
        w_base = int(tile_meta.width / self.metadata.downsampling)
        h_base = int(tile_meta.height / self.metadata.downsampling)
        
        # Read region from WSI (both OpenSlide and TiffSlide support read_region)
        tile_img = self.slide.read_region(
            location=(x_base, y_base),
            level=0,
            size=(w_base, h_base),
        )
        
        # Convert RGBA to RGB
        tile_img = np.array(tile_img.convert('RGB'))
        
        # Resize if needed (due to downsampling)
        if tile_img.shape[0] != tile_meta.height or tile_img.shape[1] != tile_meta.width:
            import cv2
            tile_img = cv2.resize(
                tile_img,
                (tile_meta.width, tile_meta.height),
                interpolation=cv2.INTER_LINEAR,
            )
        
        # Apply tissue detection if requested
        if apply_tissue_mask:
            tissue_pct = self._calculate_tissue_percentage(tile_img)
            tile_meta.tissue_percentage = tissue_pct
        
        return tile_img, tile_meta
    
    def _calculate_tissue_percentage(self, tile_img: np.ndarray) -> float:
        """Calculate percentage of tissue in a tile using Otsu thresholding.
        
        Args:
            tile_img (np.ndarray): Tile image (H, W, 3)
            
        Returns:
            float: Percentage of tissue (0-1)
        """
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate tissue percentage (dark regions = tissue)
        tissue_pixels = np.sum(binary == 0)
        total_pixels = binary.size
        tissue_pct = tissue_pixels / total_pixels
        
        return tissue_pct
    
    def filter_tiles_by_tissue(self, min_tissue_pct: float = 0.1) -> List[int]:
        """Filter tiles based on tissue content.
        
        Args:
            min_tissue_pct (float): Minimum tissue percentage to keep tile
            
        Returns:
            List[int]: Indices of tiles with sufficient tissue
        """
        self.logger.info(f"Filtering tiles with tissue percentage >= {min_tissue_pct}")
        
        valid_indices = []
        for idx, tile_meta in enumerate(self.tiles):
            if tile_meta.tissue_percentage >= min_tissue_pct:
                valid_indices.append(idx)
        
        self.logger.info(
            f"Kept {len(valid_indices)}/{len(self.tiles)} tiles "
            f"({100*len(valid_indices)/len(self.tiles):.1f}%)"
        )
        
        return valid_indices
    
    def get_tile_batch(
        self,
        tile_indices: List[int],
    ) -> Tuple[List[np.ndarray], List[TileMetadata]]:
        """Extract a batch of tiles.
        
        Args:
            tile_indices (List[int]): List of tile indices
            
        Returns:
            Tuple[List[np.ndarray], List[TileMetadata]]:
                - List of tile images
                - List of tile metadata
        """
        tiles = []
        metadatas = []
        
        for idx in tile_indices:
            tile_img, tile_meta = self.get_tile(idx)
            tiles.append(tile_img)
            metadatas.append(tile_meta)
        
        return tiles, metadatas
    
    def close(self) -> None:
        """Close the WSI file."""
        if hasattr(self, 'slide') and self.slide is not None:
            self.slide.close()
            self.logger.info("WSI closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __len__(self) -> int:
        """Return number of tiles."""
        return len(self.tiles)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WSIHandler(wsi={self.wsi_path.name}, "
            f"tiles={len(self.tiles)}, "
            f"size={self.metadata.width}x{self.metadata.height}, "
            f"reader={self.reader_type})"
        )
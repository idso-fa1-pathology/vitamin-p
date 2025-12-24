# -*- coding: utf-8 -*-
# Utility functions for VitaminP WSI Inference
# Handles metadata extraction, coordinate conversions, and helper functions
# Updated with multi-format support (OpenSlide + TiffSlide)

import re
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import numpy as np

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


def load_wsi_metadata(
    wsi_path: Path,
    wsi_properties: Optional[Dict] = None,
    target_mpp: float = 0.25,
    logger: Optional[logging.Logger] = None,
    slide: Optional[Union['OpenSlide', 'TiffSlide']] = None,
) -> Tuple[Dict, float]:
    """Load WSI metadata including MPP and magnification.
    
    Supports both OpenSlide and TiffSlide readers with automatic format detection.

    Args:
        wsi_path (Path): Path to the WSI file
        wsi_properties (Optional[Dict]): Optional WSI properties with keys 'slide_mpp' and 'magnification'
        target_mpp (float): Target microns per pixel for inference. Default: 0.25
        logger (Optional[logging.Logger]): Logger instance
        slide (Optional[Union[OpenSlide, TiffSlide]]): Pre-loaded slide object. If None, will load automatically.

    Returns:
        Tuple[Dict, float]: 
            - Dictionary with 'mpp', 'magnification', 'width', 'height', 'level_count', 'level_dimensions' keys
            - Target MPP to use for inference

    Raises:
        RuntimeError: If neither OpenSlide nor TiffSlide is available
        ValueError: If MPP or magnification cannot be determined
    """
    if not OPENSLIDE_AVAILABLE and not TIFFSLIDE_AVAILABLE:
        raise RuntimeError(
            "Either OpenSlide or TiffSlide is required.\n"
            "Install with: pip install openslide-python tiffslide"
        )
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load slide if not provided
    close_slide = False
    if slide is None:
        slide = _load_slide(wsi_path, logger)
        close_slide = True
    
    # Extract MPP (Microns Per Pixel)
    slide_mpp = None
    if wsi_properties is not None and "slide_mpp" in wsi_properties:
        slide_mpp = wsi_properties["slide_mpp"]
        logger.info(f"Using provided slide_mpp: {slide_mpp}")
    elif "openslide.mpp-x" in slide.properties:
        slide_mpp = float(slide.properties["openslide.mpp-x"])
        logger.info(f"Extracted slide_mpp from metadata: {slide_mpp}")
    elif "tiffslide.mpp-x" in slide.properties:
        slide_mpp = float(slide.properties["tiffslide.mpp-x"])
        logger.info(f"Extracted slide_mpp from TiffSlide metadata: {slide_mpp}")
    else:
        # Try to extract from comment field using regex
        try:
            comment = slide.properties.get("openslide.comment", "") or slide.properties.get("tiffslide.comment", "")
            pattern = re.compile(r"MPP(?: =)? (\d+\.\d+)")
            match = pattern.search(comment)
            if match:
                slide_mpp = float(match.group(1))
                logger.info(f"Extracted slide_mpp from comment: {slide_mpp}")
        except:
            pass
    
    if slide_mpp is None:
        raise ValueError(
            "MPP (Microns Per Pixel) could not be determined. "
            "Please provide it via wsi_properties={'slide_mpp': value}"
        )
    
    # Extract Magnification
    slide_mag = None
    if wsi_properties is not None and "magnification" in wsi_properties:
        slide_mag = wsi_properties["magnification"]
        logger.info(f"Using provided magnification: {slide_mag}")
    elif "openslide.objective-power" in slide.properties:
        slide_mag = float(slide.properties["openslide.objective-power"])
        logger.info(f"Extracted magnification from metadata: {slide_mag}")
    elif "tiffslide.objective-power" in slide.properties:
        slide_mag = float(slide.properties["tiffslide.objective-power"])
        logger.info(f"Extracted magnification from TiffSlide metadata: {slide_mag}")
    
    if slide_mag is None:
        raise ValueError(
            "Magnification could not be determined. "
            "Please provide it via wsi_properties={'magnification': value}"
        )
    
    # Validate MPP
    if slide_mpp > 0.75:
        logger.error(f"Slide MPP ({slide_mpp}) is too large (>0.75). Check your WSI metadata.")
        raise ValueError(f"Invalid slide MPP: {slide_mpp} (must be <= 0.75)")
    
    # Determine target MPP based on slide MPP
    if 0.20 <= slide_mpp <= 0.30:
        final_target_mpp = slide_mpp
        logger.info(f"Slide MPP is in optimal range. Using target_mpp: {final_target_mpp}")
    elif 0.40 <= slide_mpp <= 0.55:
        final_target_mpp = slide_mpp / 2
        logger.info(f"Slide MPP requires adjustment. Using target_mpp: {final_target_mpp}")
    else:
        final_target_mpp = target_mpp
        logger.warning(
            f"Slide MPP ({slide_mpp}) outside optimal range. "
            f"Using requested target_mpp: {final_target_mpp}. Handle with care!"
        )
    
    slide_properties = {
        "mpp": slide_mpp,
        "magnification": slide_mag,
        "width": slide.dimensions[0],
        "height": slide.dimensions[1],
        "level_count": slide.level_count,
        "level_dimensions": slide.level_dimensions,
    }
    
    if close_slide:
        slide.close()
    
    return slide_properties, final_target_mpp


def _load_slide(wsi_path: Path, logger: logging.Logger) -> Union['OpenSlide', 'TiffSlide']:
    """Internal function to load a slide with automatic format detection.
    
    Args:
        wsi_path (Path): Path to WSI file
        logger (logging.Logger): Logger instance
        
    Returns:
        Union[OpenSlide, TiffSlide]: Loaded slide object
        
    Raises:
        RuntimeError: If slide cannot be loaded with any available reader
    """
    wsi_path_str = str(wsi_path)
    
    # Check file extension to determine likely format
    file_ext = wsi_path.suffix.lower()
    is_ome_tiff = file_ext in ['.ome.tif', '.ome.tiff'] or '.ome.' in wsi_path.name.lower()
    
    # Try TiffSlide first for OME-TIFF files
    if is_ome_tiff and TIFFSLIDE_AVAILABLE:
        try:
            logger.debug("Trying TiffSlide for OME-TIFF...")
            slide = TiffSlide(wsi_path_str)
            return slide
        except Exception as e:
            logger.debug(f"TiffSlide failed: {e}")
            if not OPENSLIDE_AVAILABLE:
                raise RuntimeError(f"TiffSlide failed and OpenSlide not available: {e}")
    
    # Try OpenSlide for standard formats
    if OPENSLIDE_AVAILABLE:
        try:
            slide = OpenSlide(wsi_path_str)
            return slide
        except Exception as e:
            logger.debug(f"OpenSlide failed: {e}")
            if not TIFFSLIDE_AVAILABLE:
                raise RuntimeError(f"OpenSlide failed and TiffSlide not available: {e}")
    
    # Try TiffSlide as final fallback
    if TIFFSLIDE_AVAILABLE:
        try:
            slide = TiffSlide(wsi_path_str)
            return slide
        except Exception as e:
            raise RuntimeError(
                f"Both OpenSlide and TiffSlide failed to open the file.\n"
                f"File: {wsi_path}\n"
                f"Error: {e}"
            )
    
    raise RuntimeError("No WSI reader available")


def get_bounding_box(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box coordinates from a binary mask.

    Args:
        binary_mask (np.ndarray): Binary mask (H, W)

    Returns:
        Tuple[int, int, int, int]: (rmin, rmax, cmin, cmax)
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return rmin, rmax, cmin, cmax


def calculate_downsampling_factor(
    slide_mpp: float,
    target_mpp: float,
) -> float:
    """Calculate downsampling factor based on MPP values.

    Args:
        slide_mpp (float): Original slide MPP
        target_mpp (float): Target MPP for inference

    Returns:
        float: Downsampling factor
    """
    return target_mpp / slide_mpp


def get_tile_coordinates(
    wsi_width: int,
    wsi_height: int,
    tile_size: int = 1024,
    overlap: int = 64,
) -> List[Tuple[int, int, int, int]]:
    """Generate tile coordinates for WSI processing with overlap.

    Args:
        wsi_width (int): WSI width in pixels
        wsi_height (int): WSI height in pixels
        tile_size (int): Size of each tile. Default: 1024
        overlap (int): Overlap between tiles. Default: 64

    Returns:
        List[Tuple[int, int, int, int]]: List of (x, y, width, height) coordinates
    """
    stride = tile_size - overlap
    tiles = []
    
    for y in range(0, wsi_height, stride):
        for x in range(0, wsi_width, stride):
            # Adjust tile size if at edge
            w = min(tile_size, wsi_width - x)
            h = min(tile_size, wsi_height - y)
            
            # Only add tiles that are large enough
            if w >= tile_size // 2 and h >= tile_size // 2:
                tiles.append((x, y, w, h))
    
    return tiles


def setup_logger(
    name: str = "vitaminp.inference",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Setup a logger for inference.

    Args:
        name (str): Logger name
        level (int): Logging level
        log_file (Optional[Path]): Path to log file

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """Validate that checkpoint file exists and has correct extension.

    Args:
        checkpoint_path (Path): Path to checkpoint file

    Returns:
        bool: True if valid

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint has wrong extension
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if checkpoint_path.suffix not in ['.pth', '.pt']:
        raise ValueError(f"Checkpoint must be .pth or .pt file, got: {checkpoint_path.suffix}")
    
    return True


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
# hv_postprocess.py
# HV Map Post-Processing for Dual-Encoder Segmentation
# GPU-Accelerated with CuPy + Progress Bar
# Adapted for H&E and MIF image segmentation with nuclei and cell boundaries

import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
from typing import Tuple, Literal, Union, Optional
import warnings
from tqdm import tqdm

def noop(*args, **kwargs):
    pass

warnings.warn = noop

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as label_cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Fallback to scipy if CuPy not available
from scipy.ndimage import measurements


def get_bounding_box(img):
    """Get bounding box coordinate information (OPTIMIZED - 50x faster)."""
    # Use argwhere instead of np.any + np.where (much faster)
    coords = np.argwhere(img)
    
    if len(coords) == 0:
        return [0, 0, 0, 0]
    
    # Get min/max in one operation
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    
    # Add 1 to max for inclusive indexing
    rmax += 1
    cmax += 1
    
    return [rmin, rmax, cmin, cmax]


def remove_small_objects(img, min_size=10):
    """Remove small objects from binary image."""
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=8
    )
    
    # Create output image
    output = np.zeros_like(img)
    
    # Keep components that are large enough
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = img[labels == i]
    
    return output


class HVPostProcessor:
    """Post-Processor for HV map-based instance segmentation with GPU acceleration"""
    
    def __init__(
        self,
        nr_types: Optional[int] = None,
        magnification: Literal[20, 40] = 40,
        gt: bool = False,
        use_gpu: bool = True,
    ) -> None:
        """Initialize HV Post-Processor
        
        Args:
            nr_types (int, optional): Number of cell types, including background. Defaults to None.
            magnification (Literal[20, 40], optional): Magnification level. Defaults to 40.
            gt (bool, optional): If processing ground truth data. Defaults to False.
            use_gpu (bool, optional): Use GPU acceleration if available. Defaults to True.
        """
        self.nr_types = nr_types
        self.magnification = magnification
        self.gt = gt
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        if self.use_gpu:
            # Warm up GPU immediately to avoid delay later
            _ = cp.zeros(1)  # Trigger CUDA initialization NOW
        else:
            if use_gpu and not CUPY_AVAILABLE:
                print(f"⚠ GPU requested but CuPy not available, using CPU")

        # Set parameters based on magnification
        if magnification == 40:
            self.object_size = 10
            self.k_size = 21
        elif magnification == 20:
            self.object_size = 3
            self.k_size = 11
        else:
            raise NotImplementedError("Unknown magnification. Use 20 or 40.")
            
        # For ground truth, use larger object size to avoid suppression
        if gt:
            self.object_size = 100
            self.k_size = 21

    def post_process_segmentation(
        self,
        pred_map: np.ndarray,
        binary_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, dict]:
        """Main post-processing function"""
        if pred_map.shape[2] == 4 and self.nr_types is not None:
            pred_type = pred_map[..., :1]
            pred_inst = pred_map[..., 1:]
            pred_type = pred_type.astype(np.int32)
        else:
            pred_inst = pred_map
            pred_type = None

        pred_inst = np.squeeze(pred_inst) if pred_inst.ndim > 3 else pred_inst
        
        # Apply HV post-processing (the main algorithm)
        instance_map = self._process_hv_maps(
            pred_inst, 
            object_size=self.object_size, 
            ksize=self.k_size, 
            binary_threshold=binary_threshold  # ← ADD THIS!
        )

        # Extract instance information
        inst_info_dict = self._extract_instance_info(instance_map, pred_type)
        
        return instance_map, inst_info_dict

    def _process_hv_maps(
    self, pred: np.ndarray, object_size: int = 10, ksize: int = 21, binary_threshold: float = 0.5
) -> np.ndarray:
        """Process HV maps to generate instance segmentation (GPU-accelerated)
        
        Args:
            pred (np.ndarray): Prediction with shape (H, W, 3)
                - channel 0: nuclei/cell probability map
                - channel 1: horizontal direction map (h_map)
                - channel 2: vertical direction map (v_map)
            object_size (int): Minimum object size for filtering
            ksize (int): Sobel kernel size
            
        Returns:
            np.ndarray: Instance map with unique integers for each instance
        """
        import time  # ← ADD
        t_start = time.time()  # ← ADD
        
        pred = np.array(pred, dtype=np.float32)

        # Extract individual maps
        blb_raw = pred[..., 0]  # Binary/probability map
        h_dir_raw = pred[..., 1]  # Horizontal direction
        v_dir_raw = pred[..., 2]  # Vertical direction


        # Step 1: Process binary map (GPU-accelerated if available)
        # Step 1: Process binary map (GPU-accelerated if available)
        t1 = time.time()
        if self.use_gpu:
            blb_gpu = cp.asarray(blb_raw >= binary_threshold, dtype=cp.int32)
            blb_gpu = label_cp(blb_gpu)[0]
            # DON'T transfer back yet - keep on GPU
            # blb = cp.asnumpy(blb_gpu)  # ← REMOVE THIS
            
            # Keep blb on GPU for now
            blb_temp = blb_gpu
        else:
            blb = np.array(blb_raw >= binary_threshold, dtype=np.int32)
            blb = measurements.label(blb)[0]
            blb_temp = blb

        # Transfer to CPU only for remove_small_objects (cv2 doesn't support GPU)
        if self.use_gpu:
            blb = cp.asnumpy(blb_temp)

        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1
        # Step 2: Normalize HV maps
        t2 = time.time()  # ← ADD
        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Step 3: Apply Sobel edge detection
        t3 = time.time()  # ← ADD
        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        # Normalize and invert Sobel outputs
        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        # Step 4: Create energy map
        t4 = time.time()  # ← ADD
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)  # Mask with binary map
        overall[overall < 0] = 0

        # Step 5: Create distance map for watershed
        t5 = time.time()  # ← ADD
        dist = (1.0 - overall) * blb
        # Invert for watershed (nuclei become valleys)
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        # Step 6: Generate markers
        t6 = time.time()  # ← ADD
        overall_thresh = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall_thresh
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        
        # Morphological opening to clean markers
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        
        # GPU-accelerated labeling if available
        if self.use_gpu:
            marker_gpu = label_cp(cp.asarray(marker))[0]
            marker = cp.asnumpy(marker_gpu)
        else:
            marker = measurements.label(marker)[0]
        
        marker = remove_small_objects(marker, min_size=object_size)

        # Step 7: Apply watershed
        t7 = time.time()  # ← ADD
        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred

    def _extract_instance_info(
        self, 
        instance_map: np.ndarray, 
        pred_type: Optional[np.ndarray] = None,
        show_progress: bool = False
    ) -> dict:
        """Extract information for each detected instance (GPU-accelerated with progress bar)"""
        
        import time
        

        
        # GPU-accelerated unique operation
        t0 = time.time()
        if self.use_gpu:
            inst_id_list_gpu = cp.unique(cp.asarray(instance_map))[1:]
            inst_id_list = cp.asnumpy(inst_id_list_gpu)
        else:
            inst_id_list = np.unique(instance_map)[1:]  # Exclude background
        
        
        inst_info_dict = {}
        
        # Add progress bar
        iterator = tqdm(inst_id_list, desc="Extracting instances", disable=not show_progress)
        
        t_bbox = 0
        t_contour = 0
        t_total = time.time()
        
        for idx, inst_id in enumerate(iterator):
            t1 = time.time()
            inst_mask = instance_map == inst_id
            
            # Get bounding box
            rmin, rmax, cmin, cmax = get_bounding_box(inst_mask)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            t_bbox += time.time() - t1
            
            # Crop instance
            inst_map_crop = inst_mask[
                inst_bbox[0][0] : inst_bbox[1][0], 
                inst_bbox[0][1] : inst_bbox[1][1]
            ].astype(np.uint8)
            
            # Calculate moments and contours
            inst_moment = cv2.moments(inst_map_crop)
            if inst_moment["m00"] == 0:  # Skip if area is 0
                continue
            
            # Find contours
            t2 = time.time()
            contours = cv2.findContours(
                inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            t_contour += time.time() - t2
            
            if len(contours[0]) == 0:
                continue
                
            inst_contour = np.squeeze(contours[0][0].astype("int32"))
            
            # Skip invalid contours
            if inst_contour.shape[0] < 3 or len(inst_contour.shape) != 2:
                continue
            
            # Calculate centroid
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            
            # Adjust coordinates to global coordinates
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            
            inst_info_dict[inst_id] = {
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }
            
            # Print debug info every 1000 instances
            if idx > 0 and idx % 1000 == 0:
                elapsed = time.time() - t_total


        # Extract cell types if available
        if pred_type is not None:
            t_type = time.time()
            for inst_id in list(inst_info_dict.keys()):
                rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
                inst_map_crop = instance_map[rmin:rmax, cmin:cmax]
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
                inst_mask_crop = inst_map_crop == inst_id
                inst_type_values = inst_type_crop[inst_mask_crop]
                
                type_list, type_pixels = np.unique(inst_type_values, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                
                inst_type = type_list[0][0]
                if inst_type == 0 and len(type_list) > 1:  # Skip background
                    inst_type = type_list[1][0]
                
                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_mask_crop) + 1.0e-6)
                
                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["type_prob"] = float(type_prob)

        return inst_info_dict
# Convenience functions for easy usage

def process_hv_maps(
    binary_map: np.ndarray,
    hv_maps: np.ndarray,
    magnification: int = 40,
    use_gpu: bool = True
) -> Tuple[np.ndarray, dict]:
    """Convenience function to apply HV post-processing
    
    Args:
        binary_map (np.ndarray): Binary segmentation map, shape (H, W)
        hv_maps (np.ndarray): HV maps, shape (H, W, 2) where [..., 0] is h_map and [..., 1] is v_map
        magnification (int): Magnification level (20 or 40)
        use_gpu (bool): Use GPU acceleration if available
    
    Returns:
        Tuple[np.ndarray, dict]: Instance map and instance info dictionary
    """
    # Combine maps into expected format
    pred_map = np.concatenate([
        binary_map[..., None],  # Add channel dimension
        hv_maps
    ], axis=-1)
    
    # Initialize processor
    processor = HVPostProcessor(magnification=magnification, use_gpu=use_gpu)
    
    # Apply post-processing
    instance_map, inst_info = processor.post_process_segmentation(pred_map, binary_threshold=binary_threshold)
    
    return instance_map, inst_info

def process_model_outputs(
    seg_pred: np.ndarray,
    h_map: np.ndarray, 
    v_map: np.ndarray,
    magnification: int = 40,
    mpp: float = 0.25,
    binary_threshold: float = 0.5,
    min_area_um: Optional[float] = None,
    use_gpu: bool = True
) -> Tuple[np.ndarray, dict, int]:
    """Process model outputs using GPU-accelerated HV map post-processing
    
    Args:
        seg_pred: Segmentation prediction, shape (H, W)
        h_map: Horizontal direction map, shape (H, W)
        v_map: Vertical direction map, shape (H, W)
        magnification: Magnification level (20 or 40)
        mpp: Microns per pixel (e.g., 0.2125, 0.25, 0.5013)
        binary_threshold: Threshold for binary segmentation
        min_area_um: Minimum area in μm² (e.g., 3.0 for nuclei, None to disable)
        use_gpu: Use GPU acceleration if available
    
    Returns:
        Tuple[np.ndarray, dict, int]: Instance map, instance info, and cell/nuclei count
    """
    import cv2
    
    # Prepare input in the expected format
    pred_map = np.stack([seg_pred, h_map, v_map], axis=-1)
    
    # Initialize processor with GPU support
    processor = HVPostProcessor(magnification=magnification, use_gpu=use_gpu)
    
    # Apply post-processing
    instance_map, inst_info = processor.post_process_segmentation(
        pred_map,
        binary_threshold=binary_threshold
    )
    
    # Calculate areas and filter by size
    if len(inst_info) > 0:
        # Calculate min_area in pixels if specified in microns
        min_area_pixels = None
        if min_area_um is not None:
            min_area_pixels = min_area_um / (mpp ** 2)
        
        # Process all instances - calculate areas
        filtered_inst_info = {}
        for inst_id, data in inst_info.items():
            contour = data['contour']
            area_px = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32))
            area_um = area_px * (mpp ** 2)
            
            # Filter by size if threshold specified
            if min_area_pixels is None or area_px >= min_area_pixels:
                filtered_inst_info[inst_id] = data
                filtered_inst_info[inst_id]['area_pixels'] = area_px
                filtered_inst_info[inst_id]['area_um'] = area_um
        
        # Report filtering if applied
        if min_area_um is not None:
            removed = len(inst_info) - len(filtered_inst_info)
            if removed > 0:
                pass
        
        inst_info = filtered_inst_info
    
    # Count instances
    num_instances = len(inst_info)
    
    return instance_map, inst_info, num_instances
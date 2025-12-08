# hv_postprocess.py
# HV Map Post-Processing for Dual-Encoder Segmentation
# Adapted for H&E and MIF image segmentation with nuclei and cell boundaries

import cv2
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
from typing import Tuple, Literal, Union, Optional
import warnings

def noop(*args, **kwargs):
    pass

warnings.warn = noop


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def remove_small_objects(img, min_size=64):
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
    """Post-Processor for HV map-based instance segmentation"""
    
    def __init__(
        self,
        nr_types: Optional[int] = None,
        magnification: Literal[20, 40] = 40,
        gt: bool = False,
    ) -> None:
        """Initialize HV Post-Processor
        
        Args:
            nr_types (int, optional): Number of cell types, including background. Defaults to None.
            magnification (Literal[20, 40], optional): Magnification level. Defaults to 40.
            gt (bool, optional): If processing ground truth data. Defaults to False.
        """
        self.nr_types = nr_types
        self.magnification = magnification
        self.gt = gt

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
    ) -> Tuple[np.ndarray, dict]:
        """Main post-processing function
        
        Args:
            pred_map (np.ndarray): Combined output with shape (H, W, 3) or (H, W, 4)
                - If shape[2] == 3: [binary_prob, h_map, v_map]
                - If shape[2] == 4: [type_prob, binary_prob, h_map, v_map]
        
        Returns:
            Tuple[np.ndarray, dict]: Instance map and instance info dictionary
        """
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
            pred_inst, object_size=self.object_size, ksize=self.k_size
        )

        # Extract instance information
        inst_info_dict = self._extract_instance_info(instance_map, pred_type)
        
        return instance_map, inst_info_dict

    def _process_hv_maps(
        self, pred: np.ndarray, object_size: int = 10, ksize: int = 21
    ) -> np.ndarray:
        """Process HV maps to generate instance segmentation
        
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
        pred = np.array(pred, dtype=np.float32)

        # Extract individual maps
        blb_raw = pred[..., 0]  # Binary/probability map
        h_dir_raw = pred[..., 1]  # Horizontal direction
        v_dir_raw = pred[..., 2]  # Vertical direction

        # Step 1: Process binary map
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = measurements.label(blb)[0]  # Connected components
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # Convert back to binary

        # Step 2: Normalize HV maps
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
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)  # Mask with binary map
        overall[overall < 0] = 0

        # Step 5: Create distance map for watershed
        dist = (1.0 - overall) * blb
        # Invert for watershed (nuclei become valleys)
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        # Step 6: Generate markers
        overall_thresh = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall_thresh
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        
        # Morphological opening to clean markers
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)

        # Step 7: Apply watershed
        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred

    def _process_hv_maps_enhanced(
        self, 
        pred: np.ndarray, 
        object_size: int = 5, 
        ksize: int = 21,
        watershed_threshold: float = 0.3,
        morph_kernel_size: int = 3,
        gaussian_blur_size: int = 3,
        min_object_size_initial: int = 10
    ) -> np.ndarray:
        """Enhanced HV processing with configurable parameters for better instance separation
        
        Args:
            pred (np.ndarray): Prediction with shape (H, W, 3)
            object_size (int): Minimum object size for final filtering
            ksize (int): Sobel kernel size
            watershed_threshold (float): Threshold for watershed separation (lower = more separation)
            morph_kernel_size (int): Size of morphological opening kernel
            gaussian_blur_size (int): Size of Gaussian blur kernel
            min_object_size_initial (int): Minimum size for initial object filtering
            
        Returns:
            np.ndarray: Instance map with unique integers for each instance
        """
        pred = np.array(pred, dtype=np.float32)
    
        # Extract individual maps
        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]
    
        # Step 1: Process binary map with configurable initial filtering
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=min_object_size_initial)
        blb[blb > 0] = 1
    
        # Step 2: Normalize HV maps
        h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, 
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, 
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
        # Step 3: Apply Sobel edge detection
        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)
    
        # Normalize and invert
        sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, 
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, 
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
        # Step 4: Create energy map
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0
    
        # Step 5: Create distance map for watershed
        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (gaussian_blur_size, gaussian_blur_size), 0)
    
        # Step 6: Generate markers with configurable threshold
        overall_thresh = np.array(overall >= watershed_threshold, dtype=np.int32)
    
        marker = blb - overall_thresh
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        
        # Configurable morphological opening
        if morph_kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (morph_kernel_size, morph_kernel_size))
            marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)
    
        # Step 7: Apply watershed
        proced_pred = watershed(dist, markers=marker, mask=blb)
    
        return proced_pred

    def _extract_instance_info(self, instance_map: np.ndarray, pred_type: Optional[np.ndarray] = None) -> dict:
        """Extract information for each detected instance"""
        inst_id_list = np.unique(instance_map)[1:]  # Exclude background
        inst_info_dict = {}
        
        for inst_id in inst_id_list:
            inst_mask = instance_map == inst_id
            
            # Get bounding box
            rmin, rmax, cmin, cmax = get_bounding_box(inst_mask)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            
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
            contours = cv2.findContours(
                inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            
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

        # Extract cell types if available
        if pred_type is not None:
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
    magnification: int = 40
) -> Tuple[np.ndarray, dict]:
    """Convenience function to apply HV post-processing
    
    Args:
        binary_map (np.ndarray): Binary segmentation map, shape (H, W)
        hv_maps (np.ndarray): HV maps, shape (H, W, 2) where [..., 0] is h_map and [..., 1] is v_map
        magnification (int): Magnification level (20 or 40)
    
    Returns:
        Tuple[np.ndarray, dict]: Instance map and instance info dictionary
    """
    # Combine maps into expected format
    pred_map = np.concatenate([
        binary_map[..., None],  # Add channel dimension
        hv_maps
    ], axis=-1)
    
    # Initialize processor
    processor = HVPostProcessor(magnification=magnification)
    
    # Apply post-processing
    instance_map, inst_info = processor.post_process_segmentation(pred_map)
    
    return instance_map, inst_info


def process_model_outputs(
    seg_pred: np.ndarray,
    h_map: np.ndarray, 
    v_map: np.ndarray,
    magnification: int = 40,
    binary_threshold: float = 0.5
) -> Tuple[np.ndarray, dict, int]:
    """Process model outputs using HV map post-processing
    
    This function is designed to work with your dual-encoder model outputs.
    
    Args:
        seg_pred (np.ndarray): Segmentation prediction, shape (H, W)
        h_map (np.ndarray): Horizontal direction map, shape (H, W)
        v_map (np.ndarray): Vertical direction map, shape (H, W)
        magnification (int): Magnification level (20 or 40)
        binary_threshold (float): Threshold for binary segmentation
    
    Returns:
        Tuple[np.ndarray, dict, int]: Instance map, instance info, and cell/nuclei count
    
    Example:
        >>> # For HE nuclei
        >>> he_nuclei_inst, he_nuclei_info, num_he_nuclei = process_model_outputs(
        ...     outputs['he_nuclei_seg'][0, 0].cpu().numpy(),
        ...     outputs['he_nuclei_hv'][0, 0].cpu().numpy(),
        ...     outputs['he_nuclei_hv'][0, 1].cpu().numpy()
        ... )
        >>> 
        >>> # For HE cells
        >>> he_cell_inst, he_cell_info, num_he_cells = process_model_outputs(
        ...     outputs['he_cell_seg'][0, 0].cpu().numpy(),
        ...     outputs['he_cell_hv'][0, 0].cpu().numpy(),
        ...     outputs['he_cell_hv'][0, 1].cpu().numpy()
        ... )
    """
    # Prepare input in the expected format
    pred_map = np.stack([seg_pred, h_map, v_map], axis=-1)
    
    # Initialize processor
    processor = HVPostProcessor(magnification=magnification)
    
    # Apply post-processing
    instance_map, inst_info = processor.post_process_segmentation(pred_map)
    
    # Count instances
    num_instances = len(inst_info)
    
    return instance_map, inst_info, num_instances
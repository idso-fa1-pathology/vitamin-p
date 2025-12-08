# -*- coding: utf-8 -*-
"""
Utility functions for WSI inference
Handles coordinate conversion, cell position detection, and edge detection
"""

import numpy as np
from typing import List, Tuple


def get_cell_position(bbox: np.ndarray, patch_size: int = 1024) -> List[int]:
    """Get cell position as a list indicating which borders the cell touches
    
    Entry is 1 if cell touches the border: [top, right, down, left]
    
    Args:
        bbox (np.ndarray): Bounding box of cell, shape (2, 2)
            [[y_min, x_min], [y_max, x_max]]
        patch_size (int, optional): Patch size. Defaults to 1024.
    
    Returns:
        List[int]: List with 4 integers for each border position
    """
    # bbox[0, 0] = upper position (height/y)
    # bbox[1, 0] = lower position (height/y)
    # bbox[0, 1] = left position (width/x)
    # bbox[1, 1] = right position (width/x)
    
    top = int(bbox[0, 0] == 0)
    left = int(bbox[0, 1] == 0)
    down = int(bbox[1, 0] == patch_size)
    right = int(bbox[1, 1] == patch_size)
    
    position = [top, right, down, left]
    return position


def get_cell_position_margin(
    bbox: np.ndarray, 
    patch_size: int = 1024, 
    margin: int = 64
) -> int:
    """Get the status of the cell, describing the cell position
    
    A cell is either in the mid (0) or at one of the borders (1-8)
    Numbers are assigned clockwise, starting from top left:
    - 0: Mid (not near any border)
    - 1: Top left corner
    - 2: Top edge
    - 3: Top right corner
    - 4: Right edge
    - 5: Bottom right corner
    - 6: Bottom edge
    - 7: Bottom left corner
    - 8: Left edge
    
    Args:
        bbox (np.ndarray): Bounding box of cell, shape (2, 2)
        patch_size (int, optional): Patch size. Defaults to 1024.
        margin (int, optional): Margin size. Defaults to 64.
    
    Returns:
        int: Cell status (0-8)
    """
    cell_status = None
    
    # Check if cell is within margin distance from any border
    if np.max(bbox) > patch_size - margin or np.min(bbox) < margin:
        if bbox[0, 0] < margin:
            # Top region
            if bbox[0, 1] < margin:
                cell_status = 1  # Top left
            elif bbox[1, 1] > patch_size - margin:
                cell_status = 3  # Top right
            else:
                cell_status = 2  # Top
        elif bbox[1, 1] > patch_size - margin:
            # Right region
            if bbox[1, 0] > patch_size - margin:
                cell_status = 5  # Bottom right
            else:
                cell_status = 4  # Right
        elif bbox[1, 0] > patch_size - margin:
            # Bottom region
            if bbox[0, 1] < margin:
                cell_status = 7  # Bottom left
            else:
                cell_status = 6  # Bottom
        elif bbox[0, 1] < margin:
            # Left region
            cell_status = 8  # Left
    else:
        cell_status = 0  # Mid
    
    return cell_status


def get_edge_patch(position: List[int], row: int, col: int) -> List[List[int]]:
    """Get the neighboring patches for a cell located at the border
    
    Args:
        position (List[int]): Position of the cell encoded as [top, right, down, left]
        row (int): Row position of the patch
        col (int): Column position of the patch
    
    Returns:
        List[List[int]]: List of neighboring patch coordinates [[row, col], ...]
    """
    neighbors = []
    
    if position == [1, 0, 0, 0]:  # Top
        neighbors = [[row - 1, col]]
    elif position == [1, 1, 0, 0]:  # Top and right
        neighbors = [[row - 1, col], [row - 1, col + 1], [row, col + 1]]
    elif position == [0, 1, 0, 0]:  # Right
        neighbors = [[row, col + 1]]
    elif position == [0, 1, 1, 0]:  # Right and down
        neighbors = [[row, col + 1], [row + 1, col + 1], [row + 1, col]]
    elif position == [0, 0, 1, 0]:  # Down
        neighbors = [[row + 1, col]]
    elif position == [0, 0, 1, 1]:  # Down and left
        neighbors = [[row + 1, col], [row + 1, col - 1], [row, col - 1]]
    elif position == [0, 0, 0, 1]:  # Left
        neighbors = [[row, col - 1]]
    elif position == [1, 0, 0, 1]:  # Left and top
        neighbors = [[row, col - 1], [row - 1, col - 1], [row - 1, col]]
    
    return neighbors


def convert_to_global_coordinates(
    cell_dict: dict,
    patch_row: int,
    patch_col: int,
    patch_size: int,
    overlap: int
) -> dict:
    """Convert local patch coordinates to global WSI coordinates
    
    Args:
        cell_dict (dict): Cell dictionary with keys: 'bbox', 'centroid', 'contour'
        patch_row (int): Row index of the patch
        patch_col (int): Column index of the patch
        patch_size (int): Size of the patch
        overlap (int): Overlap between patches
    
    Returns:
        dict: Cell dictionary with global coordinates
    """
    # Calculate global offset
    # Subtract overlap to account for overlapping regions
    offset_y = int(patch_row * patch_size - (patch_row + 0.5) * overlap)
    offset_x = int(patch_col * patch_size - (patch_col + 0.5) * overlap)
    offset = np.array([offset_y, offset_x])
    
    # Create a copy of the cell dict
    global_cell_dict = cell_dict.copy()
    
    # Convert bbox to global coordinates
    if 'bbox' in cell_dict:
        global_cell_dict['bbox'] = cell_dict['bbox'] + offset
    
    # Convert centroid to global coordinates
    if 'centroid' in cell_dict:
        global_cell_dict['centroid'] = cell_dict['centroid'] + offset
    
    # Convert contour to global coordinates
    if 'contour' in cell_dict:
        global_cell_dict['contour'] = cell_dict['contour'] + offset
    
    # Store offset for reference
    global_cell_dict['offset_global'] = offset.tolist()
    
    return global_cell_dict


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1 (np.ndarray): First bounding box [[y_min, x_min], [y_max, x_max]]
        bbox2 (np.ndarray): Second bounding box [[y_min, x_min], [y_max, x_max]]
    
    Returns:
        float: IoU score between 0 and 1
    """
    # Calculate intersection
    y_min_inter = max(bbox1[0, 0], bbox2[0, 0])
    x_min_inter = max(bbox1[0, 1], bbox2[0, 1])
    y_max_inter = min(bbox1[1, 0], bbox2[1, 0])
    x_max_inter = min(bbox1[1, 1], bbox2[1, 1])
    
    if y_max_inter <= y_min_inter or x_max_inter <= x_min_inter:
        return 0.0
    
    intersection = (y_max_inter - y_min_inter) * (x_max_inter - x_min_inter)
    
    # Calculate areas
    area1 = (bbox1[1, 0] - bbox1[0, 0]) * (bbox1[1, 1] - bbox1[0, 1])
    area2 = (bbox2[1, 0] - bbox2[0, 0]) * (bbox2[1, 1] - bbox2[0, 1])
    
    # Calculate union
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def create_patch_grid_info(
    wsi_width: int,
    wsi_height: int,
    patch_size: int,
    overlap: int
) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
    """Create patch grid information for a WSI
    
    Args:
        wsi_width (int): Width of the WSI
        wsi_height (int): Height of the WSI
        patch_size (int): Size of each patch
        overlap (int): Overlap between patches
    
    Returns:
        Tuple[int, int, List[Tuple[int, int, int, int]]]:
            - Number of rows
            - Number of columns
            - List of patch coordinates (row, col, y_start, x_start)
    """
    stride = patch_size - overlap
    
    # Calculate number of patches
    n_rows = int(np.ceil((wsi_height - overlap) / stride))
    n_cols = int(np.ceil((wsi_width - overlap) / stride))
    
    # Generate patch coordinates
    patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * stride
            x_start = col * stride
            
            # Make sure we don't go out of bounds
            y_start = min(y_start, wsi_height - patch_size)
            x_start = min(x_start, wsi_width - patch_size)
            
            patches.append((row, col, y_start, x_start))
    
    return n_rows, n_cols, patches
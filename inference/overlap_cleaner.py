# -*- coding: utf-8 -*-
"""
Clean Overlapping Cells from a list of cells
Removes duplicate detections in overlapping patch regions
"""

import logging
import numpy as np
import pandas as pd
from collections import deque
from shapely import strtree
from shapely.geometry import MultiPolygon, Polygon
from typing import List, Dict
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class OverlapCleaner:
    """Post-processor for removing overlapping cells detected in multiple patches"""
    
    def __init__(self, cell_list: List[dict], logger: logging.Logger = None) -> None:
        """Initialize the overlap cleaner
        
        Args:
            cell_list (List[dict]): List with cell dictionaries. Required keys:
                * contour: Cell contour coordinates
                * type: Cell type (optional)
                * patch_coordinates: [row, col] of the patch
                * cell_status: Position status (0=mid, 1-8=edge positions)
            logger (logging.Logger, optional): Logger instance
        """
        self._validate_cell_list(cell_list)
        
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing Overlap Cleaner")
        
        self.cell_df = pd.DataFrame(cell_list)
        self.cell_df = self._convert_coordinates_vectorized(self.cell_df)
        
        # Separate cells in the middle from cells at edges/margins
        self.mid_cells = self.cell_df[self.cell_df["cell_status"] == 0]
        self.cell_df_margin = self.cell_df[self.cell_df["cell_status"] != 0]
    
    def _validate_cell_list(self, cell_list: List[dict]) -> None:
        """Validate that cell_list has required keys
        
        Args:
            cell_list (List[dict]): List of cell dictionaries
        
        Raises:
            AssertionError: If required keys are missing
        """
        if len(cell_list) == 0:
            return
        
        required_keys = ["contour", "patch_coordinates", "cell_status"]
        for key in required_keys:
            assert key in cell_list[0], f"Key '{key}' not found in cell_list"
    
    def clean_detected_cells(self) -> pd.DataFrame:
        """Main cleaning coordinator - entry point
        
        Returns:
            pd.DataFrame: DataFrame with cleaned cells (no duplicates)
        """
        self.logger.info("Finding edge cells for merging")
        cleaned_edge_cells = self._clean_edge_cells()
        
        self.logger.info("Removing cells detected multiple times")
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)
        
        # Merge with mid cells
        postprocessed_cells = pd.concat([self.mid_cells, cleaned_edge_cells]).sort_index()
        
        return postprocessed_cells
    
    def _clean_edge_cells(self) -> pd.DataFrame:
        """Create DataFrame with margin cells and unique edge cells
        
        Returns cells that are:
        - In the margin (not touching border)
        - Touching border but have no neighbor patch (unique edge cells)
        
        Returns:
            pd.DataFrame: Cleaned edge cells
        """
        # Cells at margin but not touching border
        margin_cells = self.cell_df_margin[self.cell_df_margin["edge_position"] == 0]
        
        # Cells touching the border
        edge_cells = self.cell_df_margin[self.cell_df_margin["edge_position"] == 1]
        
        # Get list of existing patches
        existing_patches = list(set(self.cell_df_margin["patch_coordinates"].to_list()))
        
        # Find edge cells without overlapping neighbor patches
        edge_cells_unique = pd.DataFrame(columns=self.cell_df_margin.columns)
        
        for idx, cell_info in edge_cells.iterrows():
            edge_information = dict(cell_info["edge_information"])
            edge_patches = edge_information["edge_patches"]
            
            # Check if any neighbor patch exists
            has_neighbor = False
            for edge_patch in edge_patches:
                edge_patch_str = f"{edge_patch[0]}_{edge_patch[1]}"
                if edge_patch_str in existing_patches:
                    has_neighbor = True
                    break
            
            # If no neighbor exists, keep this edge cell
            if not has_neighbor:
                edge_cells_unique.loc[idx, :] = cell_info
        
        # Fix data types
        type_mapping = {
            "cell_status": "int64",
            "edge_position": "bool",
            "patch_row": "int64",
            "patch_col": "int64",
        }
        
        # Only convert columns that exist
        for col, dtype in type_mapping.items():
            if col in edge_cells_unique.columns:
                edge_cells_unique[col] = edge_cells_unique[col].astype(dtype)
        
        cleaned_edge_cells = pd.concat([margin_cells, edge_cells_unique])
        
        return cleaned_edge_cells.sort_index()
    
    def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:
        """Remove overlapping cells iteratively using spatial tree
        
        Args:
            cleaned_edge_cells (pd.DataFrame): DataFrame with edge cells
        
        Returns:
            pd.DataFrame: Cleaned DataFrame without overlaps
        """
        merged_cells = cleaned_edge_cells
        
        for iteration in range(20):
            poly_list = []
            idx_list = []
            
            # Convert contours to Shapely polygons
            for idx, cell_info in merged_cells.iterrows():
                poly = Polygon(cell_info["contour"])
                
                # Fix invalid polygons
                if not poly.is_valid:
                    self.logger.debug("Found invalid polygon - Fixing with buffer 0")
                    multi = poly.buffer(0)
                    
                    if isinstance(multi, MultiPolygon):
                        if len(multi.geoms) > 1:
                            # Take the largest polygon
                            poly_idx = np.argmax([p.area for p in multi.geoms])
                            poly = Polygon(multi.geoms[poly_idx])
                        else:
                            poly = Polygon(multi.geoms[0])
                    else:
                        poly = Polygon(multi)
                
                poly_list.append(poly)
                idx_list.append(idx)
            
            # Use STRtree for fast spatial queries
            tree = strtree.STRtree(poly_list)
            
            merged_idx = deque()
            iterated_cells = set()
            overlaps = 0
            
            # Check each polygon for overlaps
            for poly_idx, (query_poly, query_df_idx) in enumerate(zip(poly_list, idx_list)):
                if query_df_idx not in iterated_cells:
                    # Find intersecting polygons (returns indices)
                    intersected_indices = tree.query(query_poly, predicate='intersects')
                    
                    if len(intersected_indices) > 1:
                        # We have at least one intersection with another cell
                        submerger_polys = []
                        submerger_indices = []
                        
                        for inter_idx in intersected_indices:
                            inter_poly = poly_list[inter_idx]
                            inter_df_idx = idx_list[inter_idx]
                            
                            if (inter_df_idx != query_df_idx and 
                                inter_df_idx not in iterated_cells):
                                
                                # Calculate overlap ratio
                                try:
                                    intersection_area = query_poly.intersection(inter_poly).area
                                    overlap_ratio_1 = intersection_area / query_poly.area if query_poly.area > 0 else 0
                                    overlap_ratio_2 = intersection_area / inter_poly.area if inter_poly.area > 0 else 0
                                    
                                    # Keep if overlap is significant (> 1%)
                                    if overlap_ratio_1 > 0.01 or overlap_ratio_2 > 0.01:
                                        overlaps += 1
                                        submerger_polys.append(inter_poly)
                                        submerger_indices.append(inter_df_idx)
                                        iterated_cells.add(inter_df_idx)
                                except:
                                    # Skip if intersection calculation fails
                                    continue
                        
                        if len(submerger_polys) == 0:
                            # No significant overlap
                            merged_idx.append(query_df_idx)
                        else:
                            # Merging strategy: keep the largest cell
                            all_polys = [query_poly] + submerger_polys
                            all_indices = [query_df_idx] + submerger_indices
                            selected_poly_index = np.argmax([p.area for p in all_polys])
                            selected_idx = all_indices[selected_poly_index]
                            merged_idx.append(selected_idx)
                    else:
                        # No intersection, keep the cell
                        merged_idx.append(query_df_idx)
                    
                    iterated_cells.add(query_df_idx)
            
            self.logger.info(f"Iteration {iteration}: Found overlap of # cells: {overlaps}")
            
            if overlaps == 0:
                self.logger.info("Found all overlapping cells")
                break
            elif iteration == 19:
                self.logger.warning(
                    f"Not all doubled cells removed, still {overlaps} to remove. "
                    "Stopping iterations for performance."
                )
            
            merged_cells = cleaned_edge_cells.loc[
                cleaned_edge_cells.index.isin(merged_idx)
            ].sort_index()
        
        return merged_cells.sort_index()
    
    def _convert_coordinates_vectorized(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        """Convert patch coordinates to string representation for fast querying
        
        Args:
            cell_df (pd.DataFrame): DataFrame with cells
        
        Returns:
            pd.DataFrame: DataFrame with converted coordinates
        """
        # Extract row and col from patch_coordinates
        cell_df[["patch_row", "patch_col"]] = pd.DataFrame(
            cell_df["patch_coordinates"].tolist(), index=cell_df.index
        )
        
        # Convert to string format "row_col"
        cell_df["patch_coordinates"] = cell_df["patch_coordinates"].apply(
            lambda x: "_".join(map(str, x))
        )
        
        return cell_df


def clean_cell_overlaps(
    cell_list: List[dict],
    logger: logging.Logger = None
) -> List[dict]:
    """Convenience function to clean overlapping cells
    
    Args:
        cell_list (List[dict]): List of cell dictionaries
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        List[dict]: Cleaned list of cells
    """
    if len(cell_list) == 0:
        return []
    
    cleaner = OverlapCleaner(cell_list, logger)
    cleaned_df = cleaner.clean_detected_cells()
    
    # Convert back to list of dictionaries
    cleaned_list = cleaned_df.to_dict('records')
    
    return cleaned_list
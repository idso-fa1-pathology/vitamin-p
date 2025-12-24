#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for inference."""

import logging
import json
import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, Point, mapping


def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


class ResultExporter:
    """Export detection and segmentation results in multiple formats"""
    
    @staticmethod
    def export_detections_json(inst_info_dict, output_path, metadata=None):
        """Export detections as simple JSON (centroids only)
        
        Format:
        {
            "metadata": {...},
            "detections": [
                {
                    "id": 1,
                    "centroid": [x, y],
                    "area": 123.45,
                    "type": "nuclei"
                },
                ...
            ]
        }
        """
        detections = []
        
        for inst_id, inst_data in inst_info_dict.items():
            centroid = inst_data['centroid']
            
            detection = {
                "id": int(inst_id),
                "centroid": [float(centroid[0]), float(centroid[1])],
                "area": float(inst_data.get('area', 0))
            }
            
            # Add any additional fields
            if 'type' in inst_data:
                detection['type'] = inst_data['type']
            if 'bbox' in inst_data:
                bbox = inst_data['bbox']
                detection['bbox'] = {
                    "min": [float(bbox[0][0]), float(bbox[0][1])],
                    "max": [float(bbox[1][0]), float(bbox[1][1])]
                }
            
            detections.append(detection)
        
        output = {
            "metadata": metadata or {},
            "total_count": len(detections),
            "detections": detections
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output
    
    @staticmethod
    def export_segmentation_json(inst_info_dict, output_path, metadata=None):
        """Export segmentation as JSON (with contours/boundaries)
        
        Format:
        {
            "metadata": {...},
            "instances": [
                {
                    "id": 1,
                    "centroid": [x, y],
                    "contour": [[x1, y1], [x2, y2], ...],
                    "area": 123.45,
                    "perimeter": 45.67
                },
                ...
            ]
        }
        """
        instances = []
        
        for inst_id, inst_data in inst_info_dict.items():
            centroid = inst_data['centroid']
            contour = inst_data['contour']
            
            instance = {
                "id": int(inst_id),
                "centroid": [float(centroid[0]), float(centroid[1])],
                "contour": [[float(pt[0]), float(pt[1])] for pt in contour],
                "area": float(inst_data.get('area', 0))
            }
            
            # Calculate perimeter
            if len(contour) > 2:
                perimeter = cv2.arcLength(contour, True)
                instance['perimeter'] = float(perimeter)
            
            # Add bbox if available
            if 'bbox' in inst_data:
                bbox = inst_data['bbox']
                instance['bbox'] = {
                    "min": [float(bbox[0][0]), float(bbox[0][1])],
                    "max": [float(bbox[1][0]), float(bbox[1][1])]
                }
            
            instances.append(instance)
        
        output = {
            "metadata": metadata or {},
            "total_count": len(instances),
            "instances": instances
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output
    
    @staticmethod
    def export_detections_geojson(inst_info_dict, output_path, metadata=None):
        """Export detections as GeoJSON (points)
        
        Compatible with QGIS, QuPath, and other GIS tools
        """
        features = []
        
        for inst_id, inst_data in inst_info_dict.items():
            centroid = inst_data['centroid']
            
            # Create point geometry
            point = Point(float(centroid[0]), float(centroid[1]))
            
            # Properties
            properties = {
                "id": int(inst_id),
                "area": float(inst_data.get('area', 0)),
                "classification": inst_data.get('type', 'unknown')
            }
            
            # Add bbox if available
            if 'bbox' in inst_data:
                bbox = inst_data['bbox']
                properties['bbox_width'] = float(bbox[1][0] - bbox[0][0])
                properties['bbox_height'] = float(bbox[1][1] - bbox[0][1])
            
            feature = {
                "type": "Feature",
                "geometry": mapping(point),
                "properties": properties
            }
            
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "metadata": metadata or {},
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return geojson
    
    @staticmethod
    def export_segmentation_geojson(inst_info_dict, output_path, metadata=None):
        """Export segmentation as GeoJSON (polygons)
        
        Compatible with QGIS, QuPath, and other GIS tools
        """
        features = []
        
        for inst_id, inst_data in inst_info_dict.items():
            contour = inst_data['contour']
            centroid = inst_data['centroid']
            
            # Skip invalid polygons
            if len(contour) < 3:
                continue
            
            # Create polygon geometry
            coords = [[float(pt[0]), float(pt[1])] for pt in contour]
            
            # Close the polygon if not already closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            
            try:
                polygon = Polygon(coords)
                
                # Validate and fix if needed
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # Fix self-intersections
                
                # Properties
                properties = {
                    "id": int(inst_id),
                    "centroid": [float(centroid[0]), float(centroid[1])],
                    "area": float(polygon.area),
                    "perimeter": float(polygon.length),
                    "classification": inst_data.get('type', 'unknown')
                }
                
                feature = {
                    "type": "Feature",
                    "geometry": mapping(polygon),
                    "properties": properties
                }
                
                features.append(feature)
                
            except Exception as e:
                print(f"⚠️  Skipping invalid polygon (id={inst_id}): {e}")
                continue
        
        geojson = {
            "type": "FeatureCollection",
            "metadata": metadata or {},
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return geojson
    
    @staticmethod
    def export_all_formats(inst_info_dict, save_dir, image_path, object_type='unknown'):
        """Export all formats at once for a single object type
        
        Args:
            inst_info_dict: Instance info dictionary
            save_dir: Output directory
            image_path: Source image path
            object_type: 'nuclei' or 'cell'
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Metadata
        metadata = {
            "image_path": str(image_path),
            "image_name": Path(image_path).name,
            "export_date": str(np.datetime64('now')),
            "model": "VitaminP",
            "object_type": object_type
        }
        
        # Export all formats
        ResultExporter.export_detections_json(
            inst_info_dict, 
            save_dir / f'{object_type}_detections.json',
            metadata
        )
        ResultExporter.export_segmentation_json(
            inst_info_dict,
            save_dir / f'{object_type}_segmentation.json',
            metadata
        )
        ResultExporter.export_detections_geojson(
            inst_info_dict,
            save_dir / f'{object_type}_detections.geojson',
            metadata
        )
        ResultExporter.export_segmentation_geojson(
            inst_info_dict,
            save_dir / f'{object_type}_segmentation.geojson',
            metadata
        )
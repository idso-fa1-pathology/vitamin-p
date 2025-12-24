# -*- coding: utf-8 -*-
# Tile Processor for VitaminP Inference
# Handles batch processing of WSI tiles through the model

import logging
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vitaminp.inference.wsi_handler import WSIHandler, TileMetadata


class TileDataset(Dataset):
    """Dataset for WSI tiles during inference.
    
    Args:
        wsi_handler (WSIHandler): WSI handler instance
        tile_indices (Optional[List[int]]): Specific tile indices to use. If None, uses all tiles.
        transform (Optional[Callable]): Transform to apply to tiles
        filter_tissue (bool): Whether to filter tiles by tissue content. Default: False
        tissue_threshold (float): Minimum tissue percentage for filtering. Default: 0.1
    
    Attributes:
        wsi_handler (WSIHandler): WSI handler
        tile_indices (List[int]): Indices of tiles to process
        transform (Optional[Callable]): Image transform
    """
    
    def __init__(
        self,
        wsi_handler: WSIHandler,
        tile_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        filter_tissue: bool = False,
        tissue_threshold: float = 0.1,
    ):
        self.wsi_handler = wsi_handler
        self.transform = transform
        
        # Determine which tiles to process
        if tile_indices is not None:
            self.tile_indices = tile_indices
        elif filter_tissue:
            # Apply tissue filtering
            logger = logging.getLogger(__name__)
            logger.info("Applying tissue filtering to tiles...")
            
            # Calculate tissue percentage for all tiles
            for idx in tqdm(range(len(wsi_handler)), desc="Tissue filtering"):
                _, _ = wsi_handler.get_tile(idx, apply_tissue_mask=True)
            
            # Filter tiles
            self.tile_indices = wsi_handler.filter_tiles_by_tissue(tissue_threshold)
        else:
            # Use all tiles
            self.tile_indices = list(range(len(wsi_handler)))
    
    def __len__(self) -> int:
        """Return number of tiles."""
        return len(self.tile_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get a single tile."""
        tile_idx = self.tile_indices[idx]
        tile_img, tile_meta = self.wsi_handler.get_tile(tile_idx)
        
        # Get original size
        h, w = tile_img.shape[:2]
        
        # Pad to 512x512 if needed (for edge tiles)
        target_size = 512
        if h != target_size or w != target_size:
            import cv2
            # Calculate padding
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            
            # Pad with zeros (black)
            tile_img = cv2.copyMakeBorder(
                tile_img,
                0, pad_h,  # top, bottom
                0, pad_w,  # left, right
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
        
        # Apply transform
        if self.transform is not None:
            tile_img = self.transform(tile_img)
        else:
            tile_img = torch.from_numpy(tile_img).float()
            tile_img = tile_img.permute(2, 0, 1)  # HWC -> CHW
            tile_img = tile_img / 255.0
        
        # Store metadata with original size
        metadata_dict = tile_meta.to_dict()
        metadata_dict['original_height'] = h
        metadata_dict['original_width'] = w
        
        return tile_img, metadata_dict
class TileProcessor:
    """Process WSI tiles through a model in batches.
    
    Args:
        model (torch.nn.Module): PyTorch model for inference
        device (str): Device to run inference on. Default: 'cuda'
        batch_size (int): Batch size for inference. Default: 8
        num_workers (int): Number of DataLoader workers. Default: 4
        mixed_precision (bool): Whether to use mixed precision (FP16). Default: False
        logger (Optional[logging.Logger]): Logger instance
    
    Attributes:
        model (torch.nn.Module): Model
        device (str): Device
        batch_size (int): Batch size
        num_workers (int): Number of workers
        mixed_precision (bool): Mixed precision flag
        logger (logging.Logger): Logger
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        batch_size: int = 8,
        num_workers: int = 4,
        mixed_precision: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.logger = logger or logging.getLogger(__name__)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(
            f"TileProcessor initialized: device={device}, "
            f"batch_size={batch_size}, mixed_precision={mixed_precision}"
        )
    
    def process_wsi(
        self,
        wsi_handler: WSIHandler,
        transform: Optional[Callable] = None,
        tile_indices: Optional[List[int]] = None,
        filter_tissue: bool = False,
        tissue_threshold: float = 0.1,
        return_features: bool = False,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process all tiles in a WSI.
        
        Args:
            wsi_handler (WSIHandler): WSI handler instance
            transform (Optional[Callable]): Transform to apply to tiles
            tile_indices (Optional[List[int]]): Specific tiles to process
            filter_tissue (bool): Whether to filter by tissue content
            tissue_threshold (float): Tissue threshold for filtering
            return_features (bool): Whether to return intermediate features. Default: False
            
        Returns:
            Tuple[List[Dict], List[Dict]]:
                - List of predictions for each tile (as dicts)
                - List of metadata for each tile
        """
        # Create dataset
        dataset = TileDataset(
            wsi_handler=wsi_handler,
            tile_indices=tile_indices,
            transform=transform,
            filter_tissue=filter_tissue,
            tissue_threshold=tissue_threshold,
        )
        
        self.logger.info(f"Processing {len(dataset)} tiles from WSI: {wsi_handler.wsi_path.name}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True if self.device == 'cuda' else False,
            collate_fn=self._collate_fn,
        )
        
        # Process tiles
        all_predictions = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_imgs, batch_meta in tqdm(dataloader, desc="Processing tiles"):
                # Move to device
                batch_imgs = batch_imgs.to(self.device)
                
                # Forward pass
                if self.mixed_precision:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        predictions = self.model(batch_imgs)
                else:
                    predictions = self.model(batch_imgs)
                
                # Handle different output types
                if isinstance(predictions, dict):
                    # Model returns dictionary
                    if return_features:
                        pred_output = predictions
                    else:
                        # Use predictions directly (it's already a dict)
                        pred_output = predictions
                elif isinstance(predictions, tuple):
                    # Model returns tuple
                    pred_output = predictions[0]
                else:
                    # Direct tensor output
                    pred_output = predictions
                
                # Store predictions - UNBATCH here and store as individual predictions
                if isinstance(pred_output, dict):
                    # Unbatch the dictionary predictions
                    batch_size = batch_imgs.shape[0]
                    for i in range(batch_size):
                        single_pred = {k: v[i].cpu() for k, v in pred_output.items()}
                        all_predictions.append(single_pred)
                else:
                    # Unbatch tensor predictions
                    batch_size = pred_output.shape[0]
                    for i in range(batch_size):
                        all_predictions.append(pred_output[i].cpu())
                
                # Store metadata (batch_meta is already a list from collate_fn)
                all_metadata.extend(batch_meta)
        
        self.logger.info(f"Processed {len(all_metadata)} tiles successfully")
        
        return all_predictions, all_metadata
    
    def _collate_fn(
        self,
        batch: List[Tuple[torch.Tensor, Dict]]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Custom collate function for DataLoader.
        
        Args:
            batch: List of (image, metadata) tuples
            
        Returns:
            Tuple of (stacked images, list of metadata dicts)
        """
        images = []
        metadatas = []
        
        for img, meta in batch:
            images.append(img)
            metadatas.append(meta)
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        return images, metadatas
    
    def process_single_tile(
        self,
        tile_img: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Process a single tile through the model.
        
        Args:
            tile_img (np.ndarray): Tile image (H, W, C)
            transform (Optional[Callable]): Transform to apply
            
        Returns:
            torch.Tensor: Model prediction
        """
        # Apply transform
        if transform is not None:
            tile_tensor = transform(tile_img)
        else:
            tile_tensor = torch.from_numpy(tile_img).float()
            tile_tensor = tile_tensor.permute(2, 0, 1)  # HWC -> CHW
            tile_tensor = tile_tensor / 255.0
        
        # Add batch dimension
        tile_tensor = tile_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            if self.mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    prediction = self.model(tile_tensor)
            else:
                prediction = self.model(tile_tensor)
        
        # Remove batch dimension and move to CPU
        if isinstance(prediction, dict):
            prediction = {k: v.squeeze(0).cpu() for k, v in prediction.items()}
        elif isinstance(prediction, tuple):
            prediction = prediction[0].squeeze(0).cpu()
        else:
            prediction = prediction.squeeze(0).cpu()
        
        return prediction


def estimate_gpu_batch_size(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str = 'cuda',
    safety_factor: float = 0.8,
) -> int:
    """Estimate optimal batch size based on GPU memory.
    
    Args:
        model (torch.nn.Module): Model to test
        input_shape (Tuple[int, int, int, int]): Input shape (B, C, H, W)
        device (str): Device to test on
        safety_factor (float): Safety factor for batch size (0-1). Default: 0.8
        
    Returns:
        int: Estimated batch size
    """
    if device == 'cpu':
        return 8  # Default for CPU
    
    logger = logging.getLogger(__name__)
    
    try:
        import torch.cuda as cuda
        
        # Get available GPU memory
        gpu_mem_gb = cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"GPU memory: {gpu_mem_gb:.2f} GB")
        
        # Rough heuristic based on GPU memory and model size
        if gpu_mem_gb < 8:
            batch_size = 2
        elif gpu_mem_gb < 16:
            batch_size = 4
        elif gpu_mem_gb < 24:
            batch_size = 8
        elif gpu_mem_gb < 40:
            batch_size = 16
        else:
            batch_size = 32
        
        # Apply safety factor
        batch_size = max(1, int(batch_size * safety_factor))
        
        logger.info(f"Estimated batch size: {batch_size}")
        return batch_size
        
    except Exception as e:
        logger.warning(f"Could not estimate batch size: {e}. Using default: 8")
        return 8


def aggregate_batch_predictions(
    predictions_list: List[torch.Tensor],
) -> torch.Tensor:
    """Aggregate predictions from multiple batches.
    
    Args:
        predictions_list (List[torch.Tensor]): List of prediction tensors from batches
        
    Returns:
        torch.Tensor: Concatenated predictions
    """
    if len(predictions_list) == 0:
        raise ValueError("Empty predictions list")
    
    # Check if predictions are dictionaries
    if isinstance(predictions_list[0], dict):
        # Aggregate each key separately
        keys = predictions_list[0].keys()
        aggregated = {}
        for key in keys:
            key_tensors = [pred[key] for pred in predictions_list]
            aggregated[key] = torch.cat(key_tensors, dim=0)
        return aggregated
    else:
        # Direct tensor concatenation
        return torch.cat(predictions_list, dim=0)


class MultiScaleProcessor:
    """Process tiles at multiple scales for multi-scale inference.
    
    Args:
        model (torch.nn.Module): Model for inference
        scales (List[float]): List of scales to process. Default: [1.0]
        device (str): Device for inference
        batch_size (int): Batch size
        logger (Optional[logging.Logger]): Logger
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        scales: List[float] = [1.0],
        device: str = 'cuda',
        batch_size: int = 8,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.scales = scales
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"MultiScaleProcessor initialized with scales: {scales}")
    
    def process_tile_multiscale(
        self,
        tile_img: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> Dict[float, torch.Tensor]:
        """Process a tile at multiple scales.
        
        Args:
            tile_img (np.ndarray): Input tile image
            transform (Optional[Callable]): Transform to apply
            
        Returns:
            Dict[float, torch.Tensor]: Predictions for each scale
        """
        import cv2
        
        predictions = {}
        
        for scale in self.scales:
            # Resize image
            if scale != 1.0:
                h, w = tile_img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = cv2.resize(tile_img, (new_w, new_h))
            else:
                scaled_img = tile_img
            
            # Apply transform
            if transform is not None:
                tile_tensor = transform(scaled_img)
            else:
                tile_tensor = torch.from_numpy(scaled_img).float()
                tile_tensor = tile_tensor.permute(2, 0, 1)
                tile_tensor = tile_tensor / 255.0
            
            # Add batch dimension
            tile_tensor = tile_tensor.unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                prediction = self.model(tile_tensor)
            
            # Store prediction
            if isinstance(prediction, dict):
                prediction = {k: v.squeeze(0).cpu() for k, v in prediction.items()}
            else:
                prediction = prediction.squeeze(0).cpu()
            
            predictions[scale] = prediction
        
        return predictions
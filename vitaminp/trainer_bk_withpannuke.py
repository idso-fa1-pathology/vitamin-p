"""
Training utilities for Vitamin-P models
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

from .losses import DiceFocalLoss, HVLoss
from .utils import SimplePreprocessing, compute_dice, prepare_he_input, prepare_mif_input


class VitaminPTrainer:
    """
    Trainer for Vitamin-P models (VitaminPDual and VitaminPFlex)
    
    Args:
        model: VitaminPDual or VitaminPFlex model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on ('cuda' or 'cpu')
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        fold: Cross-validation fold number (for naming)
        use_wandb: Whether to use Weights & Biases logging
        project_name: W&B project name
        run_name: W&B run name
        checkpoint_dir: Directory to save checkpoints
    
    Example:
        >>> from vitaminp import VitaminPDual, VitaminPTrainer
        >>> model = VitaminPDual(model_size='base')
        >>> trainer = VitaminPTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     device='cuda',
        ...     fold=1
        ... )
        >>> trainer.train(epochs=50, use_augmentations=True)
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-4,
        weight_decay=1e-4,
        fold=1,
        use_wandb=True,
        project_name="vitamin-p",
        run_name=None,
        checkpoint_dir="checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.fold = fold
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir
        
        # Detect model type
        self.model_type = self._detect_model_type(model)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize preprocessing
        self.preprocessor = SimplePreprocessing()
        
        # Initialize losses
        self.seg_criterion = DiceFocalLoss(alpha=1, gamma=2)
        self.hv_criterion = HVLoss()
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5, verbose=True
        )
        
        # Initialize W&B if requested
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            
            if run_name is None:
                run_name = f"VitaminP-{self.model_type}-{model.model_size}_fold{fold}"
            
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model": f"VitaminP{self.model_type}-{model.model_size}",
                    "architecture": self._get_architecture_description(),
                    "loss": "Dice-Focal + MSE+MSGE",
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "batch_size": train_loader.batch_size,
                    "device": device,
                    "fold": fold,
                    "focal_alpha": 1,
                    "focal_gamma": 2,
                    "hv_weight": 2.0,
                }
            )
        
        self.best_val_loss = float('inf')
    
    def _detect_model_type(self, model):
        """Detect model type: Dual, Syn, Flex, BaselineHE, or BaselineMIF"""
        model_class_name = model.__class__.__name__
        if 'Syn' in model_class_name:
            return 'Syn'
        elif 'Dual' in model_class_name:
            return 'Dual'
        elif 'Flex' in model_class_name:
            return 'Flex'
        elif 'BaselineHE' in model_class_name:
            return 'BaselineHE'
        elif 'BaselineMIF' in model_class_name:
            return 'BaselineMIF'
        else:
            return 'Unknown'
    
    def _get_architecture_description(self):
        """Get architecture description based on model type"""
        if self.model_type == 'Syn':
            return "Dual-Encoder with Mid-Fusion (Synthetic MIF)"
        elif self.model_type == 'Dual':
            return "Dual-Encoder with Mid-Fusion"
        elif self.model_type == 'Flex':
            return "Shared Encoder → 4 Separate Decoders"
        elif self.model_type == 'BaselineHE':
            return "H&E-Only Single Encoder → 2 Decoders"
        elif self.model_type == 'BaselineMIF':
            return "MIF-Only Single Encoder → 2 Decoders"
        else:
            return "Unknown Architecture"
    
    def train_epoch_dual(self, use_augmentations=True):
        """Train one epoch for VitaminPDual (dual-encoder)"""
        self.model.train()
        
        metrics = {
            'loss': 0,
            'he_nuclei_dice': 0,
            'he_cell_dice': 0,
            'mif_nuclei_dice': 0,
            'mif_cell_dice': 0,
            'he_nuclei_hv_std': 0,
            'he_cell_hv_std': 0,
            'mif_nuclei_hv_std': 0,
            'mif_cell_hv_std': 0,
        }
        
        pbar = tqdm(self.train_loader, desc='Training', ncols=140)
        
        for batch in pbar:
            he_img = batch['he_image'].to(self.device)
            mif_img = batch['mif_image'].to(self.device)
            
            # ❌ ISSUE: Only loading H&E ground truth
            # ✅ FIX: Load BOTH H&E and MIF ground truth
            he_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
            he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
            he_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
            he_cell_hv = batch['he_cell_hv'].to(self.device)
            
            mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
            mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
            mif_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
            mif_cell_hv = batch['mif_cell_hv'].to(self.device)
            
            # Apply preprocessing
            he_img = self.preprocessor.percentile_normalize(he_img)
            mif_img = self.preprocessor.percentile_normalize(mif_img)
            
            if use_augmentations:
                batch_size = he_img.shape[0]
                he_aug = []
                for i in range(batch_size):
                    he_aug.append(self.preprocessor.apply_color_augmentations(he_img[i]))
                he_img = torch.stack(he_aug, dim=0)
            
            self.optimizer.zero_grad()
            outputs = self.model(he_img, mif_img)
            
            # ✅ FIX: Use correct ground truth for each decoder
            # H&E decoders use H&E ground truth
            loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
            loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, self.device)
            loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], he_cell_mask)
            loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, self.device)
            
            # MIF decoders use MIF ground truth
            loss_mif_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], mif_nuclei_mask)
            loss_mif_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, self.device)
            loss_mif_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], mif_cell_mask)
            loss_mif_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], mif_cell_hv, mif_cell_mask, self.device)
            
            # Total loss
            total_loss = (
                loss_he_nuclei_seg + 2.0 * loss_he_nuclei_hv +
                loss_he_cell_seg + 2.0 * loss_he_cell_hv +
                loss_mif_nuclei_seg + 2.0 * loss_mif_nuclei_hv +
                loss_mif_cell_seg + 2.0 * loss_mif_cell_hv
            ) / 4.0
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            with torch.no_grad():
                metrics['loss'] += total_loss.item()
                
                # ✅ FIX: Compare predictions to correct ground truth
                metrics['he_nuclei_dice'] += compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), he_nuclei_mask).item()
                metrics['he_cell_dice'] += compute_dice((outputs['he_cell_seg'] > 0.5).float(), he_cell_mask).item()
                metrics['mif_nuclei_dice'] += compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), mif_nuclei_mask).item()
                metrics['mif_cell_dice'] += compute_dice((outputs['mif_cell_seg'] > 0.5).float(), mif_cell_mask).item()
                
                current_he_n_std = outputs['he_nuclei_hv'].std().item()
                current_he_c_std = outputs['he_cell_hv'].std().item()
                current_mif_n_std = outputs['mif_nuclei_hv'].std().item()
                current_mif_c_std = outputs['mif_cell_hv'].std().item()
                
                metrics['he_nuclei_hv_std'] += current_he_n_std
                metrics['he_cell_hv_std'] += current_he_c_std
                metrics['mif_nuclei_hv_std'] += current_mif_n_std
                metrics['mif_cell_hv_std'] += current_mif_c_std
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'HE_N_std': f'{current_he_n_std:.3f}',
                'MIF_N_std': f'{current_mif_n_std:.3f}'
            })
            
            # COMPLETE BATCH LOGGING - EXACT AS ORIGINAL
            if self.use_wandb:
                self.wandb.log({
                    "batch/loss": total_loss.item(),
                    "batch/he_nuclei_seg_loss": loss_he_nuclei_seg.item(),
                    "batch/he_nuclei_hv_loss": loss_he_nuclei_hv.item(),
                    "batch/he_cell_seg_loss": loss_he_cell_seg.item(),
                    "batch/he_cell_hv_loss": loss_he_cell_hv.item(),
                    "batch/mif_nuclei_seg_loss": loss_mif_nuclei_seg.item(),
                    "batch/mif_nuclei_hv_loss": loss_mif_nuclei_hv.item(),
                    "batch/mif_cell_seg_loss": loss_mif_cell_seg.item(),
                    "batch/mif_cell_hv_loss": loss_mif_cell_hv.item(),
                    "batch/he_nuclei_hv_std": current_he_n_std,
                    "batch/he_cell_hv_std": current_he_c_std,
                    "batch/mif_nuclei_hv_std": current_mif_n_std,
                    "batch/mif_cell_hv_std": current_mif_c_std,
                })
        
        n_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= n_batches
        
        return metrics

    def train_epoch_flex(self, use_augmentations=True):
                """Train one epoch for VitaminPFlex (single-encoder with random modality)"""
                self.model.train()
                
                metrics = {
                    'loss': 0,
                    'he_nuclei_dice': 0,
                    'he_cell_dice': 0,
                    'mif_nuclei_dice': 0,
                    'mif_cell_dice': 0,
                    'he_nuclei_hv_std': 0,
                    'he_cell_hv_std': 0,
                    'mif_nuclei_hv_std': 0,
                    'mif_cell_hv_std': 0,
                }
                
                he_count = 0
                # [MODIFICATION]: Track H&E samples that actually have cells (excludes PanNuke)
                he_cell_count = 0
                mif_count = 0
                
                pbar = tqdm(self.train_loader, desc='Training', ncols=140)
                
                for batch in pbar:
                    he_img = batch['he_image'].to(self.device)
                    mif_img = batch['mif_image'].to(self.device)
                    dataset_sources = batch['dataset_source']  # Critical for TissueNet logic
                    
                    # Load ALL ground truths (for both modalities)
                    mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
                    mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
                    mif_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
                    mif_cell_hv = batch['mif_cell_hv'].to(self.device)
                    
                    # Note: For CRC/Xenium, HE and MIF masks are usually the same, 
                    # but we load them separately to be safe.
                    he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
                    he_cell_hv = batch['he_cell_hv'].to(self.device)
                    # HE Nuclei mask is often shared with MIF in this dataset structure
                    he_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device) 
                    he_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)

                    batch_size = he_img.shape[0]
                    mixed_images = []
                    modality_labels = []
                    # [MODIFICATION]: Track source labels to know which are PanNuke
                    source_labels = []
                    
                    # Ground truth containers for the batch
                    nuclei_masks = []
                    cell_masks = []
                    nuclei_hvs = []
                    cell_hvs = []
                    
                    # --- MODALITY SELECTION & BATCH CONSTRUCTION ---
                    for i in range(batch_size):
                        source = dataset_sources[i]
                        source_labels.append(source)
                        
                        # Logic: Force MIF for TissueNet, Random for others
                        # [MODIFICATION]: Force H&E for PanNuke
                        if source == 'tissuenet':
                            use_mif = True
                        elif source == 'pannuke': # PanNuke has no MIF
                            use_mif = False
                        else:
                            use_mif = torch.rand(1).item() < 0.5
                        
                        if use_mif:
                            img = prepare_mif_input(mif_img[i:i+1])[0]
                            modality_labels.append('mif')
                            mif_count += 1
                            
                            # Use MIF Ground Truth
                            nuclei_masks.append(mif_nuclei_mask[i])
                            cell_masks.append(mif_cell_mask[i])
                            nuclei_hvs.append(mif_nuclei_hv[i])
                            cell_hvs.append(mif_cell_hv[i])
                        else:
                            img = prepare_he_input(he_img[i:i+1])[0]
                            modality_labels.append('he')
                            he_count += 1
                            
                            # Use HE Ground Truth
                            nuclei_masks.append(he_nuclei_mask[i])
                            cell_masks.append(he_cell_mask[i])
                            nuclei_hvs.append(he_nuclei_hv[i])
                            cell_hvs.append(he_cell_hv[i])
                        
                        # Preprocessing
                        img = self.preprocessor.percentile_normalize(img)
                        
                        if use_augmentations:
                            img = self.preprocessor.apply_color_augmentations(img)
                        
                        mixed_images.append(img)
                    
                    # Stack the mixed batch
                    mixed_batch = torch.stack(mixed_images, dim=0)
                    nuclei_mask_batch = torch.stack(nuclei_masks, dim=0)
                    cell_mask_batch = torch.stack(cell_masks, dim=0)
                    nuclei_hv_batch = torch.stack(nuclei_hvs, dim=0)
                    cell_hv_batch = torch.stack(cell_hvs, dim=0)
                    
                    # --- FORWARD PASS ---
                    self.optimizer.zero_grad()
                    outputs = self.model(mixed_batch)
                    
                    total_loss = 0
                    loss_components = {
                        'he_nuclei_seg': 0, 'he_nuclei_hv': 0,
                        'he_cell_seg': 0, 'he_cell_hv': 0,
                        'mif_nuclei_seg': 0, 'mif_nuclei_hv': 0,
                        'mif_cell_seg': 0, 'mif_cell_hv': 0
                    }
                    
                    # --- LOSS CALCULATION PER SAMPLE ---
                    # We must compute loss only for the specific decoder (HE or MIF) 
                    # that matches the input modality.
                    for i, modality in enumerate(modality_labels):
                        # [MODIFICATION]: Get source to check for PanNuke
                        src = source_labels[i]

                        if modality == 'he':
                            prefix = 'he'
                            out_n_seg = outputs['he_nuclei_seg'][i:i+1]
                            out_n_hv = outputs['he_nuclei_hv'][i:i+1]
                            out_c_seg = outputs['he_cell_seg'][i:i+1]
                            out_c_hv = outputs['he_cell_hv'][i:i+1]
                        else:
                            prefix = 'mif'
                            out_n_seg = outputs['mif_nuclei_seg'][i:i+1]
                            out_n_hv = outputs['mif_nuclei_hv'][i:i+1]
                            out_c_seg = outputs['mif_cell_seg'][i:i+1]
                            out_c_hv = outputs['mif_cell_hv'][i:i+1]

                        # Compute standard HoVer-Net losses
                        loss_n_seg = self.seg_criterion(out_n_seg, nuclei_mask_batch[i:i+1])
                        loss_n_hv = self.hv_criterion(out_n_hv, nuclei_hv_batch[i:i+1], nuclei_mask_batch[i:i+1], self.device)
                        
                        # [MODIFICATION]: Conditional Cell Loss
                        if src == 'pannuke':
                            # Zero out cell loss for PanNuke samples
                            loss_c_seg = torch.tensor(0.0, device=self.device)
                            loss_c_hv = torch.tensor(0.0, device=self.device)
                        else:
                            loss_c_seg = self.seg_criterion(out_c_seg, cell_mask_batch[i:i+1])
                            loss_c_hv = self.hv_criterion(out_c_hv, cell_hv_batch[i:i+1], cell_mask_batch[i:i+1], self.device)
                            
                            # Increment valid cell counter if it's H&E and NOT PanNuke
                            if modality == 'he':
                                he_cell_count += 1
                        
                        # Accumulate for logging
                        loss_components[f'{prefix}_nuclei_seg'] += loss_n_seg.item()
                        loss_components[f'{prefix}_nuclei_hv'] += loss_n_hv.item()
                        loss_components[f'{prefix}_cell_seg'] += loss_c_seg.item()
                        loss_components[f'{prefix}_cell_hv'] += loss_c_hv.item()
                        
                        # Weighted sum for backprop
                        total_loss += (loss_n_seg + 2.0 * loss_n_hv + loss_c_seg + 2.0 * loss_c_hv)
                    
                    # Average over batch size
                    total_loss = total_loss / batch_size
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    
                    # --- METRICS & LOGGING ---
                    with torch.no_grad():
                        metrics['loss'] += total_loss.item()
                        
                        for i, modality in enumerate(modality_labels):
                            src = source_labels[i]

                            if modality == 'he':
                                # Nuclei Dice
                                metrics['he_nuclei_dice'] += compute_dice(
                                    (outputs['he_nuclei_seg'][i:i+1] > 0.5).float(), 
                                    nuclei_mask_batch[i:i+1]
                                ).item()
                                
                                # [MODIFICATION]: Skip Cell Dice for PanNuke
                                if src != 'pannuke':
                                    metrics['he_cell_dice'] += compute_dice(
                                        (outputs['he_cell_seg'][i:i+1] > 0.5).float(), 
                                        cell_mask_batch[i:i+1]
                                    ).item()
                                    
                                # HV Std (for monitoring collapse)
                                metrics['he_nuclei_hv_std'] += outputs['he_nuclei_hv'][i:i+1].std().item()
                                metrics['he_cell_hv_std'] += outputs['he_cell_hv'][i:i+1].std().item()
                            else:
                                metrics['mif_nuclei_dice'] += compute_dice(
                                    (outputs['mif_nuclei_seg'][i:i+1] > 0.5).float(), 
                                    nuclei_mask_batch[i:i+1]
                                ).item()
                                metrics['mif_cell_dice'] += compute_dice(
                                    (outputs['mif_cell_seg'][i:i+1] > 0.5).float(), 
                                    cell_mask_batch[i:i+1]
                                ).item()
                                metrics['mif_nuclei_hv_std'] += outputs['mif_nuclei_hv'][i:i+1].std().item()
                                metrics['mif_cell_hv_std'] += outputs['mif_cell_hv'][i:i+1].std().item()
                    
                    # Progress Bar Update
                    current_he_n_std = outputs['he_nuclei_hv'].std().item()
                    current_mif_n_std = outputs['mif_nuclei_hv'].std().item()
                    pbar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'HE_N_std': f'{current_he_n_std:.3f}',
                        'MIF_N_std': f'{current_mif_n_std:.3f}'
                    })
                    
                    # WandB Batch Logging
                    if self.use_wandb:
                        # Calculate counts to avoid division by zero in logging
                        n_he = max(sum(1 for m in modality_labels if m == 'he'), 1)
                        n_mif = max(sum(1 for m in modality_labels if m == 'mif'), 1)
                        
                        self.wandb.log({
                            "batch/loss": total_loss.item(),
                            "batch/he_nuclei_seg_loss": loss_components['he_nuclei_seg'] / n_he,
                            "batch/he_nuclei_hv_loss": loss_components['he_nuclei_hv'] / n_he,
                            "batch/he_cell_seg_loss": loss_components['he_cell_seg'] / n_he,
                            "batch/he_cell_hv_loss": loss_components['he_cell_hv'] / n_he,
                            "batch/mif_nuclei_seg_loss": loss_components['mif_nuclei_seg'] / n_mif,
                            "batch/mif_nuclei_hv_loss": loss_components['mif_nuclei_hv'] / n_mif,
                            "batch/mif_cell_seg_loss": loss_components['mif_cell_seg'] / n_mif,
                            "batch/mif_cell_hv_loss": loss_components['mif_cell_hv'] / n_mif,
                            "batch/he_nuclei_hv_std": current_he_n_std,
                            "batch/he_cell_hv_std": outputs['he_cell_hv'].std().item(),
                            "batch/mif_nuclei_hv_std": current_mif_n_std,
                            "batch/mif_cell_hv_std": outputs['mif_cell_hv'].std().item(),
                        })
                
                # --- FINAL EPOCH METRIC AVERAGING ---
                n_batches = len(self.train_loader)
                
                # Loss averaged over all batches
                metrics['loss'] /= n_batches
                
                # Specific metrics averaged over the number of samples that actually contributed
                metrics['he_nuclei_dice'] /= max(he_count, 1)
                # [MODIFICATION]: Divide H&E Cell Dice by valid cell count (excluding PanNuke)
                metrics['he_cell_dice'] /= max(he_cell_count, 1)
                metrics['mif_nuclei_dice'] /= max(mif_count, 1)
                metrics['mif_cell_dice'] /= max(mif_count, 1)
                
                metrics['he_nuclei_hv_std'] /= max(he_count, 1)
                metrics['he_cell_hv_std'] /= max(he_count, 1)
                metrics['mif_nuclei_hv_std'] /= max(mif_count, 1)
                metrics['mif_cell_hv_std'] /= max(mif_count, 1)
                
                return metrics

    def train_epoch_baseline_he(self, use_augmentations=True):
        """Train one epoch for VitaminPBaselineHE (H&E only)"""
        self.model.train()
        
        metrics = {
            'loss': 0,
            'he_nuclei_dice': 0,
            'he_cell_dice': 0,
            'he_nuclei_hv_std': 0,
            'he_cell_hv_std': 0,
        }
        
        pbar = tqdm(self.train_loader, desc='Training', ncols=140)
        
        for batch in pbar:
            he_img = batch['he_image'].to(self.device)
            
            # Load H&E ground truth only
            he_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
            he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
            he_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
            he_cell_hv = batch['he_cell_hv'].to(self.device)
            
            # Apply preprocessing
            he_img = self.preprocessor.percentile_normalize(he_img)
            
            if use_augmentations:
                batch_size = he_img.shape[0]
                he_aug = []
                for i in range(batch_size):
                    he_aug.append(self.preprocessor.apply_color_augmentations(he_img[i]))
                he_img = torch.stack(he_aug, dim=0)
            
            self.optimizer.zero_grad()
            outputs = self.model(he_img)
            
            # Compute losses
            loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
            loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, self.device)
            loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], he_cell_mask)
            loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, self.device)
            
            # Total loss (average of nuclei and cell)
            total_loss = (
                loss_he_nuclei_seg + 2.0 * loss_he_nuclei_hv +
                loss_he_cell_seg + 2.0 * loss_he_cell_hv
            ) / 2.0
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            with torch.no_grad():
                metrics['loss'] += total_loss.item()
                metrics['he_nuclei_dice'] += compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), he_nuclei_mask).item()
                metrics['he_cell_dice'] += compute_dice((outputs['he_cell_seg'] > 0.5).float(), he_cell_mask).item()
                
                current_he_n_std = outputs['he_nuclei_hv'].std().item()
                current_he_c_std = outputs['he_cell_hv'].std().item()
                
                metrics['he_nuclei_hv_std'] += current_he_n_std
                metrics['he_cell_hv_std'] += current_he_c_std
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'HE_N_std': f'{current_he_n_std:.3f}',
            })
            
            if self.use_wandb:
                self.wandb.log({
                    "batch/loss": total_loss.item(),
                    "batch/he_nuclei_seg_loss": loss_he_nuclei_seg.item(),
                    "batch/he_nuclei_hv_loss": loss_he_nuclei_hv.item(),
                    "batch/he_cell_seg_loss": loss_he_cell_seg.item(),
                    "batch/he_cell_hv_loss": loss_he_cell_hv.item(),
                    "batch/he_nuclei_hv_std": current_he_n_std,
                    "batch/he_cell_hv_std": current_he_c_std,
                })
        
        n_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= n_batches
        
        return metrics

    def train_epoch_baseline_mif(self, use_augmentations=True):
        """Train one epoch for VitaminPBaselineMIF (MIF only)"""
        self.model.train()
        
        metrics = {
            'loss': 0,
            'mif_nuclei_dice': 0,
            'mif_cell_dice': 0,
            'mif_nuclei_hv_std': 0,
            'mif_cell_hv_std': 0,
        }
        
        pbar = tqdm(self.train_loader, desc='Training', ncols=140)
        
        for batch in pbar:
            mif_img = batch['mif_image'].to(self.device)
            
            # Load MIF ground truth only
            mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
            mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
            mif_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
            mif_cell_hv = batch['mif_cell_hv'].to(self.device)
            
            # Apply preprocessing
            mif_img = self.preprocessor.percentile_normalize(mif_img)
            
            self.optimizer.zero_grad()
            outputs = self.model(mif_img)
            
            # Compute losses
            loss_mif_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], mif_nuclei_mask)
            loss_mif_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, self.device)
            loss_mif_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], mif_cell_mask)
            loss_mif_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], mif_cell_hv, mif_cell_mask, self.device)
            
            # Total loss (average of nuclei and cell)
            total_loss = (
                loss_mif_nuclei_seg + 2.0 * loss_mif_nuclei_hv +
                loss_mif_cell_seg + 2.0 * loss_mif_cell_hv
            ) / 2.0
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            with torch.no_grad():
                metrics['loss'] += total_loss.item()
                metrics['mif_nuclei_dice'] += compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), mif_nuclei_mask).item()
                metrics['mif_cell_dice'] += compute_dice((outputs['mif_cell_seg'] > 0.5).float(), mif_cell_mask).item()
                
                current_mif_n_std = outputs['mif_nuclei_hv'].std().item()
                current_mif_c_std = outputs['mif_cell_hv'].std().item()
                
                metrics['mif_nuclei_hv_std'] += current_mif_n_std
                metrics['mif_cell_hv_std'] += current_mif_c_std
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'MIF_N_std': f'{current_mif_n_std:.3f}',
            })
            
            if self.use_wandb:
                self.wandb.log({
                    "batch/loss": total_loss.item(),
                    "batch/mif_nuclei_seg_loss": loss_mif_nuclei_seg.item(),
                    "batch/mif_nuclei_hv_loss": loss_mif_nuclei_hv.item(),
                    "batch/mif_cell_seg_loss": loss_mif_cell_seg.item(),
                    "batch/mif_cell_hv_loss": loss_mif_cell_hv.item(),
                    "batch/mif_nuclei_hv_std": current_mif_n_std,
                    "batch/mif_cell_hv_std": current_mif_c_std,
                })
        
        n_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= n_batches
        
        return metrics

    def train_epoch(self, use_augmentations=True):
        """Route to appropriate training function based on model type"""
        if self.model_type == 'Dual' or self.model_type == 'Syn':
            return self.train_epoch_dual(use_augmentations)
        elif self.model_type == 'Flex':
            return self.train_epoch_flex(use_augmentations)
        elif self.model_type == 'BaselineHE':
            return self.train_epoch_baseline_he(use_augmentations)
        elif self.model_type == 'BaselineMIF':
            return self.train_epoch_baseline_mif(use_augmentations)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        metrics = {
            'loss': 0,
            'he_nuclei_dice': 0, 'he_cell_dice': 0,
            'mif_nuclei_dice': 0, 'mif_cell_dice': 0,
            'he_nuclei_hv_std': 0, 'he_cell_hv_std': 0,
            'mif_nuclei_hv_std': 0, 'mif_cell_hv_std': 0,
        }
        
        # Counters for accurate averaging in Flex model
        val_he_count = 0
        val_he_cell_count = 0  # Track samples with valid cell masks (excludes PanNuke)
        val_mif_count = 0
        val_batch_count = 0
        
        # ---------------------------------------------------------------------
        # VALIDATION STRATEGY: DUAL / SYN
        # ---------------------------------------------------------------------
        if self.model_type == 'Dual' or self.model_type == 'Syn':
            for batch in self.val_loader:
                he_img = batch['he_image'].to(self.device)
                mif_img = batch['mif_image'].to(self.device)
                
                # Load GT
                he_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
                he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
                he_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
                he_cell_hv = batch['he_cell_hv'].to(self.device)
                
                mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
                mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
                mif_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
                mif_cell_hv = batch['mif_cell_hv'].to(self.device)
                
                he_img = self.preprocessor.percentile_normalize(he_img)
                mif_img = self.preprocessor.percentile_normalize(mif_img)
                
                outputs = self.model(he_img, mif_img)
                
                # Loss Calculation
                loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
                loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, self.device)
                loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], he_cell_mask)
                loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, self.device)
                
                loss_mif_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], mif_nuclei_mask)
                loss_mif_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, self.device)
                loss_mif_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], mif_cell_mask)
                loss_mif_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], mif_cell_hv, mif_cell_mask, self.device)
                
                batch_loss = (
                    loss_he_nuclei_seg + 2.0 * loss_he_nuclei_hv +
                    loss_he_cell_seg + 2.0 * loss_he_cell_hv +
                    loss_mif_nuclei_seg + 2.0 * loss_mif_nuclei_hv +
                    loss_mif_cell_seg + 2.0 * loss_mif_cell_hv
                ) / 4.0
                
                metrics['loss'] += batch_loss.item()
                
                # Metrics
                metrics['he_nuclei_dice'] += compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), he_nuclei_mask).item()
                metrics['he_cell_dice'] += compute_dice((outputs['he_cell_seg'] > 0.5).float(), he_cell_mask).item()
                metrics['mif_nuclei_dice'] += compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), mif_nuclei_mask).item()
                metrics['mif_cell_dice'] += compute_dice((outputs['mif_cell_seg'] > 0.5).float(), mif_cell_mask).item()
                
                metrics['he_nuclei_hv_std'] += outputs['he_nuclei_hv'].std().item()
                metrics['he_cell_hv_std'] += outputs['he_cell_hv'].std().item()
                metrics['mif_nuclei_hv_std'] += outputs['mif_nuclei_hv'].std().item()
                metrics['mif_cell_hv_std'] += outputs['mif_cell_hv'].std().item()
                
                val_batch_count += 1
            
            # Average metrics
            for key in metrics:
                metrics[key] /= max(val_batch_count, 1)

        # ---------------------------------------------------------------------
        # VALIDATION STRATEGY: FLEX (CORRECTED FOR TISSUENET & PANNUKE)
        # ---------------------------------------------------------------------
        elif self.model_type == 'Flex':
            val_loss_accum = 0
            
            for batch in self.val_loader:
                dataset_sources = batch['dataset_source']
                
                # Load GT (Using MIF nuclei mask for both as shared truth)
                nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
                cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
                mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
                
                nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
                cell_hv = batch['he_cell_hv'].to(self.device)
                mif_cell_hv = batch['mif_cell_hv'].to(self.device)

                # --- 1. HE VALIDATION PASS (Conditional) ---
                # Only validate H&E on samples that actually HAVE H&E (not TissueNet)
                valid_he_indices = [i for i, src in enumerate(dataset_sources) if src != 'tissuenet']
                
                if len(valid_he_indices) > 0:
                    # Subset batch for H&E
                    he_subset = batch['he_image'][valid_he_indices].to(self.device)
                    n_mask_sub = nuclei_mask[valid_he_indices]
                    c_mask_sub = cell_mask[valid_he_indices]
                    n_hv_sub = nuclei_hv[valid_he_indices]
                    c_hv_sub = cell_hv[valid_he_indices]
                    
                    # Get subset sources to check for PanNuke
                    subset_sources = [dataset_sources[i] for i in valid_he_indices]
                    
                    he_subset = prepare_he_input(he_subset)
                    he_subset = self.preprocessor.percentile_normalize(he_subset)
                    
                    outputs_he = self.model(he_subset)
                    
                    # Iterate over subset to handle PanNuke exclusion
                    loss_he_batch = 0
                    for k, src in enumerate(subset_sources):
                        # Nuclei Loss (Always valid)
                        l_n = (self.seg_criterion(outputs_he['he_nuclei_seg'][k:k+1], n_mask_sub[k:k+1]) +
                               2.0 * self.hv_criterion(outputs_he['he_nuclei_hv'][k:k+1], n_hv_sub[k:k+1], n_mask_sub[k:k+1], self.device))
                        
                        # Cell Loss (Skip for PanNuke)
                        if src == 'pannuke':
                            l_c = torch.tensor(0.0, device=self.device)
                        else:
                            l_c = (self.seg_criterion(outputs_he['he_cell_seg'][k:k+1], c_mask_sub[k:k+1]) +
                                   2.0 * self.hv_criterion(outputs_he['he_cell_hv'][k:k+1], c_hv_sub[k:k+1], c_mask_sub[k:k+1], self.device))
                            
                            # Metrics for Cells (Only if not PanNuke)
                            metrics['he_cell_dice'] += compute_dice((outputs_he['he_cell_seg'][k:k+1] > 0.5).float(), c_mask_sub[k:k+1]).item()
                            val_he_cell_count += 1
                        
                        loss_he_batch += (l_n + l_c)
                        
                        # Metrics for Nuclei (Always)
                        metrics['he_nuclei_dice'] += compute_dice((outputs_he['he_nuclei_seg'][k:k+1] > 0.5).float(), n_mask_sub[k:k+1]).item()
                        metrics['he_nuclei_hv_std'] += outputs_he['he_nuclei_hv'][k:k+1].std().item()
                        metrics['he_cell_hv_std'] += outputs_he['he_cell_hv'][k:k+1].std().item()

                    val_loss_accum += (loss_he_batch / len(valid_he_indices)).item()
                    val_he_count += len(valid_he_indices)

                # --- 2. MIF VALIDATION PASS (All Samples except PanNuke) ---
                # Valid for CRC, Xenium, TissueNet (PanNuke has no MIF)
                valid_mif_indices = [i for i, src in enumerate(dataset_sources) if src != 'pannuke']
                
                if len(valid_mif_indices) > 0:
                    mif_subset = batch['mif_image'][valid_mif_indices].to(self.device)
                    n_mask_sub = nuclei_mask[valid_mif_indices]
                    c_mask_sub = mif_cell_mask[valid_mif_indices]
                    n_hv_sub = nuclei_hv[valid_mif_indices]
                    c_hv_sub = mif_cell_hv[valid_mif_indices]
                    
                    mif_subset = prepare_mif_input(mif_subset)
                    mif_subset = self.preprocessor.percentile_normalize(mif_subset)
                    
                    outputs_mif = self.model(mif_subset)
                    
                    loss_mif = (
                        self.seg_criterion(outputs_mif['mif_nuclei_seg'], n_mask_sub) +
                        2.0 * self.hv_criterion(outputs_mif['mif_nuclei_hv'], n_hv_sub, n_mask_sub, self.device) +
                        self.seg_criterion(outputs_mif['mif_cell_seg'], c_mask_sub) +
                        2.0 * self.hv_criterion(outputs_mif['mif_cell_hv'], c_hv_sub, c_mask_sub, self.device)
                    ) / 2.0
                    
                    val_loss_accum += loss_mif.item()
                    
                    count = len(valid_mif_indices)
                    metrics['mif_nuclei_dice'] += compute_dice((outputs_mif['mif_nuclei_seg'] > 0.5).float(), n_mask_sub).item() * count
                    metrics['mif_cell_dice'] += compute_dice((outputs_mif['mif_cell_seg'] > 0.5).float(), c_mask_sub).item() * count
                    metrics['mif_nuclei_hv_std'] += outputs_mif['mif_nuclei_hv'].std().item() * count
                    metrics['mif_cell_hv_std'] += outputs_mif['mif_cell_hv'].std().item() * count
                    
                    val_mif_count += count
                
                val_batch_count += 1
            
            # Normalize Flex Metrics
            metrics['loss'] = val_loss_accum / max(val_batch_count, 1)
            
            if val_he_count > 0:
                metrics['he_nuclei_dice'] /= val_he_count
                metrics['he_nuclei_hv_std'] /= val_he_count
                metrics['he_cell_hv_std'] /= val_he_count
                
            if val_he_cell_count > 0:
                metrics['he_cell_dice'] /= val_he_cell_count
            
            if val_mif_count > 0:
                metrics['mif_nuclei_dice'] /= val_mif_count
                metrics['mif_cell_dice'] /= val_mif_count
                metrics['mif_nuclei_hv_std'] /= val_mif_count
                metrics['mif_cell_hv_std'] /= val_mif_count

        # ---------------------------------------------------------------------
        # VALIDATION STRATEGY: BASELINE HE
        # ---------------------------------------------------------------------
        elif self.model_type == 'BaselineHE':
            for batch in self.val_loader:
                he_img = batch['he_image'].to(self.device)
                he_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
                he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
                he_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
                he_cell_hv = batch['he_cell_hv'].to(self.device)
                
                he_img = self.preprocessor.percentile_normalize(he_img)
                outputs = self.model(he_img)
                
                loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
                loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, self.device)
                loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], he_cell_mask)
                loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, self.device)
                
                batch_loss = (
                    loss_he_nuclei_seg + 2.0 * loss_he_nuclei_hv +
                    loss_he_cell_seg + 2.0 * loss_he_cell_hv
                ) / 2.0
                
                metrics['loss'] += batch_loss.item()
                metrics['he_nuclei_dice'] += compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), he_nuclei_mask).item()
                metrics['he_cell_dice'] += compute_dice((outputs['he_cell_seg'] > 0.5).float(), he_cell_mask).item()
                metrics['he_nuclei_hv_std'] += outputs['he_nuclei_hv'].std().item()
                metrics['he_cell_hv_std'] += outputs['he_cell_hv'].std().item()
                
                val_batch_count += 1
            
            for key in metrics:
                metrics[key] /= max(val_batch_count, 1)

        # ---------------------------------------------------------------------
        # VALIDATION STRATEGY: BASELINE MIF
        # ---------------------------------------------------------------------
        elif self.model_type == 'BaselineMIF':
            for batch in self.val_loader:
                mif_img = batch['mif_image'].to(self.device)
                mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
                mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
                mif_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
                mif_cell_hv = batch['mif_cell_hv'].to(self.device)
                
                mif_img = self.preprocessor.percentile_normalize(mif_img)
                outputs = self.model(mif_img)
                
                loss_mif_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], mif_nuclei_mask)
                loss_mif_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, self.device)
                loss_mif_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], mif_cell_mask)
                loss_mif_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], mif_cell_hv, mif_cell_mask, self.device)
                
                batch_loss = (
                    loss_mif_nuclei_seg + 2.0 * loss_mif_nuclei_hv +
                    loss_mif_cell_seg + 2.0 * loss_mif_cell_hv
                ) / 2.0
                
                metrics['loss'] += batch_loss.item()
                metrics['mif_nuclei_dice'] += compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), mif_nuclei_mask).item()
                metrics['mif_cell_dice'] += compute_dice((outputs['mif_cell_seg'] > 0.5).float(), mif_cell_mask).item()
                metrics['mif_nuclei_hv_std'] += outputs['mif_nuclei_hv'].std().item()
                metrics['mif_cell_hv_std'] += outputs['mif_cell_hv'].std().item()
                
                val_batch_count += 1

            for key in metrics:
                metrics[key] /= max(val_batch_count, 1)
        
        return metrics

    def train(self, epochs, use_augmentations=True):
        """
        Train the model for specified number of epochs
        
        Args:
            epochs: Number of epochs to train
            use_augmentations: Whether to apply color augmentations
        """
        print("=" * 80)
        print(f"Training VitaminP{self.model_type}-{self.model.model_size.upper()}")
        print(f"Epochs: {epochs} | LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"Augmentations: {use_augmentations}")
        print("=" * 80)
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(use_augmentations=use_augmentations)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # WANDB LOGGING - CONDITIONAL BASED ON MODEL TYPE
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics['loss'],
                    "val/loss": val_metrics['loss'],
                    "lr": self.optimizer.param_groups[0]['lr']
                }
                
                # Add metrics based on model type
                if self.model_type == 'BaselineHE':
                    # Only HE metrics
                    log_dict.update({
                        "train/he_nuclei_dice": train_metrics['he_nuclei_dice'],
                        "train/he_cell_dice": train_metrics['he_cell_dice'],
                        "train/dice_avg": (train_metrics['he_nuclei_dice'] + train_metrics['he_cell_dice']) / 2,
                        "train/he_nuclei_hv_std": train_metrics['he_nuclei_hv_std'],
                        "train/he_cell_hv_std": train_metrics['he_cell_hv_std'],
                        "val/he_nuclei_dice": val_metrics['he_nuclei_dice'],
                        "val/he_cell_dice": val_metrics['he_cell_dice'],
                        "val/dice_avg": (val_metrics['he_nuclei_dice'] + val_metrics['he_cell_dice']) / 2,
                        "val/he_nuclei_hv_std": val_metrics['he_nuclei_hv_std'],
                        "val/he_cell_hv_std": val_metrics['he_cell_hv_std'],
                    })
                elif self.model_type == 'BaselineMIF':
                    # Only MIF metrics
                    log_dict.update({
                        "train/mif_nuclei_dice": train_metrics['mif_nuclei_dice'],
                        "train/mif_cell_dice": train_metrics['mif_cell_dice'],
                        "train/dice_avg": (train_metrics['mif_nuclei_dice'] + train_metrics['mif_cell_dice']) / 2,
                        "train/mif_nuclei_hv_std": train_metrics['mif_nuclei_hv_std'],
                        "train/mif_cell_hv_std": train_metrics['mif_cell_hv_std'],
                        "val/mif_nuclei_dice": val_metrics['mif_nuclei_dice'],
                        "val/mif_cell_dice": val_metrics['mif_cell_dice'],
                        "val/dice_avg": (val_metrics['mif_nuclei_dice'] + val_metrics['mif_cell_dice']) / 2,
                        "val/mif_nuclei_hv_std": val_metrics['mif_nuclei_hv_std'],
                        "val/mif_cell_hv_std": val_metrics['mif_cell_hv_std'],
                    })
                else:
                    # Dual and Flex - all 4 decoders
                    log_dict.update({
                        "train/he_nuclei_dice": train_metrics['he_nuclei_dice'],
                        "train/he_cell_dice": train_metrics['he_cell_dice'],
                        "train/mif_nuclei_dice": train_metrics['mif_nuclei_dice'],
                        "train/mif_cell_dice": train_metrics['mif_cell_dice'],
                        "train/dice_avg": (
                            train_metrics['he_nuclei_dice'] + 
                            train_metrics['he_cell_dice'] + 
                            train_metrics['mif_nuclei_dice'] + 
                            train_metrics['mif_cell_dice']
                        ) / 4,
                        "train/he_nuclei_hv_std": train_metrics['he_nuclei_hv_std'],
                        "train/he_cell_hv_std": train_metrics['he_cell_hv_std'],
                        "train/mif_nuclei_hv_std": train_metrics['mif_nuclei_hv_std'],
                        "train/mif_cell_hv_std": train_metrics['mif_cell_hv_std'],
                        "val/he_nuclei_dice": val_metrics['he_nuclei_dice'],
                        "val/he_cell_dice": val_metrics['he_cell_dice'],
                        "val/mif_nuclei_dice": val_metrics['mif_nuclei_dice'],
                        "val/mif_cell_dice": val_metrics['mif_cell_dice'],
                        "val/dice_avg": (
                            val_metrics['he_nuclei_dice'] + 
                            val_metrics['he_cell_dice'] + 
                            val_metrics['mif_nuclei_dice'] + 
                            val_metrics['mif_cell_dice']
                        ) / 4,
                        "val/he_nuclei_hv_std": val_metrics['he_nuclei_hv_std'],
                        "val/he_cell_hv_std": val_metrics['he_cell_hv_std'],
                        "val/mif_nuclei_hv_std": val_metrics['mif_nuclei_hv_std'],
                        "val/mif_cell_hv_std": val_metrics['mif_cell_hv_std'],
                    })
                
                self.wandb.log(log_dict)
            
            # Print metrics - CONDITIONAL
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            
            if self.model_type == 'BaselineHE':
                print(f"Val Dice - HE: N={val_metrics['he_nuclei_dice']:.4f} C={val_metrics['he_cell_dice']:.4f}")
            elif self.model_type == 'BaselineMIF':
                print(f"Val Dice - MIF: N={val_metrics['mif_nuclei_dice']:.4f} C={val_metrics['mif_cell_dice']:.4f}")
            else:
                print(f"Val Dice - HE: N={val_metrics['he_nuclei_dice']:.4f} C={val_metrics['he_cell_dice']:.4f} | "
                    f"MIF: N={val_metrics['mif_nuclei_dice']:.4f} C={val_metrics['mif_cell_dice']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'vitamin_p_{self.model_type.lower()}_{self.model.model_size}_fold{self.fold}_best.pth'
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                
                if self.use_wandb:
                    self.wandb.log({"best_val_loss": self.best_val_loss})
                
                print(f'✅ Best model saved! Val loss: {val_metrics["loss"]:.4f}\n')
        
        print(f"\n{'='*80}\nTraining Complete | Best Val Loss: {self.best_val_loss:.4f}\n{'='*80}")
        
        if self.use_wandb:
            self.wandb.finish()
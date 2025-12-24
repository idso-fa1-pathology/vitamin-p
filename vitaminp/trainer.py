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
        """Detect whether model is Dual or Flex"""
        model_class_name = model.__class__.__name__
        if 'Dual' in model_class_name:
            return 'Dual'
        elif 'Flex' in model_class_name:
            return 'Flex'
        else:
            return 'Unknown'
    
    def _get_architecture_description(self):
        """Get architecture description based on model type"""
        if self.model_type == 'Dual':
            return "Dual-Encoder with Mid-Fusion"
        elif self.model_type == 'Flex':
            return "Shared Encoder → 4 Separate Decoders"
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
            he_nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
            he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
            he_nuclei_hv = batch['he_nuclei_hv'].to(self.device)
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
        mif_count = 0
        
        pbar = tqdm(self.train_loader, desc='Training', ncols=140)
        
        for batch in pbar:
            he_img = batch['he_image'].to(self.device)
            mif_img = batch['mif_image'].to(self.device)
            
            # ❌ ISSUE #1: ONLY loading H&E ground truth!
            # This is WRONG - you need BOTH H&E and MIF ground truth
            # nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
            # cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
            # nuclei_hv = batch['he_nuclei_hv'].to(self.device)
            # cell_hv = batch['he_cell_hv'].to(self.device)
            
            # ✅ FIX: Load BOTH H&E and MIF ground truth
            he_nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
            he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
            he_nuclei_hv = batch['he_nuclei_hv'].to(self.device)
            he_cell_hv = batch['he_cell_hv'].to(self.device)
            
            mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
            mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
            mif_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
            mif_cell_hv = batch['mif_cell_hv'].to(self.device)
            
            batch_size = he_img.shape[0]
            mixed_images = []
            modality_labels = []
            
            # ✅ FIX: Prepare ground truth lists matching the modality
            nuclei_masks = []
            cell_masks = []
            nuclei_hvs = []
            cell_hvs = []
            
            # Random modality selection per sample
            for i in range(batch_size):
                use_mif = torch.rand(1).item() < 0.5
                if use_mif:
                    img = prepare_mif_input(mif_img[i:i+1])[0]
                    modality_labels.append('mif')
                    mif_count += 1
                    # ✅ FIX: Use MIF ground truth for MIF samples
                    nuclei_masks.append(mif_nuclei_mask[i])
                    cell_masks.append(mif_cell_mask[i])
                    nuclei_hvs.append(mif_nuclei_hv[i])
                    cell_hvs.append(mif_cell_hv[i])
                else:
                    img = prepare_he_input(he_img[i:i+1])[0]
                    modality_labels.append('he')
                    he_count += 1
                    # ✅ FIX: Use H&E ground truth for H&E samples
                    nuclei_masks.append(he_nuclei_mask[i])
                    cell_masks.append(he_cell_mask[i])
                    nuclei_hvs.append(he_nuclei_hv[i])
                    cell_hvs.append(he_cell_hv[i])
                
                img = self.preprocessor.percentile_normalize(img)
                
                if use_augmentations:
                    img = self.preprocessor.apply_color_augmentations(img)
                
                mixed_images.append(img)
            
            mixed_batch = torch.stack(mixed_images, dim=0)
            
            # ✅ FIX: Stack the correct ground truth
            nuclei_mask_batch = torch.stack(nuclei_masks, dim=0)
            cell_mask_batch = torch.stack(cell_masks, dim=0)
            nuclei_hv_batch = torch.stack(nuclei_hvs, dim=0)
            cell_hv_batch = torch.stack(cell_hvs, dim=0)
            
            self.optimizer.zero_grad()
            outputs = self.model(mixed_batch)
            
            # ❌ ISSUE #2: Computing loss for ALL decoders with H&E ground truth!
            # This trains MIF decoders to predict H&E patterns - COMPLETELY WRONG!
            
            # ✅ FIX: Only compute loss for the decoder matching the input modality
            total_loss = 0
            loss_components = {
                'he_nuclei_seg': 0, 'he_nuclei_hv': 0,
                'he_cell_seg': 0, 'he_cell_hv': 0,
                'mif_nuclei_seg': 0, 'mif_nuclei_hv': 0,
                'mif_cell_seg': 0, 'mif_cell_hv': 0
            }
            
            # Process each sample individually based on its modality
            for i, modality in enumerate(modality_labels):
                if modality == 'he':
                    # Train H&E decoders with H&E ground truth
                    loss_nuclei_seg = self.seg_criterion(
                        outputs['he_nuclei_seg'][i:i+1],
                        nuclei_mask_batch[i:i+1]
                    )
                    loss_nuclei_hv = self.hv_criterion(
                        outputs['he_nuclei_hv'][i:i+1],
                        nuclei_hv_batch[i:i+1],
                        nuclei_mask_batch[i:i+1],
                        self.device
                    )
                    loss_cell_seg = self.seg_criterion(
                        outputs['he_cell_seg'][i:i+1],
                        cell_mask_batch[i:i+1]
                    )
                    loss_cell_hv = self.hv_criterion(
                        outputs['he_cell_hv'][i:i+1],
                        cell_hv_batch[i:i+1],
                        cell_mask_batch[i:i+1],
                        self.device
                    )
                    
                    loss_components['he_nuclei_seg'] += loss_nuclei_seg.item()
                    loss_components['he_nuclei_hv'] += loss_nuclei_hv.item()
                    loss_components['he_cell_seg'] += loss_cell_seg.item()
                    loss_components['he_cell_hv'] += loss_cell_hv.item()
                    
                else:  # mif
                    # Train MIF decoders with MIF ground truth
                    loss_nuclei_seg = self.seg_criterion(
                        outputs['mif_nuclei_seg'][i:i+1],
                        nuclei_mask_batch[i:i+1]
                    )
                    loss_nuclei_hv = self.hv_criterion(
                        outputs['mif_nuclei_hv'][i:i+1],
                        nuclei_hv_batch[i:i+1],
                        nuclei_mask_batch[i:i+1],
                        self.device
                    )
                    loss_cell_seg = self.seg_criterion(
                        outputs['mif_cell_seg'][i:i+1],
                        cell_mask_batch[i:i+1]
                    )
                    loss_cell_hv = self.hv_criterion(
                        outputs['mif_cell_hv'][i:i+1],
                        cell_hv_batch[i:i+1],
                        cell_mask_batch[i:i+1],
                        self.device
                    )
                    
                    loss_components['mif_nuclei_seg'] += loss_nuclei_seg.item()
                    loss_components['mif_nuclei_hv'] += loss_nuclei_hv.item()
                    loss_components['mif_cell_seg'] += loss_cell_seg.item()
                    loss_components['mif_cell_hv'] += loss_cell_hv.item()
                
                # Add to total loss
                total_loss += (loss_nuclei_seg + 2.0 * loss_nuclei_hv + 
                            loss_cell_seg + 2.0 * loss_cell_hv)
            
            # Average loss over batch
            total_loss = total_loss / batch_size
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            with torch.no_grad():
                metrics['loss'] += total_loss.item()
                
                # Track metrics by modality with CORRECT ground truth
                for i, modality in enumerate(modality_labels):
                    if modality == 'he':
                        metrics['he_nuclei_dice'] += compute_dice(
                            (outputs['he_nuclei_seg'][i:i+1] > 0.5).float(),
                            nuclei_mask_batch[i:i+1]
                        ).item()
                        metrics['he_cell_dice'] += compute_dice(
                            (outputs['he_cell_seg'][i:i+1] > 0.5).float(),
                            cell_mask_batch[i:i+1]
                        ).item()
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
                
                current_he_n_std = outputs['he_nuclei_hv'].std().item()
                current_mif_n_std = outputs['mif_nuclei_hv'].std().item()
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'HE_N_std': f'{current_he_n_std:.3f}',
                'MIF_N_std': f'{current_mif_n_std:.3f}'
            })
            
            # Batch logging
            if self.use_wandb:
                # Normalize loss components for logging
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
        
        # Normalize metrics
        n_batches = len(self.train_loader)
        metrics['loss'] /= n_batches
        metrics['he_nuclei_dice'] /= max(he_count, 1)
        metrics['he_cell_dice'] /= max(he_count, 1)
        metrics['mif_nuclei_dice'] /= max(mif_count, 1)
        metrics['mif_cell_dice'] /= max(mif_count, 1)
        metrics['he_nuclei_hv_std'] /= max(he_count, 1)
        metrics['he_cell_hv_std'] /= max(he_count, 1)
        metrics['mif_nuclei_hv_std'] /= max(mif_count, 1)
        metrics['mif_cell_hv_std'] /= max(mif_count, 1)
        
        return metrics

    def train_epoch(self, use_augmentations=True):
        """Route to appropriate training function based on model type"""
        if self.model_type == 'Dual':
            return self.train_epoch_dual(use_augmentations)
        elif self.model_type == 'Flex':
            return self.train_epoch_flex(use_augmentations)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
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
        
        if self.model_type == 'Dual':
            # Dual model validation
            for batch in self.val_loader:
                he_img = batch['he_image'].to(self.device)
                mif_img = batch['mif_image'].to(self.device)
                
                # ✅ FIX: Load BOTH H&E and MIF ground truth
                he_nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
                he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
                he_nuclei_hv = batch['he_nuclei_hv'].to(self.device)
                he_cell_hv = batch['he_cell_hv'].to(self.device)
                
                mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
                mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
                mif_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
                mif_cell_hv = batch['mif_cell_hv'].to(self.device)
                
                he_img = self.preprocessor.percentile_normalize(he_img)
                mif_img = self.preprocessor.percentile_normalize(mif_img)
                
                outputs = self.model(he_img, mif_img)
                
                # ✅ FIX: Use correct ground truth for each decoder
                # H&E decoders with H&E ground truth
                loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
                loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, self.device)
                loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], he_cell_mask)
                loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, self.device)
                
                # MIF decoders with MIF ground truth
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
                
                # ✅ FIX: Compare predictions to correct ground truth
                metrics['he_nuclei_dice'] += compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), he_nuclei_mask).item()
                metrics['he_cell_dice'] += compute_dice((outputs['he_cell_seg'] > 0.5).float(), he_cell_mask).item()
                metrics['mif_nuclei_dice'] += compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), mif_nuclei_mask).item()
                metrics['mif_cell_dice'] += compute_dice((outputs['mif_cell_seg'] > 0.5).float(), mif_cell_mask).item()
                
                metrics['he_nuclei_hv_std'] += outputs['he_nuclei_hv'].std().item()
                metrics['he_cell_hv_std'] += outputs['he_cell_hv'].std().item()
                metrics['mif_nuclei_hv_std'] += outputs['mif_nuclei_hv'].std().item()
                metrics['mif_cell_hv_std'] += outputs['mif_cell_hv'].std().item()
        
        elif self.model_type == 'Flex':
            # Flex model validation (separate HE and MIF passes)
            val_loss_he = 0
            val_loss_mif = 0
            
            for batch in self.val_loader:
                # ✅ FIX: Load BOTH H&E and MIF ground truth
                he_nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
                he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
                he_nuclei_hv = batch['he_nuclei_hv'].to(self.device)
                he_cell_hv = batch['he_cell_hv'].to(self.device)
                
                mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(self.device)
                mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(self.device)
                mif_nuclei_hv = batch['mif_nuclei_hv'].to(self.device)
                mif_cell_hv = batch['mif_cell_hv'].to(self.device)
                
                # ✅ HE validation with H&E ground truth
                he_img = prepare_he_input(batch['he_image'].to(self.device))
                he_img = self.preprocessor.percentile_normalize(he_img)
                outputs_he = self.model(he_img)
                
                loss_he = (
                    self.seg_criterion(outputs_he['he_nuclei_seg'], he_nuclei_mask) +
                    2.0 * self.hv_criterion(outputs_he['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, self.device) +
                    self.seg_criterion(outputs_he['he_cell_seg'], he_cell_mask) +
                    2.0 * self.hv_criterion(outputs_he['he_cell_hv'], he_cell_hv, he_cell_mask, self.device)
                ) / 2.0
                
                val_loss_he += loss_he.item()
                metrics['he_nuclei_dice'] += compute_dice((outputs_he['he_nuclei_seg'] > 0.5).float(), he_nuclei_mask).item()
                metrics['he_cell_dice'] += compute_dice((outputs_he['he_cell_seg'] > 0.5).float(), he_cell_mask).item()
                metrics['he_nuclei_hv_std'] += outputs_he['he_nuclei_hv'].std().item()
                metrics['he_cell_hv_std'] += outputs_he['he_cell_hv'].std().item()
                
                # ✅ MIF validation with MIF ground truth
                mif_img = prepare_mif_input(batch['mif_image'].to(self.device))
                mif_img = self.preprocessor.percentile_normalize(mif_img)
                outputs_mif = self.model(mif_img)
                
                loss_mif = (
                    self.seg_criterion(outputs_mif['mif_nuclei_seg'], mif_nuclei_mask) +
                    2.0 * self.hv_criterion(outputs_mif['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, self.device) +
                    self.seg_criterion(outputs_mif['mif_cell_seg'], mif_cell_mask) +
                    2.0 * self.hv_criterion(outputs_mif['mif_cell_hv'], mif_cell_hv, mif_cell_mask, self.device)
                ) / 2.0
                
                val_loss_mif += loss_mif.item()
                metrics['mif_nuclei_dice'] += compute_dice((outputs_mif['mif_nuclei_seg'] > 0.5).float(), mif_nuclei_mask).item()
                metrics['mif_cell_dice'] += compute_dice((outputs_mif['mif_cell_seg'] > 0.5).float(), mif_cell_mask).item()
                metrics['mif_nuclei_hv_std'] += outputs_mif['mif_nuclei_hv'].std().item()
                metrics['mif_cell_hv_std'] += outputs_mif['mif_cell_hv'].std().item()
            
            n_val = len(self.val_loader)
            metrics['loss'] = (val_loss_he + val_loss_mif) / (2 * n_val)
        
        # Average metrics
        n_batches = len(self.val_loader)
        if self.model_type != 'Flex':
            for key in metrics:
                metrics[key] /= n_batches
        else:
            for key in ['he_nuclei_dice', 'he_cell_dice', 'mif_nuclei_dice', 'mif_cell_dice',
                    'he_nuclei_hv_std', 'he_cell_hv_std', 'mif_nuclei_hv_std', 'mif_cell_hv_std']:
                metrics[key] /= n_batches
        
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
            
            # COMPLETE EPOCH LOGGING - EXACT AS ORIGINAL
            if self.use_wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    # Train metrics - ALL OF THEM
                    "train/loss": train_metrics['loss'],
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
                    # Val metrics - ALL OF THEM
                    "val/loss": val_metrics['loss'],
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
                    # Learning rate
                    "lr": self.optimizer.param_groups[0]['lr']
                })
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
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
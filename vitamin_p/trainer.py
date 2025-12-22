"""
Trainer for Vitamin-P models

Supports all model variants with flexible configuration.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from pathlib import Path

from .losses import DiceFocalLoss, HVLoss
from .utils import SimplePreprocessing, compute_dice, prepare_he_input, prepare_mif_input


class VitaminPTrainer:
    """
    Universal trainer for all Vitamin-P model variants
    
    Args:
        model: Vitamin-P model instance
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device (str): Device to train on ('cuda' or 'cpu')
        lr (float): Learning rate
        weight_decay (float): Weight decay for optimizer
        use_wandb (bool): Whether to use Weights & Biases logging
        project_name (str): W&B project name
        run_name (str): W&B run name
        checkpoint_dir (str): Directory to save checkpoints
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-4,
        weight_decay=1e-4,
        use_wandb=False,
        project_name="vitamin-p",
        run_name=None,
        checkpoint_dir="checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine model type
        self.model_type = self._determine_model_type()
        
        # Losses
        self.seg_criterion = DiceFocalLoss(alpha=1, gamma=2)
        self.hv_criterion = HVLoss()
        
        # Optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5, verbose=True
        )
        
        # Preprocessor
        self.preprocessor = SimplePreprocessing()
        
        # Best validation loss
        self.best_val_loss = float('inf')
        
        # Initialize wandb if requested
        if self.use_wandb:
            if run_name is None:
                run_name = f"{self.model_type}-{model.model_size}"
            
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model": self.model_type,
                    "model_size": model.model_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "batch_size": train_loader.batch_size,
                    "device": device,
                }
            )
    
    def _determine_model_type(self):
        """Determine which model variant we're training"""
        model_name = self.model.__class__.__name__
        return model_name
    
    def train_epoch(self, epoch, use_augmentations=True):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', ncols=140)
        
        for batch in pbar:
            # Get data based on model type
            if 'Dual' in self.model_type:
                loss, batch_metrics = self._train_step_dual(batch, use_augmentations)
            else:
                loss, batch_metrics = self._train_step_single(batch, use_augmentations)
            
            total_loss += loss
            
            # Update metrics
            for k, v in batch_metrics.items():
                metrics[k] = metrics.get(k, 0) + v
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss:.4f}'})
            
            # Log to wandb (batch level)
            if self.use_wandb:
                wandb.log({"batch/loss": loss})
        
        # Average metrics
        n_batches = len(self.train_loader)
        total_loss /= n_batches
        for k in metrics:
            metrics[k] /= n_batches
        
        return total_loss, metrics
    
    def _train_step_single(self, batch, use_augmentations):
        """Training step for single encoder models (Flex, HE, MIF baselines)"""
        # Determine if we need HE or MIF input
        if 'MIF' in self.model_type:
            img = prepare_mif_input(batch['mif_image'].to(self.device))
        else:
            # For Flex and HE models, can use either HE or MIF (randomly for Flex)
            if 'Flex' in self.model_type and torch.rand(1).item() < 0.5:
                img = prepare_mif_input(batch['mif_image'].to(self.device))
                use_mif_branch = True
            else:
                img = prepare_he_input(batch['he_image'].to(self.device))
                use_mif_branch = False
        
        # Get targets
        nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
        cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
        nuclei_hv = batch['he_nuclei_hv'].to(self.device)
        cell_hv = batch['he_cell_hv'].to(self.device)
        
        # Preprocess
        img = self.preprocessor.percentile_normalize(img)
        
        if use_augmentations:
            batch_size = img.shape[0]
            img_aug = []
            for i in range(batch_size):
                img_aug.append(self.preprocessor.apply_color_augmentations(img[i]))
            img = torch.stack(img_aug, dim=0)
        
        # Forward
        self.optimizer.zero_grad()
        outputs = self.model(img)
        
        # Compute loss based on model type
        if 'MIF' in self.model_type:
            # MIF baseline
            loss_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], nuclei_mask)
            loss_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
            loss_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], cell_mask)
            loss_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], cell_hv, cell_mask, self.device)
            
            total_loss = (loss_nuclei_seg + 2.0 * loss_nuclei_hv + 
                         loss_cell_seg + 2.0 * loss_cell_hv) / 2.0
            
            dice_nuclei = compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), nuclei_mask)
            dice_cell = compute_dice((outputs['mif_cell_seg'] > 0.5).float(), cell_mask)
            
        elif 'HE' in self.model_type and 'Baseline' in self.model_type:
            # HE baseline
            loss_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], nuclei_mask)
            loss_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
            loss_cell_seg = self.seg_criterion(outputs['he_cell_seg'], cell_mask)
            loss_cell_hv = self.hv_criterion(outputs['he_cell_hv'], cell_hv, cell_mask, self.device)
            
            total_loss = (loss_nuclei_seg + 2.0 * loss_nuclei_hv + 
                         loss_cell_seg + 2.0 * loss_cell_hv) / 2.0
            
            dice_nuclei = compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), nuclei_mask)
            dice_cell = compute_dice((outputs['he_cell_seg'] > 0.5).float(), cell_mask)
            
        else:
            # Flex model - all 4 branches
            loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], nuclei_mask)
            loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
            loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], cell_mask)
            loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], cell_hv, cell_mask, self.device)
            
            loss_mif_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], nuclei_mask)
            loss_mif_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
            loss_mif_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], cell_mask)
            loss_mif_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], cell_hv, cell_mask, self.device)
            
            total_loss = (
                loss_he_nuclei_seg + 2.0 * loss_he_nuclei_hv +
                loss_he_cell_seg + 2.0 * loss_he_cell_hv +
                loss_mif_nuclei_seg + 2.0 * loss_mif_nuclei_hv +
                loss_mif_cell_seg + 2.0 * loss_mif_cell_hv
            ) / 4.0
            
            dice_nuclei = (compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), nuclei_mask) +
                          compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), nuclei_mask)) / 2
            dice_cell = (compute_dice((outputs['he_cell_seg'] > 0.5).float(), cell_mask) +
                        compute_dice((outputs['mif_cell_seg'] > 0.5).float(), cell_mask)) / 2
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        metrics = {
            'dice_nuclei': dice_nuclei,
            'dice_cell': dice_cell
        }
        
        return total_loss.item(), metrics
    
    def _train_step_dual(self, batch, use_augmentations):
        """Training step for dual encoder model"""
        he_img = batch['he_image'].to(self.device)
        mif_img = batch['mif_image'].to(self.device)
        
        nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
        cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
        nuclei_hv = batch['he_nuclei_hv'].to(self.device)
        cell_hv = batch['he_cell_hv'].to(self.device)
        
        # Preprocess
        he_img = self.preprocessor.percentile_normalize(he_img)
        mif_img = self.preprocessor.percentile_normalize(mif_img)
        
        if use_augmentations:
            batch_size = he_img.shape[0]
            he_aug = []
            for i in range(batch_size):
                he_aug.append(self.preprocessor.apply_color_augmentations(he_img[i]))
            he_img = torch.stack(he_aug, dim=0)
        
        # Forward
        self.optimizer.zero_grad()
        outputs = self.model(he_img, mif_img)
        
        # Compute losses for all 4 branches
        loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], nuclei_mask)
        loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
        loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], cell_mask)
        loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], cell_hv, cell_mask, self.device)
        
        loss_mif_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], nuclei_mask)
        loss_mif_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
        loss_mif_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], cell_mask)
        loss_mif_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], cell_hv, cell_mask, self.device)
        
        total_loss = (
            loss_he_nuclei_seg + 2.0 * loss_he_nuclei_hv +
            loss_he_cell_seg + 2.0 * loss_he_cell_hv +
            loss_mif_nuclei_seg + 2.0 * loss_mif_nuclei_hv +
            loss_mif_cell_seg + 2.0 * loss_mif_cell_hv
        ) / 4.0
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        metrics = {
            'dice_nuclei': (compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), nuclei_mask) +
                           compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), nuclei_mask)) / 2,
            'dice_cell': (compute_dice((outputs['he_cell_seg'] > 0.5).float(), cell_mask) +
                         compute_dice((outputs['mif_cell_seg'] > 0.5).float(), cell_mask)) / 2
        }
        
        return total_loss.item(), metrics
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0
        metrics = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                if 'Dual' in self.model_type:
                    loss, batch_metrics = self._val_step_dual(batch)
                else:
                    loss, batch_metrics = self._val_step_single(batch)
                
                total_loss += loss
                
                for k, v in batch_metrics.items():
                    metrics[k] = metrics.get(k, 0) + v
        
        # Average metrics
        n_batches = len(self.val_loader)
        total_loss /= n_batches
        for k in metrics:
            metrics[k] /= n_batches
        
        return total_loss, metrics
    
    def _val_step_single(self, batch):
        """Validation step for single encoder models"""
        # Similar to train step but without augmentation and gradient updates
        if 'MIF' in self.model_type:
            img = prepare_mif_input(batch['mif_image'].to(self.device))
        else:
            img = prepare_he_input(batch['he_image'].to(self.device))
        
        nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
        cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
        nuclei_hv = batch['he_nuclei_hv'].to(self.device)
        cell_hv = batch['he_cell_hv'].to(self.device)
        
        img = self.preprocessor.percentile_normalize(img)
        
        outputs = self.model(img)
        
        # Compute loss (same logic as training)
        if 'MIF' in self.model_type:
            loss_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], nuclei_mask)
            loss_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
            loss_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], cell_mask)
            loss_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], cell_hv, cell_mask, self.device)
            
            total_loss = (loss_nuclei_seg + 2.0 * loss_nuclei_hv + 
                         loss_cell_seg + 2.0 * loss_cell_hv) / 2.0
            
            dice_nuclei = compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), nuclei_mask)
            dice_cell = compute_dice((outputs['mif_cell_seg'] > 0.5).float(), cell_mask)
            
        elif 'HE' in self.model_type and 'Baseline' in self.model_type:
            loss_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], nuclei_mask)
            loss_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
            loss_cell_seg = self.seg_criterion(outputs['he_cell_seg'], cell_mask)
            loss_cell_hv = self.hv_criterion(outputs['he_cell_hv'], cell_hv, cell_mask, self.device)
            
            total_loss = (loss_nuclei_seg + 2.0 * loss_nuclei_hv + 
                         loss_cell_seg + 2.0 * loss_cell_hv) / 2.0
            
            dice_nuclei = compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), nuclei_mask)
            dice_cell = compute_dice((outputs['he_cell_seg'] > 0.5).float(), cell_mask)
            
        else:
            # Flex
            loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], nuclei_mask)
            loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
            loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], cell_mask)
            loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], cell_hv, cell_mask, self.device)
            
            loss_mif_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], nuclei_mask)
            loss_mif_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
            loss_mif_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], cell_mask)
            loss_mif_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], cell_hv, cell_mask, self.device)
            
            total_loss = (
                loss_he_nuclei_seg + 2.0 * loss_he_nuclei_hv +
                loss_he_cell_seg + 2.0 * loss_he_cell_hv +
                loss_mif_nuclei_seg + 2.0 * loss_mif_nuclei_hv +
                loss_mif_cell_seg + 2.0 * loss_mif_cell_hv
            ) / 4.0
            
            dice_nuclei = (compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), nuclei_mask) +
                          compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), nuclei_mask)) / 2
            dice_cell = (compute_dice((outputs['he_cell_seg'] > 0.5).float(), cell_mask) +
                        compute_dice((outputs['mif_cell_seg'] > 0.5).float(), cell_mask)) / 2
        
        metrics = {
            'dice_nuclei': dice_nuclei,
            'dice_cell': dice_cell
        }
        
        return total_loss.item(), metrics
    
    def _val_step_dual(self, batch):
        """Validation step for dual encoder"""
        he_img = prepare_he_input(batch['he_image'].to(self.device))
        mif_img = prepare_mif_input(batch['mif_image'].to(self.device))
        
        nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(self.device)
        cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(self.device)
        nuclei_hv = batch['he_nuclei_hv'].to(self.device)
        cell_hv = batch['he_cell_hv'].to(self.device)
        
        he_img = self.preprocessor.percentile_normalize(he_img)
        mif_img = self.preprocessor.percentile_normalize(mif_img)
        
        outputs = self.model(he_img, mif_img)
        
        # Compute losses
        loss_he_nuclei_seg = self.seg_criterion(outputs['he_nuclei_seg'], nuclei_mask)
        loss_he_nuclei_hv = self.hv_criterion(outputs['he_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
        loss_he_cell_seg = self.seg_criterion(outputs['he_cell_seg'], cell_mask)
        loss_he_cell_hv = self.hv_criterion(outputs['he_cell_hv'], cell_hv, cell_mask, self.device)
        
        loss_mif_nuclei_seg = self.seg_criterion(outputs['mif_nuclei_seg'], nuclei_mask)
        loss_mif_nuclei_hv = self.hv_criterion(outputs['mif_nuclei_hv'], nuclei_hv, nuclei_mask, self.device)
        loss_mif_cell_seg = self.seg_criterion(outputs['mif_cell_seg'], cell_mask)
        loss_mif_cell_hv = self.hv_criterion(outputs['mif_cell_hv'], cell_hv, cell_mask, self.device)
        
        total_loss = (
            loss_he_nuclei_seg + 2.0 * loss_he_nuclei_hv +
            loss_he_cell_seg + 2.0 * loss_he_cell_hv +
            loss_mif_nuclei_seg + 2.0 * loss_mif_nuclei_hv +
            loss_mif_cell_seg + 2.0 * loss_mif_cell_hv
        ) / 4.0
        
        metrics = {
            'dice_nuclei': (compute_dice((outputs['he_nuclei_seg'] > 0.5).float(), nuclei_mask) +
                           compute_dice((outputs['mif_nuclei_seg'] > 0.5).float(), nuclei_mask)) / 2,
            'dice_cell': (compute_dice((outputs['he_cell_seg'] > 0.5).float(), cell_mask) +
                         compute_dice((outputs['mif_cell_seg'] > 0.5).float(), cell_mask)) / 2
        }
        
        return total_loss.item(), metrics
    
    def train(self, epochs, use_augmentations=True):
        """Main training loop"""
        print("=" * 80)
        print(f"Training {self.model_type}")
        print(f"Epochs: {epochs} | Device: {self.device}")
        print(f"Augmentations: {use_augmentations} | W&B: {self.use_wandb}")
        print("=" * 80)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch, use_augmentations)
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Dice: N={train_metrics['dice_nuclei']:.4f} C={train_metrics['dice_cell']:.4f}")
            print(f"  Val Dice:   N={val_metrics['dice_nuclei']:.4f} C={val_metrics['dice_cell']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/dice_nuclei": train_metrics['dice_nuclei'],
                    "train/dice_cell": train_metrics['dice_cell'],
                    "val/loss": val_loss,
                    "val/dice_nuclei": val_metrics['dice_nuclei'],
                    "val/dice_cell": val_metrics['dice_cell'],
                    "lr": self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = self.checkpoint_dir / f"{self.model_type}_{self.model.model_size}_best.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✅ Best model saved! Val loss: {val_loss:.4f}")
                
                if self.use_wandb:
                    wandb.log({"best_val_loss": self.best_val_loss})
        
        print(f"\n{'='*80}")
        print(f"Training Complete | Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}")
        
        if self.use_wandb:
            wandb.finish()
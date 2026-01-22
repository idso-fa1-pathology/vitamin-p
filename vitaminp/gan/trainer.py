"""
Trainer for Pix2Pix GAN with advanced losses and architecture

Features:
- Uses Pix2PixGenerator with attention gates
- Uses PatchGANDiscriminator with spectral norm
- Uses CombinedGeneratorLoss with perceptual + SSIM + gradient losses
- Mixed precision training support
- Gradient clipping for stability
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os

from .models import Pix2PixGenerator, PatchGANDiscriminator
from .losses import GANLoss, CombinedGeneratorLoss
from .utils import GANPreprocessing


class Pix2PixTrainer:
    """
    Trainer for Pix2Pix: H&E → Synthetic MIF
    
    Features:
    - Attention gates in generator
    - Perceptual + SSIM + gradient losses
    - Spectral normalization in discriminator
    - Mixed precision training
    - Gradient clipping
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on ('cuda' or 'cpu')
        lr_g: Generator learning rate (default: 2e-4)
        lr_d: Discriminator learning rate (default: 2e-4)
        lambda_l1: Weight for L1 loss (default: 100)
        lambda_perceptual: Weight for perceptual loss (default: 10)
        lambda_ssim: Weight for SSIM loss (default: 5)
        lambda_gradient: Weight for gradient loss (default: 5)
        use_perceptual: Enable perceptual loss (default: True)
        use_ssim: Enable SSIM loss (default: True)
        use_gradient: Enable gradient loss (default: True)
        use_attention: Enable attention gates (default: True)
        use_spectral_norm: Enable spectral norm in discriminator (default: True)
        n_residual_blocks: Number of residual blocks at bottleneck (default: 4)
        mixed_precision: Enable mixed precision training (default: True)
        gradient_clip: Gradient clipping value (default: 5.0, None to disable)
        use_wandb: Whether to use Weights & Biases logging
        project_name: W&B project name
        run_name: W&B run name
        checkpoint_dir: Directory to save checkpoints
    
    Example:
        >>> from vitaminp.gan import Pix2PixTrainer
        >>> trainer = Pix2PixTrainer(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     device='cuda',
        ...     use_perceptual=True,
        ...     use_attention=True,
        ...     mixed_precision=True
        ... )
        >>> trainer.train(epochs=250)
    """
    def __init__(
        self,
        train_loader,
        val_loader,
        device='cuda',
        lr_g=2e-4,
        lr_d=2e-4,
        lambda_l1=100,
        lambda_perceptual=10,
        lambda_ssim=5,
        lambda_gradient=5,
        use_perceptual=True,
        use_ssim=True,
        use_gradient=True,
        use_attention=True,
        use_spectral_norm=True,
        n_residual_blocks=4,
        mixed_precision=True,
        gradient_clip=5.0,
        use_wandb=True,
        project_name="pathology-gan-synthesis",
        run_name=None,
        checkpoint_dir="checkpoints"
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lambda_l1 = lambda_l1
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize models
        print("🏗️  Building models...")
        self.generator = Pix2PixGenerator(
            in_channels=3,
            out_channels=2,
            use_attention=use_attention,
            use_spectral_norm=False,  # Don't use SN in generator
            n_residual_blocks=n_residual_blocks
        ).to(device)
        
        self.discriminator = PatchGANDiscriminator(
            in_channels=5,
            use_spectral_norm=use_spectral_norm
        ).to(device)
        
        print(f"  ✓ Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"  ✓ Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        # Optimizers
        self.optimizer_G = Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Schedulers
        self.scheduler_G = ReduceLROnPlateau(
            self.optimizer_G, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.scheduler_D = ReduceLROnPlateau(
            self.optimizer_D, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss functions
        print("📊 Setting up loss functions...")
        self.criterion_combined = CombinedGeneratorLoss(
            lambda_l1=lambda_l1,
            lambda_perceptual=lambda_perceptual,
            lambda_ssim=lambda_ssim,
            lambda_gradient=lambda_gradient,
            use_perceptual=use_perceptual,
            use_ssim=use_ssim,
            use_gradient=use_gradient
        ).to(device)
        
        self.criterion_D = GANLoss().to(device)
        
        print(f"  ✓ L1 weight: {lambda_l1}")
        print(f"  ✓ Perceptual: {'Enabled' if use_perceptual else 'Disabled'} (λ={lambda_perceptual})")
        print(f"  ✓ SSIM: {'Enabled' if use_ssim else 'Disabled'} (λ={lambda_ssim})")
        print(f"  ✓ Gradient: {'Enabled' if use_gradient else 'Disabled'} (λ={lambda_gradient})")
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Preprocessing
        self.preprocessor = GANPreprocessing()
        
        # Initialize W&B if requested
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            
            if run_name is None:
                run_name = "Pix2Pix_HE2MIF"
            
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model": "Pix2Pix",
                    "architecture": "Attention U-Net + Spectral Norm PatchGAN",
                    "task": "H&E to MIF Translation",
                    "lr_G": lr_g,
                    "lr_D": lr_d,
                    "lambda_l1": lambda_l1,
                    "lambda_perceptual": lambda_perceptual,
                    "lambda_ssim": lambda_ssim,
                    "lambda_gradient": lambda_gradient,
                    "use_perceptual": use_perceptual,
                    "use_ssim": use_ssim,
                    "use_gradient": use_gradient,
                    "use_attention": use_attention,
                    "use_spectral_norm": use_spectral_norm,
                    "n_residual_blocks": n_residual_blocks,
                    "mixed_precision": mixed_precision,
                    "gradient_clip": gradient_clip,
                    "batch_size": train_loader.batch_size,
                    "optimizer": "Adam",
                    "beta1": 0.5,
                    "beta2": 0.999,
                    "device": device,
                }
            )
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Train one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        metrics = {
            'g_total': 0,
            'g_gan': 0,
            'g_l1': 0,
            'g_perceptual': 0,
            'g_ssim': 0,
            'g_gradient': 0,
            'd_loss': 0,
        }
        
        pbar = tqdm(self.train_loader, desc='Training', ncols=140)
        
        for batch in pbar:
            he_img = batch['he_image'].to(self.device)
            mif_img = batch['mif_image'].to(self.device)
            
            # Normalize
            he_img = self.preprocessor.percentile_normalize(he_img)
            mif_img = self.preprocessor.percentile_normalize(mif_img)
            
            # Scale to [-1, 1] for GAN
            he_img = self.preprocessor.to_gan_range(he_img)
            mif_img = self.preprocessor.to_gan_range(mif_img)
            
            # ============================================================
            # Train Generator
            # ============================================================
            self.optimizer_G.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    # Generate fake MIF
                    fake_mif = self.generator(he_img)
                    
                    # Discriminator prediction
                    pred_fake = self.discriminator(he_img, fake_mif)
                    
                    # Combined loss
                    losses = self.criterion_combined(fake_mif, mif_img, pred_fake)
                    loss_G = losses['total']
                
                # Backward with gradient scaling
                self.scaler.scale(loss_G).backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer_G)
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
            else:
                # Generate fake MIF
                fake_mif = self.generator(he_img)
                
                # Discriminator prediction
                pred_fake = self.discriminator(he_img, fake_mif)
                
                # Combined loss
                losses = self.criterion_combined(fake_mif, mif_img, pred_fake)
                loss_G = losses['total']
                
                loss_G.backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip)
                
                self.optimizer_G.step()
            
            # ============================================================
            # Train Discriminator
            # ============================================================
            self.optimizer_D.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    # Real loss
                    pred_real = self.discriminator(he_img, mif_img)
                    loss_D_real = self.criterion_D(pred_real, True)
                    
                    # Fake loss
                    fake_mif_detached = fake_mif.detach()
                    pred_fake = self.discriminator(he_img, fake_mif_detached)
                    loss_D_fake = self.criterion_D(pred_fake, False)
                    
                    # Total discriminator loss
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                
                self.scaler.scale(loss_D).backward()
                
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer_D)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer_D)
                self.scaler.update()
            else:
                # Real loss
                pred_real = self.discriminator(he_img, mif_img)
                loss_D_real = self.criterion_D(pred_real, True)
                
                # Fake loss
                fake_mif_detached = fake_mif.detach()
                pred_fake = self.discriminator(he_img, fake_mif_detached)
                loss_D_fake = self.criterion_D(pred_fake, False)
                
                # Total discriminator loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                
                loss_D.backward()
                
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
                
                self.optimizer_D.step()
            
            # Accumulate metrics
            metrics['g_total'] += losses['total'].item()
            metrics['g_gan'] += losses['gan'].item()
            metrics['g_l1'] += losses['l1'].item()
            metrics['g_perceptual'] += losses['perceptual'].item()
            metrics['g_ssim'] += losses['ssim'].item()
            metrics['g_gradient'] += losses['gradient'].item()
            metrics['d_loss'] += loss_D.item()
            
            pbar.set_postfix({
                'G': f'{loss_G.item():.3f}',
                'D': f'{loss_D.item():.3f}',
                'L1': f'{losses["l1"].item():.3f}',
                'Perc': f'{losses["perceptual"].item():.3f}'
            })
            
            # Batch logging
            if self.use_wandb:
                self.wandb.log({
                    "batch/g_total": loss_G.item(),
                    "batch/g_gan": losses['gan'].item(),
                    "batch/g_l1": losses['l1'].item(),
                    "batch/g_perceptual": losses['perceptual'].item(),
                    "batch/g_ssim": losses['ssim'].item(),
                    "batch/g_gradient": losses['gradient'].item(),
                    "batch/d_loss": loss_D.item(),
                })
        
        # Average metrics
        n_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= n_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.generator.eval()
        self.discriminator.eval()
        
        metrics = {
            'g_total': 0,
            'g_gan': 0,
            'g_l1': 0,
            'g_perceptual': 0,
            'g_ssim': 0,
            'g_gradient': 0,
        }
        
        for batch in self.val_loader:
            he_img = batch['he_image'].to(self.device)
            mif_img = batch['mif_image'].to(self.device)
            
            he_img = self.preprocessor.percentile_normalize(he_img)
            mif_img = self.preprocessor.percentile_normalize(mif_img)
            
            he_img = self.preprocessor.to_gan_range(he_img)
            mif_img = self.preprocessor.to_gan_range(mif_img)
            
            if self.mixed_precision:
                with autocast():
                    fake_mif = self.generator(he_img)
                    pred_fake = self.discriminator(he_img, fake_mif)
                    losses = self.criterion_combined(fake_mif, mif_img, pred_fake)
            else:
                fake_mif = self.generator(he_img)
                pred_fake = self.discriminator(he_img, fake_mif)
                losses = self.criterion_combined(fake_mif, mif_img, pred_fake)
            
            metrics['g_total'] += losses['total'].item()
            metrics['g_gan'] += losses['gan'].item()
            metrics['g_l1'] += losses['l1'].item()
            metrics['g_perceptual'] += losses['perceptual'].item()
            metrics['g_ssim'] += losses['ssim'].item()
            metrics['g_gradient'] += losses['gradient'].item()
        
        # Average metrics
        n_batches = len(self.val_loader)
        for key in metrics:
            metrics[key] /= n_batches
        
        return metrics
    
    def train(self, epochs):
        """
        Train the model for specified number of epochs
        
        Args:
            epochs: Number of epochs to train
        """
        print("=" * 90)
        print(f"{'PIX2PIX TRAINING: H&E → SYNTHETIC MIF':^90}")
        print("=" * 90)
        print(f"Epochs: {epochs} | Mixed Precision: {self.mixed_precision} | Gradient Clip: {self.gradient_clip}")
        print("=" * 90)
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update schedulers
            self.scheduler_G.step(val_metrics['g_l1'])
            self.scheduler_D.step(val_metrics['g_l1'])
            
            # Epoch logging
            if self.use_wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    "train/g_total": train_metrics['g_total'],
                    "train/g_gan": train_metrics['g_gan'],
                    "train/g_l1": train_metrics['g_l1'],
                    "train/g_perceptual": train_metrics['g_perceptual'],
                    "train/g_ssim": train_metrics['g_ssim'],
                    "train/g_gradient": train_metrics['g_gradient'],
                    "train/d_loss": train_metrics['d_loss'],
                    "val/g_total": val_metrics['g_total'],
                    "val/g_l1": val_metrics['g_l1'],
                    "val/g_perceptual": val_metrics['g_perceptual'],
                    "lr_G": self.optimizer_G.param_groups[0]['lr'],
                    "lr_D": self.optimizer_D.param_groups[0]['lr'],
                })
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train - G: {train_metrics['g_total']:.4f} | D: {train_metrics['d_loss']:.4f} | "
                  f"L1: {train_metrics['g_l1']:.4f} | Perc: {train_metrics['g_perceptual']:.4f}")
            print(f"Val   - G: {val_metrics['g_total']:.4f} | "
                  f"L1: {val_metrics['g_l1']:.4f} | Perc: {val_metrics['g_perceptual']:.4f}")
            
            # Save best model
            if val_metrics['g_l1'] < self.best_val_loss:
                self.best_val_loss = val_metrics['g_l1']
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    'pix2pix_he_to_mif_best.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                    'val_l1_loss': val_metrics['g_l1'],
                    'scaler': self.scaler.state_dict() if self.scaler else None,
                }, checkpoint_path)
                
                if self.use_wandb:
                    self.wandb.log({"best_val_l1_loss": self.best_val_loss})
                
                print(f'✅ Best model saved! Val L1: {val_metrics["g_l1"]:.4f}\n')
        
        print(f"\n{'='*90}\nTraining Complete | Best Val L1 Loss: {self.best_val_loss:.4f}\n{'='*90}")
        
        # Save final model
        final_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            'pix2pix_he_to_mif_final.pth'
        )
        torch.save({
            'epoch': epochs,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, final_checkpoint_path)
        print(f"✅ Final model saved: {final_checkpoint_path}")
        
        if self.use_wandb:
            self.wandb.finish()
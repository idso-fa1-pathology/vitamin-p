"""
Trainer for Pix2Pix GAN
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

from .models import Pix2PixGenerator, PatchGANDiscriminator
from .losses import GANLoss
from .utils import GANPreprocessing


class Pix2PixTrainer:
    """
    Trainer for Pix2Pix: H&E → Synthetic MIF
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on ('cuda' or 'cpu')
        lr_g: Generator learning rate
        lr_d: Discriminator learning rate
        lambda_l1: Weight for L1 loss (default: 100)
        use_wandb: Whether to use Weights & Biases logging
        project_name: W&B project name
        run_name: W&B run name
        checkpoint_dir: Directory to save checkpoints
    
    Example:
        >>> from gan import Pix2PixTrainer
        >>> trainer = Pix2PixTrainer(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     device='cuda',
        ...     lambda_l1=100
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
        use_wandb=True,
        project_name="pathology-gan-synthesis",
        run_name=None,
        checkpoint_dir="checkpoints"
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lambda_l1 = lambda_l1
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize models
        self.generator = Pix2PixGenerator(in_channels=3, out_channels=2).to(device)
        self.discriminator = PatchGANDiscriminator(in_channels=5).to(device)
        
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
        self.criterion_GAN = GANLoss().to(device)
        self.criterion_L1 = nn.L1Loss().to(device)
        
        # Preprocessing
        self.preprocessor = GANPreprocessing()
        
        # Initialize W&B if requested
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            
            if run_name is None:
                run_name = "Pix2Pix_HE2MIF_Generator"
            
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model": "Pix2Pix",
                    "architecture": "UNet Generator + PatchGAN Discriminator",
                    "task": "H&E to MIF Translation",
                    "lr_G": lr_g,
                    "lr_D": lr_d,
                    "lambda_l1": lambda_l1,
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
            'g_loss': 0,
            'd_loss': 0,
            'l1_loss': 0,
            'gan_loss': 0,
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
            
            # Generate fake MIF
            fake_mif = self.generator(he_img)
            
            # GAN loss (fool discriminator)
            pred_fake = self.discriminator(he_img, fake_mif)
            loss_GAN = self.criterion_GAN(pred_fake, True)
            
            # L1 loss (pixel-wise similarity)
            loss_L1 = self.criterion_L1(fake_mif, mif_img)
            
            # Total generator loss
            loss_G = loss_GAN + self.lambda_l1 * loss_L1
            
            loss_G.backward()
            self.optimizer_G.step()
            
            # ============================================================
            # Train Discriminator
            # ============================================================
            self.optimizer_D.zero_grad()
            
            # Real loss
            pred_real = self.discriminator(he_img, mif_img)
            loss_D_real = self.criterion_GAN(pred_real, True)
            
            # Fake loss
            fake_mif_detached = fake_mif.detach()
            pred_fake = self.discriminator(he_img, fake_mif_detached)
            loss_D_fake = self.criterion_GAN(pred_fake, False)
            
            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            loss_D.backward()
            self.optimizer_D.step()
            
            # Accumulate metrics
            metrics['g_loss'] += loss_G.item()
            metrics['d_loss'] += loss_D.item()
            metrics['l1_loss'] += loss_L1.item()
            metrics['gan_loss'] += loss_GAN.item()
            
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}',
                'L1': f'{loss_L1.item():.4f}'
            })
            
            # Batch logging
            if self.use_wandb:
                self.wandb.log({
                    "batch/g_loss": loss_G.item(),
                    "batch/d_loss": loss_D.item(),
                    "batch/gan_loss": loss_GAN.item(),
                    "batch/l1_loss": loss_L1.item(),
                    "batch/d_real_loss": loss_D_real.item(),
                    "batch/d_fake_loss": loss_D_fake.item(),
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
            'l1_loss': 0,
            'gan_loss': 0,
            'g_loss': 0,
        }
        
        for batch in self.val_loader:
            he_img = batch['he_image'].to(self.device)
            mif_img = batch['mif_image'].to(self.device)
            
            he_img = self.preprocessor.percentile_normalize(he_img)
            mif_img = self.preprocessor.percentile_normalize(mif_img)
            
            he_img = self.preprocessor.to_gan_range(he_img)
            mif_img = self.preprocessor.to_gan_range(mif_img)
            
            fake_mif = self.generator(he_img)
            
            # Generator losses
            pred_fake = self.discriminator(he_img, fake_mif)
            loss_GAN = self.criterion_GAN(pred_fake, True)
            loss_L1 = self.criterion_L1(fake_mif, mif_img)
            loss_G = loss_GAN + self.lambda_l1 * loss_L1
            
            metrics['l1_loss'] += loss_L1.item()
            metrics['gan_loss'] += loss_GAN.item()
            metrics['g_loss'] += loss_G.item()
        
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
        print("=" * 80)
        print(f"{'PIX2PIX TRAINING: H&E → SYNTHETIC MIF':^80}")
        print("=" * 80)
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs} | LR: 2e-4 | Lambda L1: {self.lambda_l1}")
        print("=" * 80)
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update schedulers
            self.scheduler_G.step(val_metrics['l1_loss'])
            self.scheduler_D.step(val_metrics['l1_loss'])
            
            # Epoch logging
            if self.use_wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    "train/g_loss": train_metrics['g_loss'],
                    "train/d_loss": train_metrics['d_loss'],
                    "train/gan_loss": train_metrics['gan_loss'],
                    "train/l1_loss": train_metrics['l1_loss'],
                    "val/g_loss": val_metrics['g_loss'],
                    "val/gan_loss": val_metrics['gan_loss'],
                    "val/l1_loss": val_metrics['l1_loss'],
                    "lr_G": self.optimizer_G.param_groups[0]['lr'],
                    "lr_D": self.optimizer_D.param_groups[0]['lr'],
                })
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train - G Loss: {train_metrics['g_loss']:.4f} | D Loss: {train_metrics['d_loss']:.4f} | "
                  f"GAN: {train_metrics['gan_loss']:.4f} | L1: {train_metrics['l1_loss']:.4f}")
            print(f"Val   - G Loss: {val_metrics['g_loss']:.4f} | "
                  f"GAN: {val_metrics['gan_loss']:.4f} | L1: {val_metrics['l1_loss']:.4f}")
            
            # Save best model
            if val_metrics['l1_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['l1_loss']
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
                    'val_l1_loss': val_metrics['l1_loss'],
                }, checkpoint_path)
                
                if self.use_wandb:
                    self.wandb.log({"best_val_l1_loss": self.best_val_loss})
                
                print(f'✅ Best model saved! Val L1: {val_metrics["l1_loss"]:.4f}\n')
        
        print(f"\n{'='*80}\nTraining Complete | Best Val L1 Loss: {self.best_val_loss:.4f}\n{'='*80}")
        
        if self.use_wandb:
            self.wandb.finish()
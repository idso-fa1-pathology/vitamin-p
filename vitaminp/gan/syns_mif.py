# ============================================================================
# PIX2PIX: H&E → SYNTHETIC MIF GENERATOR
# Complete training pipeline ready for Kubernetes
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import and create dataloaders
from crc_dataset import Config, create_dataloaders

config = Config("config.yaml")
config.print_config()

train_loader, val_loader, test_loader = create_dataloaders(config)

print("\n✅ Ready to use!")

# ============================================================================
# PIX2PIX ARCHITECTURE
# ============================================================================

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    """Pix2Pix Generator: H&E (3ch) → MIF (2ch)"""
    def __init__(self, in_channels=3, out_channels=2):
        super(GeneratorUNet, self).__init__()

        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 512 → 256
        self.down2 = UNetDown(64, 128)                            # 256 → 128
        self.down3 = UNetDown(128, 256)                           # 128 → 64
        self.down4 = UNetDown(256, 512, dropout=0.5)              # 64 → 32
        self.down5 = UNetDown(512, 512, dropout=0.5)              # 32 → 16
        self.down6 = UNetDown(512, 512, dropout=0.5)              # 16 → 8
        self.down7 = UNetDown(512, 512, dropout=0.5)              # 8 → 4
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)  # 4 → 2

        # Decoder (upsampling)
        self.up1 = UNetUp(512, 512, dropout=0.5)                  # 2 → 4
        self.up2 = UNetUp(1024, 512, dropout=0.5)                 # 4 → 8
        self.up3 = UNetUp(1024, 512, dropout=0.5)                 # 8 → 16
        self.up4 = UNetUp(1024, 512, dropout=0.5)                 # 16 → 32
        self.up5 = UNetUp(1024, 256)                              # 32 → 64
        self.up6 = UNetUp(512, 128)                               # 64 → 128
        self.up7 = UNetUp(256, 64)                                # 128 → 256

        # Final layer
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),                          # 256 → 512
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)

class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, in_channels=5):  # 3 (H&E) + 2 (MIF) = 5
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate H&E and MIF
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# ============================================================================
# LOSSES
# ============================================================================

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)

# ============================================================================
# PREPROCESSING
# ============================================================================

class SimplePreprocessing:
    def percentile_normalize(self, img):
        if img.dim() == 4:
            B = img.shape[0]
            normalized = []
            for i in range(B):
                single_img = img[i]
                p1 = torch.quantile(single_img, 0.01)
                p99 = torch.quantile(single_img, 0.99)
                norm_img = (single_img - p1) / (p99 - p1 + 1e-8)
                normalized.append(torch.clamp(norm_img, 0, 1))
            return torch.stack(normalized, dim=0)
        else:
            p1 = torch.quantile(img, 0.01)
            p99 = torch.quantile(img, 0.99)
            img = (img - p1) / (p99 - p1 + 1e-8)
            return torch.clamp(img, 0, 1)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_pix2pix(train_loader, val_loader, epochs=100, device='cuda', 
                  lambda_l1=100):
    """
    Train Pix2Pix: H&E → Synthetic MIF
    
    Args:
        train_loader: DataLoader with H&E + MIF pairs
        val_loader: Validation DataLoader
        epochs: Number of training epochs
        device: 'cuda' or 'cpu'
        lambda_l1: Weight for L1 loss (default: 100)
    """
    
    # Initialize models
    generator = GeneratorUNet(in_channels=3, out_channels=2).to(device)
    discriminator = Discriminator(in_channels=5).to(device)
    
    # Optimizers
    optimizer_G = Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Schedulers
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Loss functions
    criterion_GAN = GANLoss().to(device)
    criterion_L1 = nn.L1Loss().to(device)
    
    # Preprocessing
    preprocessor = SimplePreprocessing()
    
    # W&B
    wandb.init(
        project="pathology-gan-synthesis",
        name="Pix2Pix_HE2MIF_Generator",
        config={
            "model": "Pix2Pix",
            "architecture": "UNet Generator + PatchGAN Discriminator",
            "task": "H&E to MIF Translation",
            "epochs": epochs,
            "lr_G": 2e-4,
            "lr_D": 2e-4,
            "lambda_l1": lambda_l1,
            "batch_size": train_loader.batch_size,
            "optimizer": "Adam",
            "beta1": 0.5,
            "beta2": 0.999,
            "device": device,
        }
    )
    
    best_val_loss = float('inf')
    
    print("=" * 80)
    print(f"{'PIX2PIX TRAINING: H&E → SYNTHETIC MIF':^80}")
    print("=" * 80)
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs} | LR: 2e-4 | Lambda L1: {lambda_l1}")
    print("=" * 80)
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        # Training metrics
        train_g_loss = 0
        train_d_loss = 0
        train_l1_loss = 0
        train_gan_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', ncols=140)
        
        for batch in pbar:
            he_img = batch['he_image'].to(device)
            mif_img = batch['mif_image'].to(device)
            
            # Normalize
            he_img = preprocessor.percentile_normalize(he_img)
            mif_img = preprocessor.percentile_normalize(mif_img)
            
            # Scale to [-1, 1] for GAN
            he_img = he_img * 2 - 1
            mif_img = mif_img * 2 - 1
            
            batch_size = he_img.size(0)
            
            # ============================================================
            # Train Generator
            # ============================================================
            optimizer_G.zero_grad()
            
            # Generate fake MIF
            fake_mif = generator(he_img)
            
            # GAN loss (fool discriminator)
            pred_fake = discriminator(he_img, fake_mif)
            loss_GAN = criterion_GAN(pred_fake, True)
            
            # L1 loss (pixel-wise similarity)
            loss_L1 = criterion_L1(fake_mif, mif_img)
            
            # Total generator loss
            loss_G = loss_GAN + lambda_l1 * loss_L1
            
            loss_G.backward()
            optimizer_G.step()
            
            # ============================================================
            # Train Discriminator
            # ============================================================
            optimizer_D.zero_grad()
            
            # Real loss
            pred_real = discriminator(he_img, mif_img)
            loss_D_real = criterion_GAN(pred_real, True)
            
            # Fake loss
            fake_mif_detached = fake_mif.detach()
            pred_fake = discriminator(he_img, fake_mif_detached)
            loss_D_fake = criterion_GAN(pred_fake, False)
            
            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            loss_D.backward()
            optimizer_D.step()
            
            # Accumulate metrics
            train_g_loss += loss_G.item()
            train_d_loss += loss_D.item()
            train_l1_loss += loss_L1.item()
            train_gan_loss += loss_GAN.item()
            
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}',
                'L1': f'{loss_L1.item():.4f}'
            })
            
            # Batch logging
            wandb.log({
                "batch/g_loss": loss_G.item(),
                "batch/d_loss": loss_D.item(),
                "batch/gan_loss": loss_GAN.item(),
                "batch/l1_loss": loss_L1.item(),
                "batch/d_real_loss": loss_D_real.item(),
                "batch/d_fake_loss": loss_D_fake.item(),
            })
        
        # Average training metrics
        n_batches = len(train_loader)
        train_g_loss /= n_batches
        train_d_loss /= n_batches
        train_l1_loss /= n_batches
        train_gan_loss /= n_batches
        
        # ============================================================
        # Validation
        # ============================================================
        generator.eval()
        discriminator.eval()
        val_l1_loss = 0
        val_gan_loss = 0
        val_g_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                he_img = batch['he_image'].to(device)
                mif_img = batch['mif_image'].to(device)
                
                he_img = preprocessor.percentile_normalize(he_img)
                mif_img = preprocessor.percentile_normalize(mif_img)
                
                he_img = he_img * 2 - 1
                mif_img = mif_img * 2 - 1
                
                fake_mif = generator(he_img)
                
                # Generator losses
                pred_fake = discriminator(he_img, fake_mif)
                loss_GAN = criterion_GAN(pred_fake, True)
                loss_L1 = criterion_L1(fake_mif, mif_img)
                loss_G = loss_GAN + lambda_l1 * loss_L1
                
                val_l1_loss += loss_L1.item()
                val_gan_loss += loss_GAN.item()
                val_g_loss += loss_G.item()
        
        val_l1_loss /= len(val_loader)
        val_gan_loss /= len(val_loader)
        val_g_loss /= len(val_loader)
        
        # Epoch logging
        wandb.log({
            "epoch": epoch + 1,
            "train/g_loss": train_g_loss,
            "train/d_loss": train_d_loss,
            "train/gan_loss": train_gan_loss,
            "train/l1_loss": train_l1_loss,
            "val/g_loss": val_g_loss,
            "val/gan_loss": val_gan_loss,
            "val/l1_loss": val_l1_loss,
            "lr_G": optimizer_G.param_groups[0]['lr'],
            "lr_D": optimizer_D.param_groups[0]['lr'],
        })
        
        # Logging
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train - G Loss: {train_g_loss:.4f} | D Loss: {train_d_loss:.4f} | GAN: {train_gan_loss:.4f} | L1: {train_l1_loss:.4f}")
        print(f"Val   - G Loss: {val_g_loss:.4f} | GAN: {val_gan_loss:.4f} | L1: {val_l1_loss:.4f}")
        
        scheduler_G.step(val_l1_loss)
        scheduler_D.step(val_l1_loss)
        
        # Save best model
        if val_l1_loss < best_val_loss:
            best_val_loss = val_l1_loss
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'val_l1_loss': val_l1_loss,
            }, 'pix2pix_he_to_mif_best.pth')
            wandb.log({"best_val_l1_loss": best_val_loss})
            print(f'✅ Best model saved! Val L1: {val_l1_loss:.4f}\n')
    
    print("\n" + "=" * 80)
    print(f"{'TRAINING COMPLETE':^80}")
    print(f"Best Val L1 Loss: {best_val_loss:.4f}")
    print("=" * 80)
    
    wandb.finish()
    
    return generator, discriminator

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "=" * 80)
    print(f"{'PIX2PIX: H&E → SYNTHETIC MIF GENERATOR':^80}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Ready to train!")
    print("=" * 80)
    
    # Train the generator
    if 'train_loader' in locals():
        generator, discriminator = train_pix2pix(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=250,
            device=device,
            lambda_l1=100
        )
        print("\n✅ Pix2Pix training complete!")
        print("Model saved as: pix2pix_he_to_mif_best.pth")
    else:
        print("⚠️  train_loader and val_loader not found!")
        print("Run this after loading your data with create_dataloaders()")
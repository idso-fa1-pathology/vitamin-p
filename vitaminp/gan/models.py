"""
Pix2Pix models for H&E → MIF translation
"""

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """Downsampling block for U-Net encoder"""
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
    """Upsampling block for U-Net decoder"""
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


class Pix2PixGenerator(nn.Module):
    """
    Pix2Pix U-Net Generator: H&E (3ch) → MIF (2ch)
    
    Architecture:
    - 8-layer encoder with skip connections
    - 7-layer decoder with skip connections
    - Output: 2-channel MIF image with Tanh activation
    
    Args:
        in_channels: Input channels (default: 3 for H&E)
        out_channels: Output channels (default: 2 for MIF)
    
    Example:
        >>> generator = Pix2PixGenerator(in_channels=3, out_channels=2)
        >>> he_img = torch.randn(2, 3, 512, 512)
        >>> synthetic_mif = generator(he_img)
        >>> print(synthetic_mif.shape)  # (2, 2, 512, 512)
    """
    def __init__(self, in_channels=3, out_channels=2):
        super(Pix2PixGenerator, self).__init__()

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
        """
        Forward pass
        
        Args:
            x: (B, 3, 512, 512) - H&E image normalized to [-1, 1]
        
        Returns:
            (B, 2, 512, 512) - Synthetic MIF image in range [-1, 1]
        """
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


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix
    
    Classifies 70x70 patches as real/fake instead of the entire image.
    
    Args:
        in_channels: Input channels (default: 5 = 3 H&E + 2 MIF)
    
    Example:
        >>> discriminator = PatchGANDiscriminator(in_channels=5)
        >>> he_img = torch.randn(2, 3, 512, 512)
        >>> mif_img = torch.randn(2, 2, 512, 512)
        >>> pred = discriminator(he_img, mif_img)
        >>> print(pred.shape)  # (2, 1, 30, 30) - patch predictions
    """
    def __init__(self, in_channels=5):  # 3 (H&E) + 2 (MIF) = 5
        super(PatchGANDiscriminator, self).__init__()

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
        """
        Forward pass
        
        Args:
            img_A: (B, 3, H, W) - H&E image
            img_B: (B, 2, H, W) - MIF image (real or generated)
        
        Returns:
            (B, 1, 30, 30) - Patch-wise predictions (logits)
        """
        # Concatenate H&E and MIF
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
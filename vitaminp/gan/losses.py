"""
Loss functions for Pix2Pix GAN training

Features:
- Perceptual loss (VGG-based) for better visual quality
- SSIM loss for structural similarity
- Gradient loss for edge preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GANLoss(nn.Module):
    """
    GAN loss for generator and discriminator training
    Uses BCEWithLogitsLoss for stable training.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, prediction, target_is_real):
        """
        Args:
            prediction: Discriminator output (logits)
            target_is_real: If True, target is real (1), else fake (0)
        Returns:
            Loss value
        """
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    
    Compares high-level features instead of just pixels.
    Helps generate more realistic textures and structures.
    
    Args:
        layers: Which VGG layers to use (default: ['relu2_2', 'relu3_4', 'relu4_4'])
        weights: Weights for each layer (default: [1.0, 1.0, 1.0])
    
    Example:
        >>> perceptual = PerceptualLoss().cuda()
        >>> fake_mif = generator(he_img)
        >>> loss = perceptual(fake_mif, real_mif)
    """
    def __init__(self, layers=['relu2_2', 'relu3_4', 'relu4_4'], weights=[1.0, 1.0, 1.0]):
        super().__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Layer mapping
        layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
        }
        
        # Extract required layers
        self.layers = layers
        self.weights = weights
        self.vgg_layers = nn.ModuleList()
        
        prev_idx = 0
        for layer_name in layers:
            idx = layer_map[layer_name]
            self.vgg_layers.append(vgg[prev_idx:idx+1])
            prev_idx = idx + 1
        
        self.criterion = nn.L1Loss()
    
    def forward(self, fake, real):
        """
        Args:
            fake: Generated MIF (B, 2, H, W) in range [-1, 1]
            real: Real MIF (B, 2, H, W) in range [-1, 1]
        Returns:
            Perceptual loss
        """
        # Convert 2-channel MIF to 3-channel for VGG
        # Repeat channel dimension: (B, 2, H, W) → (B, 3, H, W)
        fake_rgb = fake.repeat(1, 2, 1, 1) if fake.shape[1] == 2 else fake
        fake_rgb = fake_rgb[:, :3, :, :]  # Take first 3 channels
        
        real_rgb = real.repeat(1, 2, 1, 1) if real.shape[1] == 2 else real
        real_rgb = real_rgb[:, :3, :, :]
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(fake.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(fake.device)
        
        fake_rgb = (fake_rgb * 0.5 + 0.5 - mean) / std
        real_rgb = (real_rgb * 0.5 + 0.5 - mean) / std
        
        # Extract features and compute loss
        loss = 0.0
        fake_feat = fake_rgb
        real_feat = real_rgb
        
        for vgg_layer, weight in zip(self.vgg_layers, self.weights):
            fake_feat = vgg_layer(fake_feat)
            real_feat = vgg_layer(real_feat)
            loss += weight * self.criterion(fake_feat, real_feat)
        
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss
    
    Measures structural similarity between images.
    Better than L1 for perceptual quality.
    
    Args:
        window_size: Size of gaussian window (default: 11)
        size_average: Average over batch (default: True)
    """
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        """Create gaussian kernel"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        """Create 2D gaussian window"""
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2):
        """Compute SSIM"""
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        """
        Args:
            img1, img2: Images in range [-1, 1]
        Returns:
            1 - SSIM (loss to minimize)
        """
        return 1 - self.ssim(img1, img2)


class GradientLoss(nn.Module):
    """
    Gradient loss for edge preservation
    
    Compares image gradients to preserve sharp edges and details.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    
    def gradient(self, x):
        """Compute image gradients"""
        # Horizontal gradient
        h_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        
        # Vertical gradient
        v_x = x[:, :, :-1, :] - x[:, :, 1:, :]
        
        return h_x, v_x
    
    def forward(self, fake, real):
        """
        Args:
            fake: Generated image
            real: Real image
        Returns:
            Gradient loss
        """
        fake_h, fake_v = self.gradient(fake)
        real_h, real_v = self.gradient(real)
        
        loss_h = self.criterion(fake_h, real_h)
        loss_v = self.criterion(fake_v, real_v)
        
        return loss_h + loss_v


class CombinedGeneratorLoss(nn.Module):
    """
    Combined loss for generator training
    
    Combines:
    - Adversarial loss (GAN)
    - L1 loss (pixel-wise)
    - Perceptual loss (VGG features)
    - SSIM loss (structural similarity)
    - Gradient loss (edge preservation)
    
    Args:
        lambda_l1: Weight for L1 loss (default: 100)
        lambda_perceptual: Weight for perceptual loss (default: 10)
        lambda_ssim: Weight for SSIM loss (default: 5)
        lambda_gradient: Weight for gradient loss (default: 5)
        use_perceptual: Enable perceptual loss (default: True)
        use_ssim: Enable SSIM loss (default: True)
        use_gradient: Enable gradient loss (default: True)
    
    Example:
        >>> criterion = CombinedGeneratorLoss(
        ...     lambda_l1=100,
        ...     lambda_perceptual=10,
        ...     use_perceptual=True
        ... ).cuda()
        >>> 
        >>> # In training loop:
        >>> fake_mif = generator(he_img)
        >>> pred_fake = discriminator(he_img, fake_mif)
        >>> 
        >>> loss_dict = criterion(
        ...     fake_mif=fake_mif,
        ...     real_mif=mif_img,
        ...     pred_fake=pred_fake
        ... )
        >>> 
        >>> total_loss = loss_dict['total']
        >>> total_loss.backward()
    """
    def __init__(
        self,
        lambda_l1=100,
        lambda_perceptual=10,
        lambda_ssim=5,
        lambda_gradient=5,
        use_perceptual=True,
        use_ssim=True,
        use_gradient=True
    ):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_gradient = lambda_gradient
        
        # Core losses
        self.criterion_GAN = GANLoss()
        self.criterion_L1 = nn.L1Loss()
        
        # Optional advanced losses
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.criterion_perceptual = PerceptualLoss()
        
        self.use_ssim = use_ssim
        if use_ssim:
            self.criterion_ssim = SSIMLoss()
        
        self.use_gradient = use_gradient
        if use_gradient:
            self.criterion_gradient = GradientLoss()
    
    def forward(self, fake_mif, real_mif, pred_fake):
        """
        Args:
            fake_mif: Generated MIF (B, 2, H, W)
            real_mif: Real MIF (B, 2, H, W)
            pred_fake: Discriminator prediction on fake (for adversarial loss)
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # 1. Adversarial loss
        losses['gan'] = self.criterion_GAN(pred_fake, True)
        
        # 2. L1 loss (weighted by channel)
        ch0_weight = 1.0
        ch1_weight = 2.0
        loss_L1_ch0 = self.criterion_L1(fake_mif[:, 0:1], real_mif[:, 0:1])
        loss_L1_ch1 = self.criterion_L1(fake_mif[:, 1:2], real_mif[:, 1:2])
        losses['l1'] = (ch0_weight * loss_L1_ch0 + ch1_weight * loss_L1_ch1) / (ch0_weight + ch1_weight)
        
        # 3. Perceptual loss (optional)
        if self.use_perceptual:
            losses['perceptual'] = self.criterion_perceptual(fake_mif, real_mif)
        else:
            losses['perceptual'] = torch.tensor(0.0).to(fake_mif.device)
        
        # 4. SSIM loss (optional)
        if self.use_ssim:
            losses['ssim'] = self.criterion_ssim(fake_mif, real_mif)
        else:
            losses['ssim'] = torch.tensor(0.0).to(fake_mif.device)
        
        # 5. Gradient loss (optional)
        if self.use_gradient:
            losses['gradient'] = self.criterion_gradient(fake_mif, real_mif)
        else:
            losses['gradient'] = torch.tensor(0.0).to(fake_mif.device)
        
        # Total loss
        losses['total'] = (
            losses['gan'] +
            self.lambda_l1 * losses['l1'] +
            self.lambda_perceptual * losses['perceptual'] +
            self.lambda_ssim * losses['ssim'] +
            self.lambda_gradient * losses['gradient']
        )
        
        return losses


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Create dummy tensors
    fake_mif = torch.randn(2, 2, 256, 256).cuda()
    real_mif = torch.randn(2, 2, 256, 256).cuda()
    pred_fake = torch.randn(2, 1, 30, 30).cuda()
    
    # Test combined loss
    criterion = CombinedGeneratorLoss(
        lambda_l1=100,
        lambda_perceptual=10,
        lambda_ssim=5,
        lambda_gradient=5,
        use_perceptual=True,
        use_ssim=True,
        use_gradient=True
    ).cuda()
    
    losses = criterion(fake_mif, real_mif, pred_fake)
    
    print("\nLoss values:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\n✅ All tests passed!")
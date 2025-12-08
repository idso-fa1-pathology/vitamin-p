import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal loss for segmentation"""
    def __init__(self, alpha=1, gamma=2, smooth=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        return dice_loss + focal_loss


class MSGELossMaps(nn.Module):
    """Multi-scale Gradient Error Loss for HV maps"""
    def __init__(self):
        super().__init__()

    def get_sobel_kernel(self, size: int, device: str):
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range, indexing="ij")
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_gradient_hv(self, hv: torch.Tensor, device: str):
        kernel_h, kernel_v = self.get_sobel_kernel(5, device=device)
        kernel_h = kernel_h.view(1, 1, 5, 5)
        kernel_v = kernel_v.view(1, 1, 5, 5)

        h_ch = hv[..., 0].unsqueeze(1)
        v_ch = hv[..., 1].unsqueeze(1)

        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()
        return dhv

    def forward(self, input: torch.Tensor, target: torch.Tensor, focus: torch.Tensor, device: str):
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        focus = focus.permute(0, 2, 3, 1)
        focus = focus[..., 0]

        focus = (focus[..., None]).float()
        focus = torch.cat([focus, focus], axis=-1).to(device)
        true_grad = self.get_gradient_hv(target, device)
        pred_grad = self.get_gradient_hv(input, device)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return loss
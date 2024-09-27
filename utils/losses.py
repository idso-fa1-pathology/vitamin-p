import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, hv_weight=1.0, hv_mse_weight=0.5, hv_msge_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.hv_weight = hv_weight
        self.hv_mse_weight = hv_mse_weight
        self.hv_msge_weight = hv_msge_weight

    def gradient(self, x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        grad_x = F.conv2d(x[:, 0:1], sobel_x, padding=1)
        grad_y = F.conv2d(x[:, 1:2], sobel_y, padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2)

    def hv_loss(self, pred, true):
        mse_loss = self.mse_loss(pred, true)
        pred_grad = self.gradient(pred)
        true_grad = self.gradient(true)
        msge_loss = self.mse_loss(pred_grad, true_grad)
        return self.hv_mse_weight * mse_loss + self.hv_msge_weight * msge_loss

    def forward(self, cell_seg_pred, nuclear_seg_pred, cell_hv_pred, nuclei_hv_pred, 
                cell_seg_true, nuclear_seg_true, cell_hv_true, nuclei_hv_true):
        cell_seg_true_binary = (cell_seg_true > 0).float()
        nuclear_seg_true_binary = (nuclear_seg_true > 0).float()

        cell_bce_loss = self.bce_loss(cell_seg_pred, cell_seg_true_binary)
        cell_dice_loss = self.dice_loss(cell_seg_pred, cell_seg_true_binary)
        cell_seg_loss = self.bce_weight * cell_bce_loss + self.dice_weight * cell_dice_loss

        nuclear_bce_loss = self.bce_loss(nuclear_seg_pred, nuclear_seg_true_binary)
        nuclear_dice_loss = self.dice_loss(nuclear_seg_pred, nuclear_seg_true_binary)
        nuclear_seg_loss = self.bce_weight * nuclear_bce_loss + self.dice_weight * nuclear_dice_loss

        cell_hv_loss = self.hv_loss(cell_hv_pred, cell_hv_true)
        nuclei_hv_loss = self.hv_loss(nuclei_hv_pred, nuclei_hv_true)

        total_loss = cell_seg_loss + nuclear_seg_loss + self.hv_weight * (cell_hv_loss + nuclei_hv_loss)

        return total_loss, cell_seg_loss, nuclear_seg_loss, cell_hv_loss, nuclei_hv_loss
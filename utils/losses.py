import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + 1e-5) / (TP + self.alpha*FP + self.beta*FN + 1e-5)  
        FocalTversky = (1 - Tversky)**self.gamma
        
        return FocalTversky

class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes, num_tissue_types):
        super(MultiTaskLoss, self).__init__()
        self.focal_tversky_loss = FocalTverskyLoss()
        self.cell_class_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.tissue_class_loss = nn.CrossEntropyLoss()
        self.global_cell_loss = nn.BCEWithLogitsLoss()
        self.hv_loss = nn.MSELoss()

    def forward(self, cell_seg_out, cell_class_out, tc_out, global_cell_out, hv_out,
                cell_seg_target, cell_class_target, tissue_type_target, global_cell_target, hv_target):
        
        # Adjust shapes if necessary
        if cell_class_target.dim() == 5:
            cell_class_target = cell_class_target.squeeze(1).permute(0, 3, 1, 2)
        if hv_target.dim() == 5:
            hv_target = hv_target.squeeze(1).permute(0, 3, 1, 2)

        # Cell segmentation loss (Focal Tversky)
        cell_seg_loss = self.focal_tversky_loss(cell_seg_out, cell_seg_target)
        
        # Cell classification loss
        cell_class_out_reshaped = cell_class_out.permute(0, 2, 3, 1).contiguous().view(-1, cell_class_out.size(1))
        cell_class_target_reshaped = cell_class_target.permute(0, 2, 3, 1).contiguous().view(-1, cell_class_target.size(1))
        cell_class_loss = self.cell_class_loss(cell_class_out_reshaped, cell_class_target_reshaped.argmax(dim=1))
        
        # Tissue classification loss
        tissue_class_loss = self.tissue_class_loss(tc_out, tissue_type_target.argmax(dim=1))
        
        # Global cell classification loss
        global_cell_loss = self.global_cell_loss(global_cell_out, global_cell_target)
        
        # HV branch: Horizontal and vertical distance map loss
        hv_loss = self.hv_loss(hv_out, hv_target)
        
        # Compute gradients of HV maps
        hv_out_grad = self.compute_gradients(hv_out)
        hv_target_grad = self.compute_gradients(hv_target)
        hv_grad_loss = self.hv_loss(hv_out_grad, hv_target_grad)
        
        # Combine losses
        total_loss = cell_seg_loss + cell_class_loss + tissue_class_loss + global_cell_loss + hv_loss + hv_grad_loss
        
        return total_loss, {
            'cell_seg_loss': cell_seg_loss.item(),
            'cell_class_loss': cell_class_loss.item(),
            'tissue_class_loss': tissue_class_loss.item(),
            'global_cell_loss': global_cell_loss.item(),
            'hv_loss': hv_loss.item(),
            'hv_grad_loss': hv_grad_loss.item()
        }

    def compute_gradients(self, tensor):
        # Compute gradients
        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        
        # Pad the gradients to match the original size
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        
        return torch.cat([dx, dy], dim=1)

def get_loss_function(num_classes, num_tissue_types):
    return MultiTaskLoss(num_classes, num_tissue_types)
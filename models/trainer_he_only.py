import torch
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import os

from .losses import DiceFocalLoss, MSGELossMaps


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_he_only_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, weight_decay=1e-4, checkpoint_dir='checkpoints'):
    """
    Train the HE-only single-modality pathology segmentation model
    
    Args:
        model: HEOnlyUNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Trained model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    seg_criterion = DiceFocalLoss(alpha=1, gamma=2)
    hv_criterion = MSGELossMaps()
    
    # Parameter grouping
    encoder_params = []
    decoder_params = []
    bn_params = []
    
    for name, param in model.named_parameters():
        if 'bn' in name or 'norm' in name:
            bn_params.append(param)
        elif any(x in name for x in ['conv1', 'bn1', 'layer']):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': lr * 0.1, 'weight_decay': weight_decay * 0.5},
        {'params': decoder_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': bn_params, 'lr': lr, 'weight_decay': 0.0}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, verbose=True, min_lr=1e-7
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train - HE Only]')
        
        for batch_idx, batch in enumerate(train_pbar):
            # Only H&E input
            he_img = batch['he_image'].to(device)
            
            # Ground truth masks and HV maps (HE only)
            he_nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(device)
            he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(device)
            he_nuclei_hv = batch['he_nuclei_hv'].to(device)
            he_cell_hv = batch['he_cell_hv'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with only HE image
            outputs = model(he_img)
            
            # Compute losses for HE branches only
            loss_nuclei_seg = seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
            loss_nuclei_hv = hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, device)
            
            loss_cell_seg = seg_criterion(outputs['he_cell_seg'], he_cell_mask)
            loss_cell_hv = hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, device)
            
            # Total loss
            total_loss = (loss_nuclei_seg + 0.6 * loss_nuclei_hv +
                         loss_cell_seg + 0.6 * loss_cell_hv) / 2.0
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val - HE Only]')
        
        with torch.no_grad():
            for batch in val_pbar:
                he_img = batch['he_image'].to(device)
                
                he_nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(device)
                he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(device)
                he_nuclei_hv = batch['he_nuclei_hv'].to(device)
                he_cell_hv = batch['he_cell_hv'].to(device)
                
                outputs = model(he_img)
                
                loss_nuclei_seg = seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
                loss_nuclei_hv = hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, device)
                
                loss_cell_seg = seg_criterion(outputs['he_cell_seg'], he_cell_mask)
                loss_cell_hv = hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, device)
                
                total_loss = (loss_nuclei_seg + 0.6 * loss_nuclei_hv +
                             loss_cell_seg + 0.6 * loss_cell_hv) / 2.0
                
                val_loss += total_loss.item()
                val_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'\nEpoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'{checkpoint_dir}/best_he_only_model.pth')
            print(f'💾 Best HE-only model saved! Val loss: {val_loss:.4f}')
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    return model
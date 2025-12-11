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


def detect_model_type(model):
    """
    Detect if model is dual-encoder, HE-only, or MIF-only
    
    Returns:
        'dual_encoder', 'he_only', or 'mif_only'
    """
    model_name = model.__class__.__name__
    
    if 'HEOnly' in model_name:
        return 'he_only'
    elif 'MIFOnly' in model_name:
        return 'mif_only'
    elif 'DualEncoder' in model_name or 'MultiModal' in model_name:
        return 'dual_encoder'
    else:
        # Default: check attributes
        if hasattr(model, 'he_backbone') and hasattr(model, 'mif_backbone'):
            return 'dual_encoder'
        elif hasattr(model, 'backbone'):
            # Single backbone - check forward signature
            import inspect
            sig = inspect.signature(model.forward)
            num_params = len(sig.parameters)
            if num_params == 2:  # self + one image
                # Could be either HE or MIF, assume HE
                return 'he_only'
        
        return 'dual_encoder'  # Default fallback


def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, weight_decay=1e-4, 
                experiment_name=None):
    """
    Universal trainer for dual-encoder, HE-only, and MIF-only models
    Auto-detects model type and trains accordingly
    
    Args:
        model: DualEncoderUNetV2, HEOnlyUNetV2, or MIFOnlyUNetV2
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        experiment_name: Optional experiment name for checkpoint saving
    
    Returns:
        Trained model
    """
    os.makedirs('checkpoints', exist_ok=True)
    
    set_seed(42)
    
    # Detect model type
    model_type = detect_model_type(model)
    print(f"📊 Detected model type: {model_type}")
    
    # Get backbone name from model
    backbone_name = getattr(model, 'backbone_name', 'unknown')
    print(f"🏗️  Backbone: {backbone_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    seg_criterion = DiceFocalLoss(alpha=1, gamma=2)
    hv_criterion = MSGELossMaps()
    
    # Parameter grouping - separate learning rates for different parts
    he_encoder_params = []
    mif_encoder_params = []
    shared_encoder_params = []
    decoder_params = []
    bn_params = []
    
    for name, param in model.named_parameters():
        if 'bn' in name or 'norm' in name:
            bn_params.append(param)
        elif 'he_' in name and any(x in name for x in ['conv1', 'bn1', 'layer', 'backbone']):
            he_encoder_params.append(param)
        elif 'mif_' in name and any(x in name for x in ['conv1', 'bn1', 'layer', 'backbone']):
            mif_encoder_params.append(param)
        elif 'shared_' in name or 'fusion' in name:
            shared_encoder_params.append(param)
        elif 'backbone' in name:
            # Single backbone models
            he_encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    # Build optimizer with non-empty parameter groups only
    param_groups = []
    
    if he_encoder_params:
        param_groups.append({'params': he_encoder_params, 'lr': lr * 0.1, 'weight_decay': weight_decay * 0.5})
    
    if mif_encoder_params:
        param_groups.append({'params': mif_encoder_params, 'lr': lr * 0.1, 'weight_decay': weight_decay * 0.5})
    
    if shared_encoder_params:
        param_groups.append({'params': shared_encoder_params, 'lr': lr * 0.5, 'weight_decay': weight_decay})
    
    if decoder_params:
        param_groups.append({'params': decoder_params, 'lr': lr, 'weight_decay': weight_decay})
    
    if bn_params:
        param_groups.append({'params': bn_params, 'lr': lr, 'weight_decay': 0.0})
    
    # If no parameter groups were created, add all parameters as one group
    if not param_groups:
        param_groups = [{'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay}]
    
    optimizer = optim.AdamW(param_groups)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Checkpoint name based on experiment name or model type + backbone
    if experiment_name:
        checkpoint_name = f'checkpoints/{experiment_name}_best.pth'
    else:
        checkpoint_name = f'checkpoints/{model_type}_{backbone_name}_best.pth'
    
    print(f"💾 Checkpoint will be saved as: {checkpoint_name}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_pbar):
            # Get inputs based on model type
            he_img = batch['he_image'].to(device)
            mif_img = batch['mif_image'].to(device)
            
            # Ground truth masks and HV maps
            he_nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(device)
            he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(device)
            mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(device)
            mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(device)
            
            he_nuclei_hv = batch['he_nuclei_hv'].to(device)
            he_cell_hv = batch['he_cell_hv'].to(device)
            mif_nuclei_hv = batch['mif_nuclei_hv'].to(device)
            mif_cell_hv = batch['mif_cell_hv'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - model type specific
            if model_type == 'dual_encoder':
                outputs = model(he_img, mif_img)
            elif model_type == 'he_only':
                outputs = model(he_img)
            elif model_type == 'mif_only':
                outputs = model(mif_img)
            
            # Compute losses based on model type
            if model_type == 'dual_encoder':
                # All 4 branches
                loss_he_nuclei_seg = seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
                loss_he_nuclei_hv = hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, device)
                
                loss_he_cell_seg = seg_criterion(outputs['he_cell_seg'], he_cell_mask)
                loss_he_cell_hv = hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, device)
                
                loss_mif_nuclei_seg = seg_criterion(outputs['mif_nuclei_seg'], mif_nuclei_mask)
                loss_mif_nuclei_hv = hv_criterion(outputs['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, device)
                
                loss_mif_cell_seg = seg_criterion(outputs['mif_cell_seg'], mif_cell_mask)
                loss_mif_cell_hv = hv_criterion(outputs['mif_cell_hv'], mif_cell_hv, mif_cell_mask, device)
                
                total_loss = (loss_he_nuclei_seg + 0.6 * loss_he_nuclei_hv +
                             loss_he_cell_seg + 0.6 * loss_he_cell_hv +
                             loss_mif_nuclei_seg + 0.6 * loss_mif_nuclei_hv +
                             loss_mif_cell_seg + 0.6 * loss_mif_cell_hv) / 4.0
            
            elif model_type == 'he_only':
                # Only HE branches
                loss_he_nuclei_seg = seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
                loss_he_nuclei_hv = hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, device)
                
                loss_he_cell_seg = seg_criterion(outputs['he_cell_seg'], he_cell_mask)
                loss_he_cell_hv = hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, device)
                
                total_loss = (loss_he_nuclei_seg + 0.6 * loss_he_nuclei_hv +
                             loss_he_cell_seg + 0.6 * loss_he_cell_hv) / 2.0
            
            elif model_type == 'mif_only':
                # Only MIF branches
                loss_mif_nuclei_seg = seg_criterion(outputs['mif_nuclei_seg'], mif_nuclei_mask)
                loss_mif_nuclei_hv = hv_criterion(outputs['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, device)
                
                loss_mif_cell_seg = seg_criterion(outputs['mif_cell_seg'], mif_cell_mask)
                loss_mif_cell_hv = hv_criterion(outputs['mif_cell_hv'], mif_cell_hv, mif_cell_mask, device)
                
                total_loss = (loss_mif_nuclei_seg + 0.6 * loss_mif_nuclei_hv +
                             loss_mif_cell_seg + 0.6 * loss_mif_cell_hv) / 2.0
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_pbar:
                he_img = batch['he_image'].to(device)
                mif_img = batch['mif_image'].to(device)
                
                he_nuclei_mask = batch['he_nuclei_mask'].float().unsqueeze(1).to(device)
                he_cell_mask = batch['he_cell_mask'].float().unsqueeze(1).to(device)
                mif_nuclei_mask = batch['mif_nuclei_mask'].float().unsqueeze(1).to(device)
                mif_cell_mask = batch['mif_cell_mask'].float().unsqueeze(1).to(device)
                
                he_nuclei_hv = batch['he_nuclei_hv'].to(device)
                he_cell_hv = batch['he_cell_hv'].to(device)
                mif_nuclei_hv = batch['mif_nuclei_hv'].to(device)
                mif_cell_hv = batch['mif_cell_hv'].to(device)
                
                # Forward pass - model type specific
                if model_type == 'dual_encoder':
                    outputs = model(he_img, mif_img)
                elif model_type == 'he_only':
                    outputs = model(he_img)
                elif model_type == 'mif_only':
                    outputs = model(mif_img)
                
                # Compute validation loss
                if model_type == 'dual_encoder':
                    loss_he_nuclei_seg = seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
                    loss_he_nuclei_hv = hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, device)
                    
                    loss_he_cell_seg = seg_criterion(outputs['he_cell_seg'], he_cell_mask)
                    loss_he_cell_hv = hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, device)
                    
                    loss_mif_nuclei_seg = seg_criterion(outputs['mif_nuclei_seg'], mif_nuclei_mask)
                    loss_mif_nuclei_hv = hv_criterion(outputs['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, device)
                    
                    loss_mif_cell_seg = seg_criterion(outputs['mif_cell_seg'], mif_cell_mask)
                    loss_mif_cell_hv = hv_criterion(outputs['mif_cell_hv'], mif_cell_hv, mif_cell_mask, device)
                    
                    total_loss = (loss_he_nuclei_seg + 0.6 * loss_he_nuclei_hv +
                                 loss_he_cell_seg + 0.6 * loss_he_cell_hv +
                                 loss_mif_nuclei_seg + 0.6 * loss_mif_nuclei_hv +
                                 loss_mif_cell_seg + 0.6 * loss_mif_cell_hv) / 4.0
                
                elif model_type == 'he_only':
                    loss_he_nuclei_seg = seg_criterion(outputs['he_nuclei_seg'], he_nuclei_mask)
                    loss_he_nuclei_hv = hv_criterion(outputs['he_nuclei_hv'], he_nuclei_hv, he_nuclei_mask, device)
                    
                    loss_he_cell_seg = seg_criterion(outputs['he_cell_seg'], he_cell_mask)
                    loss_he_cell_hv = hv_criterion(outputs['he_cell_hv'], he_cell_hv, he_cell_mask, device)
                    
                    total_loss = (loss_he_nuclei_seg + 0.6 * loss_he_nuclei_hv +
                                 loss_he_cell_seg + 0.6 * loss_he_cell_hv) / 2.0
                
                elif model_type == 'mif_only':
                    loss_mif_nuclei_seg = seg_criterion(outputs['mif_nuclei_seg'], mif_nuclei_mask)
                    loss_mif_nuclei_hv = hv_criterion(outputs['mif_nuclei_hv'], mif_nuclei_hv, mif_nuclei_mask, device)
                    
                    loss_mif_cell_seg = seg_criterion(outputs['mif_cell_seg'], mif_cell_mask)
                    loss_mif_cell_hv = hv_criterion(outputs['mif_cell_hv'], mif_cell_hv, mif_cell_mask, device)
                    
                    total_loss = (loss_mif_nuclei_seg + 0.6 * loss_mif_nuclei_hv +
                                 loss_mif_cell_seg + 0.6 * loss_mif_cell_hv) / 2.0
                
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
            }, checkpoint_name)
            print(f'💾 Best model saved to {checkpoint_name}! Val loss: {val_loss:.4f}')
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    return model
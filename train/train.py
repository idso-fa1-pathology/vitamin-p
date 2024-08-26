import torch
from tqdm import tqdm
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from vitamin_p.models.losses import FocalDiceLoss
from vitamin_p.models.metrics import iou_score, calculate_object_based_metrics

def train_model(model, train_loader, val_loader, num_epochs=100, patience=20, save_dir='models'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = FocalDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model = None
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iou = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, masks in train_bar:
            images = images.float().to(device)
            masks = masks.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += iou_score(outputs, masks).item()
            
            # Calculate object-based metrics
            pred_masks = (outputs > 0.5).float()
            batch_precision, batch_recall, batch_f1 = 0, 0, 0
            for true_mask, pred_mask in zip(masks, pred_masks):
                p, r, f = calculate_object_based_metrics(true_mask, pred_mask)
                batch_precision += p
                batch_recall += r
                batch_f1 += f
            
            train_precision += batch_precision / len(masks)
            train_recall += batch_recall / len(masks)
            train_f1 += batch_f1 / len(masks)
            
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'iou': f"{iou_score(outputs, masks).item():.4f}",
                'precision': f"{batch_precision / len(masks):.4f}",
                'recall': f"{batch_recall / len(masks):.4f}",
                'f1': f"{batch_f1 / len(masks):.4f}"
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_precision = train_precision / len(train_loader)
        avg_train_recall = train_recall / len(train_loader)
        avg_train_f1 = train_f1 / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        val_precision = 0
        val_recall = 0
        val_f1 = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.float().to(device)
                masks = masks.float().to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_iou += iou_score(outputs, masks).item()
                
                # Calculate object-based metrics
                pred_masks = (outputs > 0.5).float()
                batch_precision, batch_recall, batch_f1 = 0, 0, 0
                for true_mask, pred_mask in zip(masks, pred_masks):
                    p, r, f = calculate_object_based_metrics(true_mask, pred_mask)
                    batch_precision += p
                    batch_recall += r
                    batch_f1 += f
                
                val_precision += batch_precision / len(masks)
                val_recall += batch_recall / len(masks)
                val_f1 += batch_f1 / len(masks)
                
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'iou': f"{iou_score(outputs, masks).item():.4f}",
                    'precision': f"{batch_precision / len(masks):.4f}",
                    'recall': f"{batch_recall / len(masks):.4f}",
                    'f1': f"{batch_f1 / len(masks):.4f}"
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_precision = val_precision / len(val_loader)
        avg_val_recall = val_recall / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train Precision: {avg_train_precision:.4f}, Train Recall: {avg_train_recall:.4f}, Train F1: {avg_train_f1:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Precision: {avg_val_precision:.4f}, Val Recall: {avg_val_recall:.4f}, Val F1: {avg_val_f1:.4f}')
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            counter = 0
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(best_model, save_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    model.load_state_dict(best_model)
    return model
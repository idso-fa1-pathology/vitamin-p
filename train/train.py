import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_cell_loss = 0.0
    val_nuclear_loss = 0.0
    val_cell_hv_loss = 0.0
    val_nuclei_hv_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images, cell_masks, nuclei_masks, cell_hv_maps, nuclei_hv_maps = batch

            images = images.to(device)
            cell_masks = cell_masks.to(device)
            nuclei_masks = nuclei_masks.to(device)
            cell_hv_maps = cell_hv_maps.to(device)
            nuclei_hv_maps = nuclei_hv_maps.to(device)

            cell_seg_pred, nuclear_seg_pred, cell_hv_pred, nuclei_hv_pred = model(images)

            loss, cell_loss, nuclear_loss, cell_hv_loss, nuclei_hv_loss = criterion(
                cell_seg_pred, nuclear_seg_pred, cell_hv_pred, nuclei_hv_pred,
                cell_masks, nuclei_masks, cell_hv_maps, nuclei_hv_maps
            )

            val_loss += loss.item()
            val_cell_loss += cell_loss.item()
            val_nuclear_loss += nuclear_loss.item()
            val_cell_hv_loss += cell_hv_loss.item()
            val_nuclei_hv_loss += nuclei_hv_loss.item()

    val_loss /= len(val_loader)
    val_cell_loss /= len(val_loader)
    val_nuclear_loss /= len(val_loader)
    val_cell_hv_loss /= len(val_loader)
    val_nuclei_hv_loss /= len(val_loader)

    return val_loss, val_cell_loss, val_nuclear_loss, val_cell_hv_loss, val_nuclei_hv_loss

def train_model(model, train_loader, val_loader, criterion, num_epochs=100, learning_rate=1e-5, save_interval=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    max_grad_norm = 1.0
    accumulation_steps = 4

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cell_loss = 0.0
        train_nuclear_loss = 0.0
        train_cell_hv_loss = 0.0
        train_nuclei_hv_loss = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, cell_masks, nuclei_masks, cell_hv_maps, nuclei_hv_maps = batch

            images = images.to(device)
            cell_masks = cell_masks.to(device)
            nuclei_masks = nuclei_masks.to(device)
            cell_hv_maps = cell_hv_maps.to(device)
            nuclei_hv_maps = nuclei_hv_maps.to(device)
            
            cell_seg_pred, nuclear_seg_pred, cell_hv_pred, nuclei_hv_pred = model(images)
            
            loss, cell_loss, nuclear_loss, cell_hv_loss, nuclei_hv_loss = criterion(
                cell_seg_pred, nuclear_seg_pred, cell_hv_pred, nuclei_hv_pred,
                cell_masks, nuclei_masks, cell_hv_maps, nuclei_hv_maps
            )
            
            loss = loss / accumulation_steps
            
            if torch.isnan(loss) or torch.isnan(cell_loss) or torch.isnan(nuclear_loss) or \
               torch.isnan(cell_hv_loss) or torch.isnan(nuclei_hv_loss):
                print(f"NaN detected in loss calculation at batch {batch_idx}")
                print(f"Loss: {loss.item()}, Cell Loss: {cell_loss.item()}, Nuclear Loss: {nuclear_loss.item()}, "
                      f"Cell HV Loss: {cell_hv_loss.item()}, Nuclei HV Loss: {nuclei_hv_loss.item()}")
                continue

            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 10:
                            print(f"Large gradient in {name}: {grad_norm}")
                        if torch.isnan(param.grad).any():
                            print(f"NaN detected in gradients of {name} at batch {batch_idx}")
                
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            train_cell_loss += cell_loss.item()
            train_nuclear_loss += nuclear_loss.item()
            train_cell_hv_loss += cell_hv_loss.item()
            train_nuclei_hv_loss += nuclei_hv_loss.item()
        
        train_loss /= len(train_loader)
        train_cell_loss /= len(train_loader)
        train_nuclear_loss /= len(train_loader)
        train_cell_hv_loss /= len(train_loader)
        train_nuclei_hv_loss /= len(train_loader)
        
        val_loss, val_cell_loss, val_nuclear_loss, val_cell_hv_loss, val_nuclei_hv_loss = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Cell Loss: {train_cell_loss:.4f}, Nuclear Loss: {train_nuclear_loss:.4f}, "
              f"Cell HV Loss: {train_cell_hv_loss:.4f}, Nuclei HV Loss: {train_nuclei_hv_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Cell Loss: {val_cell_loss:.4f}, Nuclear Loss: {val_nuclear_loss:.4f}, "
              f"Cell HV Loss: {val_cell_hv_loss:.4f}, Nuclei HV Loss: {val_nuclei_hv_loss:.4f}")
        
        scheduler.step(val_loss)

        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"Model saved at epoch {epoch+1}")

    return model
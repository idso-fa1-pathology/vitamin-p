import torch
import torch.optim as optim
from tqdm import tqdm
from utils.losses import MultiTaskLoss

def train_model(model, train_loader, val_loader, num_epochs=10, batch_size=16, learning_rate=1e-4):
    device = next(model.parameters()).device
    
    criterion = MultiTaskLoss(
        num_classes=5, 
        num_tissue_types=19
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loss_dict_sum = {
            'cell_seg_loss': 0, 'cell_class_loss': 0, 'tissue_class_loss': 0,
            'global_cell_loss': 0, 'hv_loss': 0, 'hv_grad_loss': 0
        }
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, binary_masks, multi_class_masks, hv_maps, tissue_types, global_cell_labels = batch
            images = images.to(device)
            binary_masks = binary_masks.to(device)
            multi_class_masks = multi_class_masks.to(device)
            hv_maps = hv_maps.to(device)
            tissue_types = tissue_types.to(device)
            global_cell_labels = global_cell_labels.to(device)

            optimizer.zero_grad()
            
            try:
                cell_seg_out, cell_class_out, tc_out, global_cell_out, hv_out = model(images)

                loss, loss_dict = criterion(cell_seg_out, cell_class_out, tc_out, global_cell_out, hv_out,
                                            binary_masks, multi_class_masks, tissue_types, global_cell_labels, hv_maps)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_dict_sum[k] += v
            
            except RuntimeError as e:
                print(f"Error in training batch: {e}")
                raise e

        avg_train_loss = total_loss / len(train_loader)
        avg_loss_dict = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_loss_dict_sum = {k: 0 for k in loss_dict_sum.keys()}
        with torch.no_grad():
            for batch in val_loader:
                images, binary_masks, multi_class_masks, hv_maps, tissue_types, global_cell_labels = batch
                images = images.to(device)
                binary_masks = binary_masks.to(device)
                multi_class_masks = multi_class_masks.to(device)
                hv_maps = hv_maps.to(device)
                tissue_types = tissue_types.to(device)
                global_cell_labels = global_cell_labels.to(device)

                cell_seg_out, cell_class_out, tc_out, global_cell_out, hv_out = model(images)

                loss, loss_dict = criterion(cell_seg_out, cell_class_out, tc_out, global_cell_out, hv_out,
                                            binary_masks, multi_class_masks, tissue_types, global_cell_labels, hv_maps)

                val_loss += loss.item()
                for k, v in loss_dict.items():
                    val_loss_dict_sum[k] += v

        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_dict = {k: v / len(val_loader) for k, v in val_loss_dict_sum.items()}

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        for k, v in avg_loss_dict.items():
            print(f"  {k}: {v:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        for k, v in avg_val_loss_dict.items():
            print(f"  {k}: {v:.4f}")

        scheduler.step(avg_val_loss)

    print("\nTraining completed!")
    torch.save(model.state_dict(), "improved_cellswin_model.pth")
    print("Model saved successfully!")

    return model
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train(model, train_loader, val_loader, criterion, num_epochs=10, learning_rate=1e-4, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loss_dict_sum = {
            'cell_seg_loss': 0, 'cell_class_loss': 0, 'tissue_class_loss': 0,
            'global_cell_loss': 0, 'hv_loss': 0, 'hv_grad_loss': 0
        }
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, binary_masks, multi_class_masks, hv_maps, tissue_types, global_cell_labels = [b.to(device) for b in batch]

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
                images, binary_masks, multi_class_masks, hv_maps, tissue_types, global_cell_labels = [b.to(device) for b in batch]

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

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

    print("\nTraining completed!")
    return model

def evaluate(model, test_loader, criterion, device='cuda'):
    model.eval()
    test_loss = 0.0
    test_loss_dict_sum = {
        'cell_seg_loss': 0, 'cell_class_loss': 0, 'tissue_class_loss': 0,
        'global_cell_loss': 0, 'hv_loss': 0, 'hv_grad_loss': 0
    }
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, binary_masks, multi_class_masks, hv_maps, tissue_types, global_cell_labels = [b.to(device) for b in batch]

            cell_seg_out, cell_class_out, tc_out, global_cell_out, hv_out = model(images)

            loss, loss_dict = criterion(cell_seg_out, cell_class_out, tc_out, global_cell_out, hv_out,
                                        binary_masks, multi_class_masks, tissue_types, global_cell_labels, hv_maps)

            test_loss += loss.item()
            for k, v in loss_dict.items():
                test_loss_dict_sum[k] += v

            # Collect predictions and targets for metrics calculation
            all_predictions.append({
                'cell_seg': cell_seg_out.cpu(),
                'cell_class': cell_class_out.cpu(),
                'tissue_class': tc_out.cpu(),
                'global_cell': global_cell_out.cpu(),
                'hv': hv_out.cpu()
            })
            all_targets.append({
                'cell_seg': binary_masks.cpu(),
                'cell_class': multi_class_masks.cpu(),
                'tissue_class': tissue_types.cpu(),
                'global_cell': global_cell_labels.cpu(),
                'hv': hv_maps.cpu()
            })

    avg_test_loss = test_loss / len(test_loader)
    avg_test_loss_dict = {k: v / len(test_loader) for k, v in test_loss_dict_sum.items()}

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    for k, v in avg_test_loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Calculate and print additional metrics here
    # For example: IoU, Dice score, accuracy, etc.

    return avg_test_loss, avg_test_loss_dict, all_predictions, all_targets

# You can add more helper functions here, such as:
# - Functions to calculate specific metrics (IoU, Dice score, etc.)
# - Functions to visualize results
# - Functions to save and load models
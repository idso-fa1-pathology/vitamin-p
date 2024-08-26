import torch
from torch.utils.data import DataLoader
import yaml
import os
from data.data_loading import load_all_folds, create_train_val_test_split
from data.dataset import CellSegmentationDataset
from models.model import CellSwin
from train.train import train_model
from utils.visualization import visualize_prediction_with_metrics

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    print("Starting main function")
    # Load configuration
    config_path = 'configs/config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    print(f"Loaded config: {config}")

    # Load data
    try:
        all_images, all_masks, all_types = load_all_folds(config['data']['base_path'])
        print(f"All images shape: {[img.shape for img in all_images]}")
        print(f"All masks shape: {[mask.shape for mask in all_masks]}")
        print(f"All types shape: {[type.shape for type in all_types]}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Choose a split (e.g., the first one)
    chosen_split = 0
    data_splits = create_train_val_test_split(all_images, all_masks, all_types)
    print(f"Number of splits: {len(data_splits)}")
    print(f"Train images shape: {data_splits[chosen_split]['train']['images'].shape}")
    print(f"Train masks shape: {data_splits[chosen_split]['train']['masks'].shape}")

    # Create datasets
    train_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['train']['images'],
        data_splits[chosen_split]['train']['masks'],
        augment=config['augmentation']['train']
    )
    val_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['val']['images'],
        data_splits[chosen_split]['val']['masks'],
        augment=config['augmentation']['val']
    )
    test_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['test']['images'],
        data_splits[chosen_split]['test']['masks'],
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

    # After creating datasets
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Sample image shape: {train_dataset[0][0].shape}")
    print(f"Sample mask shape: {train_dataset[0][1].shape}")

    # Initialize model
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = CellSwin().float().to(device)

    # Train model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        save_dir=config['paths']['save_dir']
    )

    # Visualize predictions
    visualize_prediction_with_metrics(trained_model, test_dataset, device)

if __name__ == "__main__":
    main()
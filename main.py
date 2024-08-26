import torch
from torch.utils.data import DataLoader
import yaml
from data.data_loading import load_all_folds, create_train_val_test_split
from data.dataset import CellSegmentationDataset
from models.model import CellSwin
from train import train_model
from utils.visualization import visualize_prediction_with_metrics

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Load data
    all_images, all_masks, all_types = load_all_folds(config['base_path'])
    data_splits = create_train_val_test_split(all_images, all_masks, all_types)
    
    # Choose a split (e.g., the first one)
    chosen_split = 0
    
    # Create datasets
    train_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['train']['images'],
        data_splits[chosen_split]['train']['masks'],
        augment=config['augment_train']
    )
    
    val_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['val']['images'],
        data_splits[chosen_split]['val']['masks'],
        augment=config['augment_val']
    )
    
    test_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['test']['images'],
        data_splits[chosen_split]['test']['masks'],
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Initialize model
    model = CellSwin(pretrained=config['pretrained'])
    
    # Train model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'], 
        patience=config['patience'], 
        save_dir=config['save_dir']
    )
    
    # Visualize predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_prediction_with_metrics(trained_model, test_dataset, device)

if __name__ == "__main__":
    main()
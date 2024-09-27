import os
import yaml
import torch
from models.model import ModifiedCellSwin
from utils.losses import CombinedLoss
from train.train import train_model
from data.data_loader import get_data_loaders

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    config = load_config(config_path)

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        config['data']['path'], 
        batch_size=config['data']['batch_size']
    )

    # Initialize model
    model = ModifiedCellSwin()

    # Initialize loss function
    criterion = CombinedLoss(
        bce_weight=config['loss']['bce_weight'],
        dice_weight=config['loss']['dice_weight'],
        hv_weight=config['loss']['hv_weight'],
        hv_mse_weight=config['loss']['hv_mse_weight'],
        hv_msge_weight=config['loss']['hv_msge_weight']
    )

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        num_epochs=config['training']['num_epochs'], 
        learning_rate=float(config['training']['learning_rate']),  # Convert to float
        save_interval=config['training']['save_interval']
    )

    print("Training complete.")

if __name__ == "__main__":
    main()
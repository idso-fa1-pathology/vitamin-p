import os
import yaml
import torch
from models.model import ModifiedCellSwin
from utils.losses import get_loss_function
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
    print(f"Config loaded. Data path: {config['data']['path']}")

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        base_path=config['data']['path'],
        batch_size=config['data']['batch_size']
    )
    
    if train_loader is None or val_loader is None or test_loader is None:
        print("Error: One or more data loaders are None. Check data loading process.")
        return

    # Initialize model
    model = ModifiedCellSwin(num_cell_classes=config['model']['num_cell_classes'], 
                             num_tissue_classes=config['model']['num_tissue_classes'])

    # Initialize loss function
    criterion = get_loss_function(num_classes=config['model']['num_cell_classes'], 
                                  num_tissue_types=config['model']['num_tissue_classes'])

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
        learning_rate=float(config['training']['learning_rate']),
        save_interval=config['training']['save_interval'],
        device=device
    )

    print("Training complete.")

if __name__ == "__main__":
    main()
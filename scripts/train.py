"""
Training script for Kubernetes job submission
Usage: python scripts/train.py --model flex --size base --fold 1
"""

import argparse
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import Config, create_dataloaders
from vitamin_p import (
    VitaminPFlex, 
    VitaminPDual, 
    VitaminPHEBaseline, 
    VitaminPMIFBaseline,
    VitaminPTrainer
)


def main():
    parser = argparse.ArgumentParser(description='Train Vitamin-P models')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True, 
                       choices=['flex', 'dual', 'he', 'mif'],
                       help='Model type: flex, dual, he, mif')
    
    # Model configuration
    parser.add_argument('--size', type=str, default='base',
                       choices=['small', 'base', 'large', 'giant'],
                       help='Model size')
    
    parser.add_argument('--fold', type=int, required=True,
                       help='Fold number (1, 2, or 3)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=250,
                       help='Number of epochs')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (if None, uses config default)')
    
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable augmentations')
    
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone weights')
    
    # W&B configuration
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    parser.add_argument('--wandb-project', type=str, default='vitamin-p',
                       help='W&B project name')
    
    parser.add_argument('--run-name', type=str, default=None,
                       help='W&B run name (auto-generated if not provided)')
    
    # Paths
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Config directory')
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Load config and data
    config_path = f"{args.config_dir}/config_fold{args.fold}.yaml"
    print(f"\nLoading config: {config_path}")
    config = Config(config_path)
    config.print_config()
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Initialize model
    print(f"\n{'='*80}")
    print(f"Initializing model: {args.model.upper()} - {args.size.upper()}")
    print(f"{'='*80}")
    
    if args.model == 'flex':
        model = VitaminPFlex(
            model_size=args.size,
            freeze_backbone=args.freeze_backbone
        )
    elif args.model == 'dual':
        model = VitaminPDual(
            model_size=args.size,
            dropout_rate=0.3,
            freeze_backbone=args.freeze_backbone
        )
    elif args.model == 'he':
        model = VitaminPHEBaseline(
            model_size=args.size,
            freeze_backbone=args.freeze_backbone
        )
    elif args.model == 'mif':
        model = VitaminPMIFBaseline(
            model_size=args.size,
            freeze_backbone=args.freeze_backbone
        )
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"Vitamin-P-{args.model.upper()}-{args.size}-fold{args.fold}"
    
    # Initialize trainer
    trainer = VitaminPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=1e-4,
        use_wandb=not args.no_wandb,
        project_name=args.wandb_project,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    print(f"\n{'='*80}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*80}\n")
    
    trainer.train(
        epochs=args.epochs,
        use_augmentations=not args.no_augment
    )
    
    print("\n✅ Training completed successfully!")


if __name__ == '__main__':
    main()
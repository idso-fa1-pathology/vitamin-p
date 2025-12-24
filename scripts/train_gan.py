"""
Training script for Pix2Pix GAN (H&E → Synthetic MIF)

Usage: 
    python scripts/train_gan.py --fold 1 --epochs 250
    python scripts/train_gan.py --fold 2 --epochs 250 --lambda_l1 150
"""

import argparse
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import Config, create_dataloaders
from vitaminp.gan import Pix2PixTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix GAN for H&E → MIF synthesis')
    
    # Training configuration
    parser.add_argument('--fold', type=int, required=True,
                       choices=[1, 2, 3, 4, 5, 6],
                       help='Fold number (1-5)')
    
    parser.add_argument('--epochs', type=int, default=250,
                       help='Number of epochs')
    
    parser.add_argument('--lr_g', type=float, default=2e-4,
                       help='Generator learning rate')
    
    parser.add_argument('--lr_d', type=float, default=2e-4,
                       help='Discriminator learning rate')
    
    parser.add_argument('--lambda_l1', type=float, default=100,
                       help='Weight for L1 loss (default: 100)')
    
    # W&B configuration
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    parser.add_argument('--wandb-project', type=str, default='pathology-gan-synthesis',
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
    print(f"\n{'='*80}")
    print(f"Pix2Pix GAN Training Setup")
    print(f"{'='*80}")
    print(f"Task: H&E → Synthetic MIF")
    print(f"Device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*80}\n")
    
    # Load config and data
    config_path = f"{args.config_dir}/config_fold{args.fold}.yaml"
    print(f"Loading config: {config_path}")
    
    try:
        config = Config(config_path)
        config.print_config()
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
    
    print("\nLoading dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        print(f"✅ Data loaded successfully!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"Pix2Pix-HE2MIF-fold{args.fold}-lambda{args.lambda_l1}"
    
    # Initialize trainer
    print(f"\n{'='*80}")
    print(f"Initializing Pix2Pix Trainer")
    print(f"{'='*80}")
    print(f"Generator LR: {args.lr_g}")
    print(f"Discriminator LR: {args.lr_d}")
    print(f"Lambda L1: {args.lambda_l1}")
    print(f"Epochs: {args.epochs}")
    print(f"W&B logging: {not args.no_wandb}")
    if not args.no_wandb:
        print(f"W&B project: {args.wandb_project}")
        print(f"W&B run name: {args.run_name}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*80}\n")
    
    try:
        trainer = Pix2PixTrainer(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            lambda_l1=args.lambda_l1,
            use_wandb=not args.no_wandb,
            project_name=args.wandb_project,
            run_name=args.run_name,
            checkpoint_dir=args.checkpoint_dir
        )
    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Train
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")
    
    try:
        trainer.train(epochs=args.epochs)
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
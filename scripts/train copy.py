"""
Training script for Kubernetes job submission
Usage: 
    python scripts/train.py --model dual --size base --fold 1
    python scripts/train.py --model syn --size base --fold 1
    python scripts/train.py --model flex --size large --fold 2
    python scripts/train.py --model baseline_he --size base --fold 1
    python scripts/train.py --model baseline_mif --size base --fold 1
    
    # WITH UNI PRETRAINING
    python scripts/train.py --model flex --size large --fold 2 --pretrain uni
    python scripts/train.py --model flex --size large --fold 2 --pretrain uni --freeze-backbone
"""

import argparse
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import Config, create_dataloaders
from vitaminp import VitaminPDual, VitaminPSyn, VitaminPFlex, VitaminPBaselineHE, VitaminPBaselineMIF, VitaminPTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Vitamin-P models')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['dual', 'syn', 'flex', 'baseline_he', 'baseline_mif'],
                       help='Model type: dual (Dual-Encoder), syn (Dual-Encoder with Synthetic MIF), flex (Single-Encoder), baseline_he (H&E-only), baseline_mif (MIF-only)')
    
    # Model configuration
    parser.add_argument('--size', type=str, default='base',
                       choices=['small', 'base', 'large', 'giant'],
                       help='Model size')
    
    parser.add_argument('--fold', type=int, required=True,
                       choices=[1, 2, 3, 4, 5, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                       help='Fold number (1-5)')
    
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (for dual, syn, and baseline models)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=250,
                       help='Number of epochs')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable augmentations')
    
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone weights')
    
    # Pretraining configuration
    parser.add_argument('--pretrain', type=str, default='none',
                       choices=['none', 'uni'],
                       help='Pretrained weights: none (random init) or uni (UNI histopathology pretraining)')
    
    parser.add_argument('--pretrain-path', type=str, default='./pretrain/pytorch_model.bin',
                       help='Path to pretrained UNI weights file')
    
    # W&B configuration
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    parser.add_argument('--wandb-project', type=str, default='vitamin-p',
                       help='W&B project name')
    
    parser.add_argument('--run-name', type=str, default=None,
                       help='W&B run name (auto-generated if not provided)')
    
    # Paths
    parser.add_argument('--config-dir', type=str, default='configs/training',
                       help='Config directory')
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')

    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"Vitamin-P Training Setup")
    print(f"{'='*80}")
    print(f"Model: VitaminP{args.model.replace('_', ' ').title()}")
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
    
    # Initialize model
    print(f"\n{'='*80}")
    print(f"Initializing VitaminP{args.model.replace('_', ' ').title()} Model")
    print(f"{'='*80}")
    print(f"Model size: {args.size.upper()}")
    if args.model in ['dual', 'syn', 'baseline_he', 'baseline_mif']:
        print(f"Dropout rate: {args.dropout}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Pretraining: {args.pretrain.upper()}")
    print(f"{'='*80}\n")
    
    try:
        if args.model == 'dual':
            model = VitaminPDual(
                model_size=args.size,
                dropout_rate=args.dropout,
                freeze_backbone=args.freeze_backbone
            )
        elif args.model == 'syn':
            model = VitaminPSyn(
                model_size=args.size,
                dropout_rate=args.dropout,
                freeze_backbone=args.freeze_backbone
            )
        elif args.model == 'flex':
            model = VitaminPFlex(
                model_size=args.size,
                freeze_backbone=args.freeze_backbone
            )
        elif args.model == 'baseline_he':
            model = VitaminPBaselineHE(
                model_size=args.size,
                dropout_rate=args.dropout,
                freeze_backbone=args.freeze_backbone
            )
        elif args.model == 'baseline_mif':
            model = VitaminPBaselineMIF(
                model_size=args.size,
                dropout_rate=args.dropout,
                freeze_backbone=args.freeze_backbone
            )
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*80}")
        print(f"Model Statistics (Random Init)")
        print(f"{'='*80}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== LOAD PRETRAINED WEIGHTS (FIXED WITH PARTIAL LOADING) ==========
    if args.pretrain == 'uni':
        print(f"\n{'='*80}")
        print(f"Loading UNI Pretrained Weights (Partial - Transformer Blocks)")
        print(f"{'='*80}")
        try:
            # Check if file exists
            if not os.path.exists(args.pretrain_path):
                print(f"❌ Pretrain file not found: {args.pretrain_path}")
                sys.exit(1)
            
            # Load weights
            uni_weights = torch.load(args.pretrain_path, map_location=device)
            print(f"✅ Loaded {len(uni_weights)} weight keys from UNI")
            
            # Add prefix 'dinov2.' to all keys
            uni_weights_prefixed = {f'dinov2.{k}': v for k, v in uni_weights.items()}
            
            # Remove incompatible layers (different patch sizes)
            incompatible_keys = [
                'dinov2.pos_embed',
                'dinov2.patch_embed.proj.weight',
                'dinov2.patch_embed.proj.bias'
            ]
            uni_weights_filtered = {k: v for k, v in uni_weights_prefixed.items() 
                                    if k not in incompatible_keys}
            
            print(f"✅ Filtered to {len(uni_weights_filtered)} compatible keys")
            print(f"   (Kept transformer blocks, initialized patch_embed & pos_embed)")
            
            # Load into backbone
            if hasattr(model, 'backbone'):
                result = model.backbone.load_state_dict(uni_weights_filtered, strict=False)
                print(f"✅ UNI weights loaded into backbone!")
                print(f"   Missing keys (expected): {len(result.missing_keys)}")
            elif hasattr(model, 'he_backbone'):
                # For Dual/Syn models
                model.he_backbone.load_state_dict(uni_weights_filtered, strict=False)
                model.mif_backbone.load_state_dict(uni_weights_filtered, strict=False)
                model.shared_backbone.load_state_dict(uni_weights_filtered, strict=False)
                print(f"✅ UNI weights loaded into all backbones!")
            
            file_size_gb = os.path.getsize(args.pretrain_path) / (1024**3)
            print(f"   File: {args.pretrain_path}")
            print(f"   Size: {file_size_gb:.2f} GB")
            
        except Exception as e:
            print(f"❌ Error loading UNI weights: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print(f"{'='*80}\n")
    # ========== END PRETRAIN BLOCK ==========
    # ========== END PRETRAIN BLOCK ==========
    
    # Generate run name if not provided
    if args.run_name is None:
        aug_suffix = "" if args.no_augment else "-aug"
        freeze_suffix = "-frozen" if args.freeze_backbone else ""
        pretrain_suffix = f"-{args.pretrain}" if args.pretrain != 'none' else ""
        model_name = args.model.replace('_', '-').title()
        args.run_name = f"VitaminP{model_name}-{args.size}-fold{args.fold}{aug_suffix}{freeze_suffix}{pretrain_suffix}"
    
    # Initialize trainer
    print(f"{'='*80}")
    print(f"Initializing Trainer")
    print(f"{'='*80}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Epochs: {args.epochs}")
    print(f"Augmentations: {not args.no_augment}")
    print(f"W&B logging: {not args.no_wandb}")
    if not args.no_wandb:
        print(f"W&B project: {args.wandb_project}")
        print(f"W&B run name: {args.run_name}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*80}\n")
    
    try:
        trainer = VitaminPTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            fold=args.fold,
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
        trainer.train(
            epochs=args.epochs,
            use_augmentations=not args.no_augment
        )
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
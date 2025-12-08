#!/usr/bin/env python3
"""
Vitamin-P: Multimodal Model Training Script
Trains the DualEncoderUNet (HE + MIF) model for nuclei and cell segmentation
"""

import os
import sys
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

print(f"✅ Working directory: {os.getcwd()}")

# Import required modules
from crc_dataset import Config, create_dataloaders
from models import DualEncoderUNet, train_model, set_seed

def main():
    """Main training function"""
    
    print("\n" + "="*60)
    print("VITAMIN-P: MULTIMODAL MODEL TRAINING")
    print("="*60)
    
    # Set seed for reproducibility
    print("\n🌱 Setting random seed to 42...")
    set_seed(42)
    
    # Load configuration
    print("\n📋 Loading configuration...")
    config = Config("config.yaml")
    config.print_config()
    
    # Create dataloaders
    print("\n📦 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print("✅ Dataloaders ready!")
    
    # Initialize multimodal model
    print("\n" + "="*60)
    print("INITIALIZING MULTIMODAL MODEL (HE + MIF)")
    print("="*60)
    model_multimodal = DualEncoderUNet(
        backbone='resnet101', 
        pretrained=True, 
        dropout_rate=0.3
    )
    
    # Print model statistics
    total_params = sum(p.numel() for p in model_multimodal.parameters())
    trainable_params = sum(p.numel() for p in model_multimodal.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train multimodal model
    print("\n🚀 Starting training...")
    print("="*60)
    model_multimodal = train_model(
        model_multimodal, 
        train_loader, 
        val_loader, 
        num_epochs=50, 
        lr=1e-3, 
        weight_decay=1e-4
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print("✅ Multimodal model saved: checkpoints/best_dual_encoder_model.pth")
    print("\nTraining finished successfully! 🎉")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Partial results may be saved in checkpoints/")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred during training:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
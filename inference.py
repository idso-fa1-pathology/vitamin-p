#!/usr/bin/env python3
"""
Vitamin-P: WSI Inference Script
Process entire WSI for nuclei and cell segmentation
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import tifffile
import gc

# Set up paths
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from inference.wsi_inference import WSIInference
from models import HEOnlyUNet
import argparse


def load_model(model_path, device='cuda:0'):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    
    model = HEOnlyUNet(
        backbone='resnet50', 
        pretrained=False, 
        dropout_rate=0.3
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    best_loss = checkpoint.get('best_loss', None)
    if best_loss is not None:
        print(f"Best Loss: {best_loss:.4f}")
    
    return model


def process_wsi_chunked(wsi_path, wsi_inference, wsi_name, modality, target, chunk_size=10000, overlap=512):
    """Process large WSI in chunks to avoid memory issues"""
    print(f"\nProcessing large WSI in chunks...")
    print(f"Chunk size: {chunk_size}x{chunk_size}, Overlap: {overlap}")
    
    # Get WSI dimensions
    with tifffile.TiffFile(wsi_path) as tif:
        page = tif.pages[0]
        wsi_height, wsi_width = page.shape[:2]
        print(f"WSI dimensions: {wsi_height} x {wsi_width}")
    
    # Calculate chunk grid
    stride = chunk_size - overlap
    n_rows = int(np.ceil(wsi_height / stride))
    n_cols = int(np.ceil(wsi_width / stride))
    total_chunks = n_rows * n_cols
    
    print(f"Processing grid: {n_rows} rows x {n_cols} cols = {total_chunks} chunks")
    
    all_cells = []
    
    # Process each chunk
    for row in range(n_rows):
        for col in range(n_cols):
            chunk_idx = row * n_cols + col + 1
            print(f"\nProcessing chunk {chunk_idx}/{total_chunks} (row={row}, col={col})")
            
            # Calculate chunk coordinates
            y_start = row * stride
            x_start = col * stride
            y_end = min(y_start + chunk_size, wsi_height)
            x_end = min(x_start + chunk_size, wsi_width)
            
            # Load chunk
            with tifffile.TiffFile(wsi_path) as tif:
                chunk = tif.pages[0].asarray()[y_start:y_end, x_start:x_end]
            
            # Handle channels
            if chunk.ndim == 2:
                chunk = np.stack([chunk] * 3, axis=-1)
            elif chunk.shape[-1] > 3:
                chunk = chunk[..., :3]
            
            # Normalize
            if chunk.dtype != np.uint8:
                if chunk.max() <= 1.0:
                    chunk = (chunk * 255).astype(np.uint8)
                else:
                    chunk = chunk.astype(np.uint8)
            
            # Process chunk
            results = wsi_inference.process_wsi(
                wsi_array=chunk,
                wsi_name=f"{wsi_name}_chunk_{chunk_idx}",
                modality=modality,
                target=target
            )
            
            # Adjust cell coordinates to global
            for cell in results['cells']:
                offset = np.array([y_start, x_start])
                cell['centroid'] = cell['centroid'] + offset
                cell['bbox'] = cell['bbox'] + offset  # bbox is already 2x2
                cell['contour'] = cell['contour'] + offset
            
            all_cells.extend(results['cells'])
            print(f"  Detected {results['num_cells']} cells in this chunk")
            print(f"  Total cells so far: {len(all_cells)}")
            
            # Clear memory
            del chunk, results
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"\n✅ Total cells detected across all chunks: {len(all_cells)}")
    
    return {
        'cells': all_cells,
        'num_cells': len(all_cells),
        'wsi_metadata': {
            'name': wsi_name,
            'shape': (wsi_height, wsi_width),
            'num_chunks': total_chunks,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'modality': modality,
            'target': target
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Vitamin-P WSI Inference')
    parser.add_argument('--wsi_path', required=True, help='Path to WSI file (.ome.tif)')
    parser.add_argument('--wsi_name', required=True, help='Name for output files')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', default='outputs', help='Output directory for results')
    parser.add_argument('--patch_size', type=int, default=1024, help='Patch size for processing')
    parser.add_argument('--overlap', type=int, default=64, help='Overlap between patches')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Chunk size for large WSI')
    parser.add_argument('--chunk_overlap', type=int, default=512, help='Overlap between chunks')
    parser.add_argument('--modality', default='he', choices=['he', 'mif'], help='Image modality')
    parser.add_argument('--targets', nargs='+', default=['nuclei', 'cell'], 
                       choices=['nuclei', 'cell'], help='Targets to segment')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--magnification', type=int, default=40, choices=[20, 40])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"VITAMIN-P WSI INFERENCE")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Modality: {args.modality}")
    print(f"Targets: {', '.join(args.targets)}")
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Initialize inference
    print(f"\nInitializing WSI Inference...")
    wsi_inference = WSIInference(
        model=model,
        device=args.device,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        magnification=args.magnification,
        num_classes=6,
        output_dir=output_dir
    )
    
    # Process each target
    for target in args.targets:
        print(f"\n{'='*80}")
        print(f"Processing {target.upper()}")
        print(f"{'='*80}")
        
        results = process_wsi_chunked(
            wsi_path=args.wsi_path,
            wsi_inference=wsi_inference,
            wsi_name=f"{args.wsi_name}_{target}",
            modality=args.modality,
            target=target,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap
        )
        
        print(f"✅ {target.upper()}: {results['num_cells']:,} cells detected")
    
    print(f"\n{'='*80}")
    print(f"✅ INFERENCE COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Vitamin-P: WSI Inference Script
Process entire WSI for nuclei and cell segmentation
"""
import os
import sys
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from inference.wsi_inference import WSIInference, process_entire_wsi_merged
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_path', required=True, help='Path to WSI file')
    parser.add_argument('--wsi_name', required=True, help='Name for output')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--chunk_size', type=int, default=10000)
    parser.add_argument('--chunk_overlap', type=int, default=512)
    parser.add_argument('--modality', default='he', choices=['he', 'mif'])
    parser.add_argument('--targets', nargs='+', default=['nuclei', 'cell'])
    args = parser.parse_args()
    
    # Initialize inference
    wsi_inference = WSIInference(model_path=args.model_path)
    
    # Process each target
    for target in args.targets:
        print(f"\n{'='*80}")
        print(f"Processing {target.upper()}")
        print(f"{'='*80}")
        
        results = process_entire_wsi_merged(
            wsi_path=args.wsi_path,
            wsi_inference=wsi_inference,
            wsi_name=args.wsi_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            modality=args.modality,
            target=target,
            save_intermediate=False
        )
        
        print(f"✅ {target}: {results['num_cells']:,} detected")

if __name__ == "__main__":
    main()
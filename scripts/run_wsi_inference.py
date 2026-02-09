#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Command-line interface for VitaminP WSI Inference
# Allows running inference from the terminal

import argparse
import sys
from pathlib import Path
import torch
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vitaminp import VitaminPFlex, VitaminPDual
from vitaminp.inference import WSIPredictor, ChannelConfig
from vitaminp.inference.utils import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VitaminP inference on Whole Slide Images (WSI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['flex', 'dual'],
        help='Model type: flex (VitaminPFlex) or dual (VitaminPDual)'
    )
    required.add_argument(
        '--model_size',
        type=str,
        required=True,
        choices=['small', 'base', 'large', 'xlarge'],
        help='Model size'
    )
    required.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    required.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    # WSI input (mutually exclusive: single WSI or multiple WSIs)
    wsi_group = parser.add_mutually_exclusive_group(required=True)
    wsi_group.add_argument(
        '--wsi_path',
        type=str,
        help='Path to a single WSI file'
    )
    wsi_group.add_argument(
        '--wsi_folder',
        type=str,
        help='Path to folder containing multiple WSI files'
    )
    wsi_group.add_argument(
        '--wsi_list',
        type=str,
        help='Path to text file with list of WSI paths (one per line)'
    )
    
    # Model parameters
    model_group = parser.add_argument_group('model parameters')
    model_group.add_argument(
        '--freeze_backbone',
        action='store_true',
        help='Whether backbone was frozen during training'
    )
    model_group.add_argument(
        '--branches',
        type=str,
        nargs='+',
        default=['he_nuclei'],
        choices=['he_nuclei', 'he_cell', 'mif_nuclei', 'mif_cell'],
        help='Branch(es) to run. Pass multiple to activate constrained watershed, '
             'e.g.: --branches he_nuclei he_cell'
    )
    
    # Inference parameters
    inference_group = parser.add_argument_group('inference parameters')
    inference_group.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    inference_group.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (auto-estimated if not provided)'
    )
    inference_group.add_argument(
        '--patch_size',
        type=int,
        default=512,
        help='Size of image patches/tiles'
    )
    inference_group.add_argument(
        '--overlap',
        type=int,
        default=64,
        help='Overlap between patches (pixels)'
    )
    inference_group.add_argument(
        '--target_mpp',
        type=float,
        default=0.25,
        help='Target microns per pixel resolution'
    )
    inference_group.add_argument(
        '--magnification',
        type=int,
        default=40,
        choices=[20, 40],
        help='Magnification level for HV postprocessing'
    )
    inference_group.add_argument(
        '--mixed_precision',
        action='store_true',
        help='Use mixed precision (FP16) for faster inference'
    )
    inference_group.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )
    
    # WSI processing parameters
    wsi_proc_group = parser.add_argument_group('WSI processing')
    wsi_proc_group.add_argument(
        '--filter_tissue',
        action='store_true',
        help='Filter tiles by tissue content'
    )
    wsi_proc_group.add_argument(
        '--tissue_threshold',
        type=float,
        default=0.1,
        help='Minimum tissue percentage to keep a tile'
    )
    wsi_proc_group.add_argument(
        '--wsi_extension',
        type=str,
        default='svs',
        help='WSI file extension when using --wsi_folder'
    )
    wsi_proc_group.add_argument(
        '--wsi_properties',
        type=str,
        default=None,
        help='WSI properties as JSON string: {"slide_mpp": 0.25, "magnification": 20}'
    )
    
    # MIF channel configuration
    mif_group = parser.add_argument_group('MIF channel configuration')
    mif_group.add_argument(
        '--mif_nuclear_channel',
        type=int,
        default=None,
        help='Nuclear channel index for MIF images (e.g., 0 for SYTO13)'
    )
    mif_group.add_argument(
        '--mif_membrane_channels',
        type=str,
        default=None,
        help='Membrane channel indices (comma-separated, e.g., "1,2")'
    )
    mif_group.add_argument(
        '--mif_membrane_combination',
        type=str,
        default='max',
        choices=['max', 'mean', 'sum'],
        help='How to combine membrane channels'
    )
    
    # Post-processing parameters
    postproc_group = parser.add_argument_group('post-processing')
    postproc_group.add_argument(
        '--clean_overlaps',
        action='store_true',
        default=True,
        help='Enable overlap cleaning'
    )
    postproc_group.add_argument(
        '--iou_threshold',
        type=float,
        default=0.5,
        help='IoU threshold for overlap cleaning'
    )
    postproc_group.add_argument(
        '--detection_threshold',
        type=float,
        default=0.5,
        help='Binary threshold for instance extraction (0.5-0.8)'
    )
    postproc_group.add_argument(
        '--min_area_um',
        type=float,
        default=5.0,
        help='Minimum cell area in μm² (filters small artifacts)'
    )
    postproc_group.add_argument(
        '--simplify_epsilon',
        type=float,
        default=1.0,
        help='Contour simplification factor (higher = fewer points). Use 2.0-3.0 for large files'
    )
    postproc_group.add_argument(
        '--coord_precision',
        type=int,
        default=1,
        help='Decimal places for coordinates (0 = integer, 1 = 0.1px precision, 2 = 0.01px)'
    )
    postproc_group.add_argument(
        '--use_constrained_watershed',
        action='store_true',
        default=True,
        help='Use nuclei-constrained watershed for cell branches. '
             'Requires matching nuclei branch in --branches (e.g. he_nuclei + he_cell). '
             'Falls back to standard HoVer-Net if nuclei branch is missing.'
    )
    postproc_group.add_argument(
        '--no_constrained_watershed',
        action='store_true',
        default=False,
        help='Disable constrained watershed (use standard HoVer-Net for cell branches)'
    )
    postproc_group.add_argument(
        '--cell_threshold',
        type=float,
        default=0.5,
        help='Probability threshold for cell seg map in constrained watershed'
    )
    
    # Output parameters
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--save_geojson',
        action='store_true',
        default=True,
        help='Save detections as GeoJSON'
    )
    output_group.add_argument(
        '--save_parquet',
        action='store_true',
        default=False,
        help='Save detections as Parquet (efficient binary format, requires geopandas)'
    )
    output_group.add_argument(
        '--save_visualization',
        action='store_true',
        default=True,
        help='Save visualization with contours'
    )
    output_group.add_argument(
        '--save_heatmap',
        action='store_true',
        help='Save prediction heatmap'
    )
    output_group.add_argument(
        '--save_csv',
        action='store_true',
        help='Save detections as CSV (in addition to JSON)'
    )
    
    return parser.parse_args()


def load_model(args):
    """Load VitaminP model based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        torch.nn.Module: Loaded model
    """
    print(f"Initializing {args.model_type.upper()} model (size: {args.model_size})...")
    
    if args.model_type == 'flex':
        model = VitaminPFlex(
            model_size=args.model_size,
            freeze_backbone=args.freeze_backbone,
        )
    elif args.model_type == 'dual':
        model = VitaminPDual(
            model_size=args.model_size,
            freeze_backbone=args.freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print("Model initialized successfully")
    return model


def get_wsi_paths(args):
    """Get list of WSI paths based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        List[Path]: List of WSI paths to process
    """
    wsi_paths = []
    
    if args.wsi_path:
        # Single WSI
        wsi_paths = [Path(args.wsi_path)]
        
    elif args.wsi_folder:
        # Folder of WSIs
        folder = Path(args.wsi_folder)
        if not folder.exists():
            raise FileNotFoundError(f"WSI folder not found: {folder}")
        
        # Find all files with specified extension
        wsi_paths = list(folder.glob(f"*.{args.wsi_extension}"))
        
        if len(wsi_paths) == 0:
            raise ValueError(
                f"No WSI files with extension '.{args.wsi_extension}' found in {folder}"
            )
        
        print(f"Found {len(wsi_paths)} WSI files in {folder}")
        
    elif args.wsi_list:
        # List file
        list_file = Path(args.wsi_list)
        if not list_file.exists():
            raise FileNotFoundError(f"WSI list file not found: {list_file}")
        
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    wsi_paths.append(Path(line))
        
        print(f"Loaded {len(wsi_paths)} WSI paths from {list_file}")
    
    # Validate paths
    for wsi_path in wsi_paths:
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")
    
    return wsi_paths


def resolve_constrained_watershed_flag(args):
    """Resolve the constrained watershed flag from the two boolean args.
    
    --use_constrained_watershed is True by default.
    --no_constrained_watershed explicitly disables it.
    The --no flag takes priority if both are somehow set.
    
    Returns:
        bool
    """
    if args.no_constrained_watershed:
        return False
    return args.use_constrained_watershed


def build_predict_kwargs(args, wsi_path, output_dir, wsi_properties):
    """Build the keyword arguments dict for predictor.predict().
    
    Centralises the single-vs-multi branch logic so both the single-WSI
    and batch paths call predict() identically.
    
    Args:
        args: Parsed CLI args
        wsi_path: Path to the WSI being processed
        output_dir: Output directory for this WSI
        wsi_properties: Parsed WSI properties dict (or None)
    
    Returns:
        dict: kwargs ready to unpack into predictor.predict()
    """
    kwargs = dict(
        wsi_path=str(wsi_path),
        output_dir=str(output_dir),
        wsi_properties=wsi_properties,
        filter_tissue=args.filter_tissue,
        tissue_threshold=args.tissue_threshold,
        clean_overlaps=args.clean_overlaps,
        iou_threshold=args.iou_threshold,
        save_heatmap=args.save_heatmap,
        save_json=True,
        save_geojson=args.save_geojson,
        save_visualization=args.save_visualization,
        save_csv=args.save_csv,
        detection_threshold=args.detection_threshold,
        min_area_um=args.min_area_um,
        simplify_epsilon=args.simplify_epsilon,
        coord_precision=args.coord_precision,
        save_parquet=args.save_parquet,
    )

    # Single branch → use `branch` (str), multi → use `branches` (list)
    if len(args.branches) == 1:
        kwargs['branch'] = args.branches[0]
    else:
        kwargs['branches'] = args.branches

    return kwargs


def log_results(logger, results, branches):
    """Log prediction results, handling both single and multi-branch return shapes.
    
    Args:
        logger: Logger instance
        results: Return value from predictor.predict()
        branches: The branches list that was passed to predict
    """
    if len(branches) == 1:
        # Flat dict
        logger.info(f"  Branch: {results['branch']}")
        logger.info(f"  Detections: {results['num_detections']}")
        if 'processing_time' in results:
            logger.info(f"  Processing time: {results['processing_time']:.2f}s")
    else:
        # Nested dict keyed by branch name
        total = 0
        for branch_name, branch_results in results.items():
            n = branch_results['num_detections']
            total += n
            logger.info(f"  {branch_name}: {n} detections")
        logger.info(f"  Total detections: {total}")


def main():
    """Main function."""
    args = parse_args()
    
    # Resolve constrained watershed flag
    use_constrained_watershed = resolve_constrained_watershed_flag(args)
    
    # Setup logger
    logger = setup_logger(
        name="vitaminp.inference.cli",
        log_file=Path(args.output_dir) / "inference.log"
    )
    
    logger.info("=" * 60)
    logger.info("VitaminP WSI Inference")
    logger.info("=" * 60)
    logger.info(f"Branches: {args.branches}")
    logger.info(f"Constrained watershed: {use_constrained_watershed}")
    if use_constrained_watershed and len(args.branches) > 1:
        logger.info(f"Cell threshold: {args.cell_threshold}")
    
    # Parse optional JSON arguments
    wsi_properties = None
    if args.wsi_properties:
        try:
            wsi_properties = json.loads(args.wsi_properties)
            logger.info(f"WSI properties: {wsi_properties}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid WSI properties JSON: {e}")
            sys.exit(1)
    
    # Create MIF channel config if specified
    mif_channel_config = None
    if args.mif_nuclear_channel is not None:
        membrane_channels = None
        if args.mif_membrane_channels:
            membrane_channels = [int(ch) for ch in args.mif_membrane_channels.split(',')]
        
        mif_channel_config = ChannelConfig(
            nuclear_channel=args.mif_nuclear_channel,
            membrane_channel=membrane_channels,
            membrane_combination=args.mif_membrane_combination
        )
        logger.info(f"MIF channel config: {mif_channel_config.get_description()}")
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load model
    try:
        model = load_model(args)
        
        # Load checkpoint and move to device
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint)
        model = model.to(args.device)
        model.eval()
        logger.info(f"✓ Model loaded and moved to {args.device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Get WSI paths
    try:
        wsi_paths = get_wsi_paths(args)
    except Exception as e:
        logger.error(f"Failed to get WSI paths: {e}")
        sys.exit(1)
    
    # Create predictor
    try:
        predictor = WSIPredictor(
            model=model,
            checkpoint_path=args.checkpoint,
            device=args.device,
            patch_size=args.patch_size,
            overlap=args.overlap,
            target_mpp=args.target_mpp,
            magnification=args.magnification,
            mixed_precision=args.mixed_precision,
            logger=logger,
            mif_channel_config=mif_channel_config,
            tissue_dilation=1,
            use_constrained_watershed=use_constrained_watershed,
            cell_threshold=args.cell_threshold,
        )
    except Exception as e:
        logger.error(f"Failed to create predictor: {e}")
        sys.exit(1)
    
    # Run inference
    try:
        if len(wsi_paths) == 1:
            # Single WSI
            predict_kwargs = build_predict_kwargs(args, wsi_paths[0], args.output_dir, wsi_properties)
            results = predictor.predict(**predict_kwargs)
            
            logger.info("\n" + "=" * 60)
            logger.info("INFERENCE COMPLETE")
            logger.info("=" * 60)
            log_results(logger, results, args.branches)
            logger.info(f"Results saved to: {args.output_dir}")
            
        else:
            # Multiple WSIs - loop through them
            all_results = {}
            
            for wsi_path in wsi_paths:
                wsi_name = wsi_path.stem
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing: {wsi_name}")
                logger.info(f"{'=' * 60}")
                
                try:
                    wsi_output_dir = Path(args.output_dir) / wsi_name
                    predict_kwargs = build_predict_kwargs(args, wsi_path, wsi_output_dir, wsi_properties)
                    result = predictor.predict(**predict_kwargs)
                    all_results[wsi_name] = result
                    
                    logger.info(f"✓ {wsi_name}:")
                    log_results(logger, result, args.branches)
                    
                except Exception as e:
                    logger.error(f"Failed to process {wsi_name}: {e}", exc_info=True)
                    all_results[wsi_name] = {'error': str(e)}
            
            # Batch summary
            logger.info("\n" + "=" * 60)
            logger.info("BATCH INFERENCE COMPLETE")
            logger.info("=" * 60)
            
            total_detections = 0
            total_time = 0.0
            successful = 0
            
            for wsi_name, result in all_results.items():
                if 'error' in result:
                    logger.error(f"✗ {wsi_name}: {result['error']}")
                    continue
                
                successful += 1
                
                # Accumulate totals — handle single vs multi branch
                if len(args.branches) == 1:
                    total_detections += result['num_detections']
                    total_time += result.get('processing_time', 0.0)
                else:
                    for branch_name, branch_results in result.items():
                        total_detections += branch_results['num_detections']
                    # processing_time not returned per-branch in multi mode;
                    # use 0.0 as safe default
            
            logger.info(f"Processed {successful}/{len(wsi_paths)} WSIs successfully")
            logger.info(f"Branches: {args.branches}")
            logger.info(f"Total detections: {total_detections}")
            if total_time > 0:
                logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
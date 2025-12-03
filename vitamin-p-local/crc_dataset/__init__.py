"""
CRC Dataset Package
Multi-modal pathology dataset for colorectal cancer with instance segmentation
"""

__version__ = "1.0.0"

# Import main components
try:
    from .config import Config
    from .dataset import (
        CRCZarrDataset,
        CRCCachedDataset,
        create_dataloaders,
        collate_fn
    )
    from .hv_generator import (
        generate_hv_map_pannuke,
        generate_hv_map_simple,
        batch_generate_hv_maps,
        visualize_hv_map
    )
    from .augmentation import MedicalImageAugmentation
    from .utils import (
        check_memory_available,
        print_memory_usage,
        validate_zarr_structure,
        print_batch_statistics,
        visualize_batch,
        count_instances_in_batch,
        test_dataloader_speed,
        get_dataset_statistics,
        print_statistics
    )
except ImportError as e:
    print(f"❌ Error importing CRC dataset components: {e}")
    import traceback
    traceback.print_exc()
    raise

# Define public API
__all__ = [
    # Config
    'Config',
    
    # Dataset classes
    'CRCZarrDataset',
    'CRCCachedDataset',
    'create_dataloaders',
    'collate_fn',
    
    # HV map generation
    'generate_hv_map_pannuke',
    'generate_hv_map_simple',
    'batch_generate_hv_maps',
    'visualize_hv_map',
    
    # Augmentation
    'MedicalImageAugmentation',
    
    # Utilities
    'check_memory_available',
    'print_memory_usage',
    'validate_zarr_structure',
    'print_batch_statistics',
    'visualize_batch',
    'count_instances_in_batch',
    'test_dataloader_speed',
    'get_dataset_statistics',
    'print_statistics',
]

print(f"✅ CRC Dataset Package v{__version__} loaded")
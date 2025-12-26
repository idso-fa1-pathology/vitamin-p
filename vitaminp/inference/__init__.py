from .wsi_handler import MultiFormatImageLoader
from .tile_processor import TileProcessor
from .utils import ResultExporter, setup_logger
from .overlap_cleaner import OverlapCleaner
from .postprocessing import process_model_outputs
from .predictor import WSIPredictor
from .channel_config import ChannelConfig, COMMON_CONFIGS, get_config_from_name  # ← NEW

__all__ = [
    'WSIPredictor',
    'MultiFormatImageLoader',
    'TileProcessor',
    'ResultExporter',
    'OverlapCleaner',
    'setup_logger',
    'process_model_outputs',
    'ChannelConfig',           # ← NEW
    'COMMON_CONFIGS',          # ← NEW
    'get_config_from_name',    # ← NEW
]
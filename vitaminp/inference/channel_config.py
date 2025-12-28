#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Channel Configuration for MIF Images
Professional channel specification system
"""

from typing import List, Optional, Dict, Union  # ← ADD Union HERE
from dataclasses import dataclass
import numpy as np


@dataclass
class ChannelConfig:
    """Configuration for MIF channel selection
    
    Examples:
        # Use channels 0 and 1 directly
        config = ChannelConfig(
            nuclear_channel=0,
            membrane_channel=1
        )
        
        # Combine multiple channels for membrane
        config = ChannelConfig(
            nuclear_channel=0,
            membrane_channel=[1, 2, 3],
            membrane_combination='max'
        )
    """
    nuclear_channel: int
    membrane_channel: Optional[Union[int, List[int]]] = None  # ← CHANGE THIS LINE
    membrane_combination: str = 'max'  # 'max', 'sum', 'mean'
    channel_names: Optional[Dict[int, str]] = None
    
    def __post_init__(self):
        """Validate configuration"""
        valid_combinations = ['max', 'sum', 'mean']
        if self.membrane_combination not in valid_combinations:
            raise ValueError(
                f"membrane_combination must be one of {valid_combinations}, "
                f"got '{self.membrane_combination}'"
            )
    
    def select_channels(self, mif_array: np.ndarray) -> np.ndarray:
        """Select and combine channels from MIF array
        
        Args:
            mif_array: Input array of shape (C, H, W) or (H, W, C)
            
        Returns:
            Array of shape (2, H, W) with [nuclear, membrane] channels
        """
        # Handle different input formats
        if mif_array.ndim == 3:
            if mif_array.shape[0] < 20:  # (C, H, W)
                channels_first = True
            else:  # (H, W, C)
                channels_first = False
                mif_array = np.transpose(mif_array, (2, 0, 1))
        else:
            raise ValueError(f"Expected 3D array, got shape {mif_array.shape}")
        
        n_channels = mif_array.shape[0]
        
        # Validate nuclear channel
        if self.nuclear_channel >= n_channels:
            raise ValueError(
                f"nuclear_channel={self.nuclear_channel} but image has only "
                f"{n_channels} channels"
            )
        
        # Extract nuclear channel
        nuclear = mif_array[self.nuclear_channel]
        
        # Extract membrane channel(s)
        if self.membrane_channel is None:
            # Use first non-nuclear channel
            membrane_idx = 1 if self.nuclear_channel != 1 else 0
            membrane = mif_array[membrane_idx]
        elif isinstance(self.membrane_channel, int):
            # Single membrane channel
            if self.membrane_channel >= n_channels:
                raise ValueError(
                    f"membrane_channel={self.membrane_channel} but image has only "
                    f"{n_channels} channels"
                )
            membrane = mif_array[self.membrane_channel]
        else:
            # Multiple membrane channels - combine them
            membrane_channels = []
            for ch_idx in self.membrane_channel:
                if ch_idx >= n_channels:
                    raise ValueError(
                        f"membrane_channel contains {ch_idx} but image has only "
                        f"{n_channels} channels"
                    )
                membrane_channels.append(mif_array[ch_idx])
            
            # Combine using specified method
            membrane_stack = np.stack(membrane_channels, axis=0)
            if self.membrane_combination == 'max':
                membrane = np.max(membrane_stack, axis=0)
            elif self.membrane_combination == 'sum':
                membrane = np.sum(membrane_stack, axis=0)
            elif self.membrane_combination == 'mean':
                membrane = np.mean(membrane_stack, axis=0)
        
        # Stack nuclear and membrane
        output = np.stack([nuclear, membrane], axis=0)  # (2, H, W)
        
        return output
    
    def get_description(self) -> str:
        """Get human-readable description of channel configuration"""
        nuclear_name = (
            self.channel_names.get(self.nuclear_channel, f"Channel {self.nuclear_channel}")
            if self.channel_names else f"Channel {self.nuclear_channel}"
        )
        
        if isinstance(self.membrane_channel, int):
            membrane_name = (
                self.channel_names.get(self.membrane_channel, f"Channel {self.membrane_channel}")
                if self.channel_names else f"Channel {self.membrane_channel}"
            )
        elif isinstance(self.membrane_channel, list):
            if self.channel_names:
                names = [self.channel_names.get(ch, f"Ch{ch}") for ch in self.membrane_channel]
            else:
                names = [f"Ch{ch}" for ch in self.membrane_channel]
            membrane_name = f"{self.membrane_combination}({', '.join(names)})"
        else:
            membrane_name = "Auto-selected"
        
        return f"Nuclear: {nuclear_name}, Membrane: {membrane_name}"


# Predefined configurations for common MIF panels
COMMON_CONFIGS = {
    'dapi_pan_cytokeratin': ChannelConfig(
        nuclear_channel=0,  # DAPI
        membrane_channel=1,  # Pan-cytokeratin
        channel_names={0: 'DAPI', 1: 'Pan-CK'}
    ),
    
    'syto13_alexa': ChannelConfig(
        nuclear_channel=0,  # SYTO 13
        membrane_channel=1,  # Alexa 532
        channel_names={0: 'SYTO 13', 1: 'Alexa 532'}
    ),
    
    'dapi_all_markers': ChannelConfig(
        nuclear_channel=0,  # DAPI
        membrane_channel=[1, 2, 3],  # Combine all protein markers
        membrane_combination='max',
        channel_names={0: 'DAPI', 1: 'Marker1', 2: 'Marker2', 3: 'Marker3'}
    ),
    
    'custom_2channel': ChannelConfig(
        nuclear_channel=0,
        membrane_channel=1
    ),
}


def get_config_from_name(config_name: str) -> ChannelConfig:
    """Get predefined configuration by name
    
    Args:
        config_name: Name of predefined config
        
    Returns:
        ChannelConfig instance
        
    Raises:
        ValueError: If config name not found
    """
    if config_name not in COMMON_CONFIGS:
        available = ', '.join(COMMON_CONFIGS.keys())
        raise ValueError(
            f"Unknown config '{config_name}'. Available configs: {available}"
        )
    
    return COMMON_CONFIGS[config_name]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights
import yaml
import os

def load_config(config_path):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the parent directory
    parent_dir = os.path.dirname(current_dir)
    # Construct the path to the config file
    config_file_path = os.path.join(parent_dir, 'configs', config_path)
    
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config('config.yaml')

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=config['seblock']['reduction']):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, reduction=config['seblock']['reduction'])
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.se(x)
        return x + residual

class SwinEncoder(nn.Module):
    def __init__(self, pretrained=config['model']['encoder']['pretrained']):
        super().__init__()
        if pretrained:
            weights = Swin_T_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.swin = swin_t(weights=weights)
        self.swin.head = nn.Identity()  # Remove the classifier head

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.swin.features):
            x = layer(x)
            if i in [2, 4, 6]:  # Collect features from specific layers
                features.append(x)
        return features

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        results = []
        last_feature = self.lateral_convs[-1](features[-1])
        results.append(self.smooth_convs[-1](last_feature))

        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            top_down = F.interpolate(last_feature, scale_factor=2, mode='nearest')
            last_feature = lateral + top_down
            results.insert(0, self.smooth_convs[i](last_feature))

        return results

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        decoder_config = config['model']['decoder']
        self.fpn = FPN(decoder_config['fpn']['in_channels'], decoder_config['fpn']['out_channels'])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch) for in_ch, out_ch in decoder_config['decoder_blocks']
        ])
        self.final_conv = nn.Conv2d(decoder_config['decoder_blocks'][-1][1], decoder_config['final_out_channels'], kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        fpn_features = self.fpn([f.permute(0, 3, 1, 2) for f in features])
        
        x = fpn_features[-1]
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        return self.sigmoid(x)

class CellSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SwinEncoder()
        self.decoder = UNetDecoder()

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output
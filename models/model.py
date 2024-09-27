import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
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
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.se(x)
        return x

class SwinEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = Swin_T_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.swin = swin_t(weights=weights)
        self.swin.head = nn.Identity()
        
        self.swin.features[0][0] = nn.Conv2d(2, 96, kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.swin.features):
            x = layer(x)
            if i in [2, 4, 6]:
                features.append(x)
        return features

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class MultiHeadDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder1 = DecoderBlock(768, 384)
        self.decoder2 = DecoderBlock(384 + 384, 192)
        self.decoder3 = DecoderBlock(192 + 192, 96)
        self.decoder4 = DecoderBlock(96 + 192, 48)
        
        self.cell_seg_conv = nn.Conv2d(48, 1, kernel_size=1)
        self.nuclear_seg_conv = nn.Conv2d(48, 1, kernel_size=1)
        self.cell_hv_conv = nn.Conv2d(48, 2, kernel_size=1)
        self.nuclei_hv_conv = nn.Conv2d(48, 2, kernel_size=1)
        
        self.attention1 = AttentionBlock(F_g=384, F_l=384, F_int=192)
        self.attention2 = AttentionBlock(F_g=192, F_l=192, F_int=96)
        self.attention3 = AttentionBlock(F_g=96, F_l=192, F_int=48)

    def forward(self, features):
        x = features[-1].permute(0, 3, 1, 2)
        x = self.decoder1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        skip1 = features[-2].permute(0, 3, 1, 2)
        skip1 = F.interpolate(skip1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, self.attention1(x, skip1)], dim=1)
        x = self.decoder2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        skip2 = features[-3].permute(0, 3, 1, 2)
        skip2 = F.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, self.attention2(x, skip2)], dim=1)
        x = self.decoder3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        skip3 = features[0].permute(0, 3, 1, 2)
        skip3 = F.interpolate(skip3, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, self.attention3(x, skip3)], dim=1)
        x = self.decoder4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Cell segmentation
        cell_seg_out = self.cell_seg_conv(x)
        cell_seg_out = F.interpolate(cell_seg_out, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Nuclear segmentation
        nuclear_seg_out = self.nuclear_seg_conv(x)
        nuclear_seg_out = F.interpolate(nuclear_seg_out, scale_factor=2, mode='bilinear', align_corners=False)
        
        # HV map for cells
        cell_hv_out = self.cell_hv_conv(x)
        cell_hv_out = F.interpolate(cell_hv_out, scale_factor=2, mode='bilinear', align_corners=False)
        
        # HV map for nuclei
        nuclei_hv_out = self.nuclei_hv_conv(x)
        nuclei_hv_out = F.interpolate(nuclei_hv_out, scale_factor=2, mode='bilinear', align_corners=False)
        
        return cell_seg_out, nuclear_seg_out, cell_hv_out, nuclei_hv_out

class ModifiedCellSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SwinEncoder()
        self.decoder = MultiHeadDecoder()

    def forward(self, x):
        features = self.encoder(x)
        cell_seg_out, nuclear_seg_out, cell_hv_out, nuclei_hv_out = self.decoder(features)
        
        return cell_seg_out, nuclear_seg_out, cell_hv_out, nuclei_hv_out

    def _debug_check_nan_inf(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            print(f"Inf detected in {name}")
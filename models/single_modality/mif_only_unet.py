import torch
import torch.nn as nn
import torchvision.models as models


class MIFOnlyUNet(nn.Module):
    """
    Single-encoder UNet for MIF pathology images only
    
    Architecture:
    - MIF (2ch): ResNet encoder layers 1-4 (full encoder)
    - Output: 2 segmentation masks + 2 HV maps (nuclei + cell)
    
    This is an ablation model to compare against the dual-encoder multimodal model.
    """
    def __init__(self, backbone='resnet50', pretrained=True, dropout_rate=0.3):
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.channels = channels
        self.dropout_rate = dropout_rate
        
        # ========== MIF ENCODER (full layers 1-4) ==========
        # Modify first conv for 2 channels
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Initialize with pretrained weights (average of RGB channels)
            with torch.no_grad():
                self.conv1.weight[:, :, :, :] = resnet.conv1.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Dropout
        self.dropout_high = nn.Dropout2d(dropout_rate * 1.5)
        self.dropout_mid = nn.Dropout2d(dropout_rate)
        
        # ========== DECODERS ==========
        self._build_decoders()
        
    def _build_decoders(self):
        """Build 2 decoder branches for nuclei and cell segmentation"""
        channels = self.channels
        dropout_rate = self.dropout_rate
        
        # NUCLEI DECODER
        self.up4_nuclei = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4_nuclei = self.conv_block(channels[4], channels[3], dropout_rate * 1.2)
        self.up3_nuclei = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3_nuclei = self.conv_block(channels[3], channels[2], dropout_rate)
        self.up2_nuclei = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2_nuclei = self.conv_block(channels[2], channels[1], dropout_rate * 0.8)
        self.up1_nuclei = nn.ConvTranspose2d(channels[1], 64, 2, stride=2)
        self.dec1_nuclei = self.conv_block(64 + 64, 64, dropout_rate * 0.5)
        self.up0_nuclei = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0_nuclei = self.conv_block(32, 32, dropout_rate * 0.3)
        self.final_nuclei_seg = nn.Conv2d(32, 1, 1)
        self.final_nuclei_hv = nn.Conv2d(32, 2, 1)
        
        # CELL DECODER
        self.up4_cell = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4_cell = self.conv_block(channels[4], channels[3], dropout_rate * 1.2)
        self.up3_cell = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3_cell = self.conv_block(channels[3], channels[2], dropout_rate)
        self.up2_cell = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2_cell = self.conv_block(channels[2], channels[1], dropout_rate * 0.8)
        self.up1_cell = nn.ConvTranspose2d(channels[1], 64, 2, stride=2)
        self.dec1_cell = self.conv_block(64 + 64, 64, dropout_rate * 0.5)
        self.up0_cell = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0_cell = self.conv_block(32, 32, dropout_rate * 0.3)
        self.final_cell_seg = nn.Conv2d(32, 1, 1)
        self.final_cell_hv = nn.Conv2d(32, 2, 1)
    
    def conv_block(self, in_ch, out_ch, dropout_rate=0.0):
        """Conv block with optional dropout"""
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
            
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ])
        
        return nn.Sequential(*layers)
    
    def decode_branch(self, x4, x3, x2, x1, x0, branch_name):
        """Generic decoder branch with skip connections"""
        up4 = getattr(self, f'up4_{branch_name}')
        dec4 = getattr(self, f'dec4_{branch_name}')
        up3 = getattr(self, f'up3_{branch_name}')
        dec3 = getattr(self, f'dec3_{branch_name}')
        up2 = getattr(self, f'up2_{branch_name}')
        dec2 = getattr(self, f'dec2_{branch_name}')
        up1 = getattr(self, f'up1_{branch_name}')
        dec1 = getattr(self, f'dec1_{branch_name}')
        up0 = getattr(self, f'up0_{branch_name}')
        dec0 = getattr(self, f'dec0_{branch_name}')
        final_seg = getattr(self, f'final_{branch_name}_seg')
        final_hv = getattr(self, f'final_{branch_name}_hv')
        
        # Decoder with skip connections
        d4 = dec4(torch.cat([up4(x4), x3], dim=1))
        d3 = dec3(torch.cat([up3(d4), x2], dim=1))
        d2 = dec2(torch.cat([up2(d3), x1], dim=1))
        d1 = dec1(torch.cat([up1(d2), x0], dim=1))
        d0 = dec0(up0(d1))
        
        seg_out = torch.sigmoid(final_seg(d0))
        hv_out = torch.tanh(final_hv(d0))
        
        return seg_out, hv_out
    
    def forward(self, mif_img):
        """
        Args:
            mif_img: (B, 2, H, W) - MIF image
        
        Returns:
            Dictionary with nuclei and cell segmentation outputs
        """
        # ========== ENCODER ==========
        x0 = self.relu(self.bn1(self.conv1(mif_img)))
        x0_pool = self.maxpool(x0)
        x1 = self.layer1(x0_pool)
        x2 = self.layer2(x1)
        
        x3 = self.layer3(x2)
        if self.training:
            x3 = self.dropout_mid(x3)
        
        x4 = self.layer4(x3)
        if self.training:
            x4 = self.dropout_high(x4)
        
        # ========== DECODERS ==========
        nuclei_seg, nuclei_hv = self.decode_branch(x4, x3, x2, x1, x0, 'nuclei')
        cell_seg, cell_hv = self.decode_branch(x4, x3, x2, x1, x0, 'cell')
        
        return {
            'mif_nuclei_seg': nuclei_seg,
            'mif_nuclei_hv': nuclei_hv,
            'mif_cell_seg': cell_seg,
            'mif_cell_hv': cell_hv,
        }
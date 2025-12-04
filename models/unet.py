import torch
import torch.nn as nn
import torchvision.models as models


class DualEncoderUNet(nn.Module):
    """
    Dual-encoder UNet for H&E + MIF pathology images with mid-fusion
    
    Architecture:
    - H&E (3ch): ResNet encoder layers 1-2 (separate)
    - MIF (2ch): ResNet encoder layers 1-2 (separate)
    - Fusion: Concatenate + shared layers 3-4
    - Output: 4 segmentation masks + 4 HV maps
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
        
        # ========== H&E ENCODER (layers 1-2) ==========
        self.he_conv1 = resnet.conv1  # Pretrained 3-channel
        self.he_bn1 = resnet.bn1
        self.he_relu = resnet.relu
        self.he_maxpool = resnet.maxpool
        self.he_layer1 = resnet.layer1
        self.he_layer2 = resnet.layer2
        
        # ========== MIF ENCODER (layers 1-2) ==========
        # Create separate ResNet for MIF
        if backbone == 'resnet34':
            resnet_mif = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet50':
            resnet_mif = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet101':
            resnet_mif = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify first conv for 2 channels
        self.mif_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Initialize with pretrained weights (average of RGB channels)
            with torch.no_grad():
                self.mif_conv1.weight[:, :, :, :] = resnet_mif.conv1.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        
        self.mif_bn1 = resnet_mif.bn1
        self.mif_relu = resnet_mif.relu
        self.mif_maxpool = resnet_mif.maxpool
        self.mif_layer1 = resnet_mif.layer1
        self.mif_layer2 = resnet_mif.layer2
        
        # ========== FUSION LAYER ==========
        # Concatenate H&E + MIF features after layer2
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels[2] * 2, channels[2], 1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # ========== SHARED ENCODER (layers 3-4) ==========
        if backbone == 'resnet34':
            resnet_shared = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet50':
            resnet_shared = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet101':
            resnet_shared = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.shared_layer3 = resnet_shared.layer3
        self.shared_layer4 = resnet_shared.layer4
        
        # Dropout
        self.dropout_high = nn.Dropout2d(dropout_rate * 1.5)
        self.dropout_mid = nn.Dropout2d(dropout_rate)
        
        # ========== DECODERS ==========
        self._build_decoders()
        
    def _build_decoders(self):
        """Build 4 separate decoder branches"""
        channels = self.channels
        dropout_rate = self.dropout_rate
        
        # HE NUCLEI DECODER
        self.up4_he_nuclei = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4_he_nuclei = self.conv_block(channels[4], channels[3], dropout_rate * 1.2)
        self.up3_he_nuclei = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3_he_nuclei = self.conv_block(channels[3], channels[2], dropout_rate)
        self.up2_he_nuclei = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2_he_nuclei = self.conv_block(channels[2], channels[1], dropout_rate * 0.8)
        self.up1_he_nuclei = nn.ConvTranspose2d(channels[1], 64, 2, stride=2)
        self.dec1_he_nuclei = self.conv_block(192, 64, dropout_rate * 0.5)  # 64 + 128 = 192
        self.up0_he_nuclei = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0_he_nuclei = self.conv_block(32, 32, dropout_rate * 0.3)
        self.final_he_nuclei_seg = nn.Conv2d(32, 1, 1)
        self.final_he_nuclei_hv = nn.Conv2d(32, 2, 1)
        
        # HE CELL DECODER
        self.up4_he_cell = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4_he_cell = self.conv_block(channels[4], channels[3], dropout_rate * 1.2)
        self.up3_he_cell = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3_he_cell = self.conv_block(channels[3], channels[2], dropout_rate)
        self.up2_he_cell = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2_he_cell = self.conv_block(channels[2], channels[1], dropout_rate * 0.8)
        self.up1_he_cell = nn.ConvTranspose2d(channels[1], 64, 2, stride=2)
        self.dec1_he_cell = self.conv_block(192, 64, dropout_rate * 0.5)  # 64 + 128 = 192
        self.up0_he_cell = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0_he_cell = self.conv_block(32, 32, dropout_rate * 0.3)
        self.final_he_cell_seg = nn.Conv2d(32, 1, 1)
        self.final_he_cell_hv = nn.Conv2d(32, 2, 1)
        
        # MIF NUCLEI DECODER
        self.up4_mif_nuclei = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4_mif_nuclei = self.conv_block(channels[4], channels[3], dropout_rate * 1.2)
        self.up3_mif_nuclei = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3_mif_nuclei = self.conv_block(channels[3], channels[2], dropout_rate)
        self.up2_mif_nuclei = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2_mif_nuclei = self.conv_block(channels[2], channels[1], dropout_rate * 0.8)
        self.up1_mif_nuclei = nn.ConvTranspose2d(channels[1], 64, 2, stride=2)
        self.dec1_mif_nuclei = self.conv_block(192, 64, dropout_rate * 0.5)  # 64 + 128 = 192
        self.up0_mif_nuclei = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0_mif_nuclei = self.conv_block(32, 32, dropout_rate * 0.3)
        self.final_mif_nuclei_seg = nn.Conv2d(32, 1, 1)
        self.final_mif_nuclei_hv = nn.Conv2d(32, 2, 1)
        
        # MIF CELL DECODER
        self.up4_mif_cell = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4_mif_cell = self.conv_block(channels[4], channels[3], dropout_rate * 1.2)
        self.up3_mif_cell = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3_mif_cell = self.conv_block(channels[3], channels[2], dropout_rate)
        self.up2_mif_cell = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2_mif_cell = self.conv_block(channels[2], channels[1], dropout_rate * 0.8)
        self.up1_mif_cell = nn.ConvTranspose2d(channels[1], 64, 2, stride=2)
        self.dec1_mif_cell = self.conv_block(192, 64, dropout_rate * 0.5)  # 64 + 128 = 192
        self.up0_mif_cell = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0_mif_cell = self.conv_block(32, 32, dropout_rate * 0.3)
        self.final_mif_cell_seg = nn.Conv2d(32, 1, 1)
        self.final_mif_cell_hv = nn.Conv2d(32, 2, 1)
    
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
    
    def decode_branch(self, x4, x3, x2, x1, x0_he, x0_mif, branch_name):
        """Generic decoder branch with skip connections from both modalities"""
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
        
        # Combine early features from both modalities (64 + 64 = 128)
        x0_combined = torch.cat([x0_he, x0_mif], dim=1)
        
        # up1(d2) = 64 channels, x0_combined = 128 channels -> 192 total
        d1 = dec1(torch.cat([up1(d2), x0_combined], dim=1))
        d0 = dec0(up0(d1))
        
        seg_out = torch.sigmoid(final_seg(d0))
        hv_out = torch.tanh(final_hv(d0))
        
        return seg_out, hv_out
    
    def forward(self, he_img, mif_img):
        """
        Args:
            he_img: (B, 3, H, W) - H&E image
            mif_img: (B, 2, H, W) - MIF image
        """
        # ========== H&E ENCODER ==========
        he_x0 = self.he_relu(self.he_bn1(self.he_conv1(he_img)))
        he_x0_pool = self.he_maxpool(he_x0)
        he_x1 = self.he_layer1(he_x0_pool)
        he_x2 = self.he_layer2(he_x1)
        
        # ========== MIF ENCODER ==========
        mif_x0 = self.mif_relu(self.mif_bn1(self.mif_conv1(mif_img)))
        mif_x0_pool = self.mif_maxpool(mif_x0)
        mif_x1 = self.mif_layer1(mif_x0_pool)
        mif_x2 = self.mif_layer2(mif_x1)
        
        # ========== FUSION ==========
        fused_x2 = self.fusion_conv(torch.cat([he_x2, mif_x2], dim=1))
        
        # ========== SHARED ENCODER ==========
        x3 = self.shared_layer3(fused_x2)
        if self.training:
            x3 = self.dropout_mid(x3)
        
        x4 = self.shared_layer4(x3)
        if self.training:
            x4 = self.dropout_high(x4)
        
        # ========== DECODERS ==========
        he_nuclei_seg, he_nuclei_hv = self.decode_branch(x4, x3, fused_x2, he_x1, he_x0, mif_x0, 'he_nuclei')
        he_cell_seg, he_cell_hv = self.decode_branch(x4, x3, fused_x2, he_x1, he_x0, mif_x0, 'he_cell')
        mif_nuclei_seg, mif_nuclei_hv = self.decode_branch(x4, x3, fused_x2, mif_x1, he_x0, mif_x0, 'mif_nuclei')
        mif_cell_seg, mif_cell_hv = self.decode_branch(x4, x3, fused_x2, mif_x1, he_x0, mif_x0, 'mif_cell')
        
        return {
            'he_nuclei_seg': he_nuclei_seg,
            'he_nuclei_hv': he_nuclei_hv,
            'he_cell_seg': he_cell_seg,
            'he_cell_hv': he_cell_hv,
            'mif_nuclei_seg': mif_nuclei_seg,
            'mif_nuclei_hv': mif_nuclei_hv,
            'mif_cell_seg': mif_cell_seg,
            'mif_cell_hv': mif_cell_hv,
        }


# Alias for backward compatibility
MultiModalPathologyUNet = DualEncoderUNet
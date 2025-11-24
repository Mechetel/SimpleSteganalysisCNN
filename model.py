import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class StegoCNN(nn.Module):
    def __init__(self, pretrained=True, freeze_resnet=False):
        super(StegoCNN, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final fully connected layer
        # ResNet-50 has 2048 features before the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze ResNet backbone if requested
        if freeze_resnet:
            for param in self.features.parameters():
                param.requires_grad = False

        # Add custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



import numpy as np
from utils import get_srm_kernels
import torch.nn.functional as F

# ==================== Attention Modules ====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel Attention for CBAM"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention for CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class GCBlock(nn.Module):
    """Global Context Block"""
    def __init__(self, channels, reduction=16):
        super(GCBlock, self).__init__()
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LayerNorm([channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        mask = self.conv_mask(x).view(b, 1, h * w)
        mask = self.softmax(mask).view(b, 1, h, w)
        context = (x * mask).view(b, c, h * w).sum(dim=2, keepdim=True).view(b, c, 1, 1)
        context = self.transform(context)
        return x + context


class TripletAttention(nn.Module):
    """Triplet Attention Module"""
    def __init__(self, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.cw = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )
        self.hc = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )
        self.hw = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # C-W attention
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_avg1 = torch.mean(x_perm1, dim=1, keepdim=True)
        x_max1, _ = torch.max(x_perm1, dim=1, keepdim=True)
        att1 = self.sigmoid(self.cw(torch.cat([x_avg1, x_max1], dim=1)))
        att1 = att1.permute(0, 2, 1, 3).contiguous()

        # H-C attention
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_avg2 = torch.mean(x_perm2, dim=1, keepdim=True)
        x_max2, _ = torch.max(x_perm2, dim=1, keepdim=True)
        att2 = self.sigmoid(self.hc(torch.cat([x_avg2, x_max2], dim=1)))
        att2 = att2.permute(0, 3, 2, 1).contiguous()

        # H-W attention
        x_avg3 = torch.mean(x, dim=1, keepdim=True)
        x_max3, _ = torch.max(x, dim=1, keepdim=True)
        att3 = self.sigmoid(self.hw(torch.cat([x_avg3, x_max3], dim=1)))

        return (x * att1 + x * att2 + x * att3) / 3.0


class InceptionAttentionBlock(nn.Module):
    """Inception-style block with all four attention mechanisms"""
    def __init__(self, in_channels, out_channels, reduction=16):
        super(InceptionAttentionBlock, self).__init__()

        branch_channels = out_channels // 4

        # Branch 1: SE Block with 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            SEBlock(branch_channels, reduction)
        )

        # Branch 2: CBAM with 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            CBAMBlock(branch_channels, reduction)
        )

        # Branch 3: GC Block with 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            GCBlock(branch_channels, reduction)
        )

        # Branch 4: Triplet Attention with 3x3 conv
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            TripletAttention()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out)


# ==================== Building Blocks ====================

class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution for efficiency"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    """Residual Block with optional separable convolution"""
    def __init__(self, channels, use_separable=False):
        super(ResidualBlock, self).__init__()
        if use_separable:
            self.conv1 = SeparableConv2d(channels, channels, 3, 1, 1)
            self.conv2 = SeparableConv2d(channels, channels, 3, 1, 1)
        else:
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


# ==================== Main Network ====================

class INATNet(nn.Module):
    """
    Advanced Steganalysis Network combining best practices from:
    - XuNet: SRM filters, ABS layer, efficient 1x1 convolutions
    - SRNet: Deep architecture, minimal heuristics
    - YeNet/YedroudjNet: Preprocessing with BatchNorm
    - ZhuNet: Strong discriminative power with residual connections
    """
    def __init__(self):
        super(INATNet, self).__init__()

        # ===== PREPROCESSING STAGE (XuNet/YeNet inspired) =====
        # Initialize with 30 SRM high-pass filters
        self.srm_conv = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self._init_srm_filters()
        # Freeze SRM filters (common practice)
        for param in self.srm_conv.parameters():
            param.requires_grad = False

        # TLU (Truncated Linear Unit) - clips values to [-T, T]
        self.tlu_threshold = 3.0

        # Additional learnable preprocessing
        self.prep_conv = nn.Conv2d(30, 64, kernel_size=3, padding=1, bias=False)
        self.prep_bn = nn.BatchNorm2d(64)

        # ===== FEATURE EXTRACTION STAGE =====
        # Stage 1: Initial feature extraction
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )

        # Stage 2: Increase channels
        self.stage2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2_res = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 128 -> 64

        # Stage 3: Deep features with separable convolutions
        self.stage3_conv = nn.Sequential(
            SeparableConv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage3_res = nn.Sequential(
            ResidualBlock(256, use_separable=True),
            ResidualBlock(256, use_separable=True),
            ResidualBlock(256, use_separable=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 64 -> 32

        # Stage 4: High-level features
        self.stage4_conv = nn.Sequential(
            SeparableConv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.stage4_res = nn.Sequential(
            ResidualBlock(512, use_separable=True),
            ResidualBlock(512, use_separable=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2)  # 32 -> 16

        # ===== INCEPTION ATTENTION BLOCK (near the end for best performance) =====
        self.inception_attention = InceptionAttentionBlock(512, 512, reduction=16)

        # Final feature refinement
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )

        # ===== CLASSIFICATION STAGE =====
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_srm_filters(self):
        """Initialize SRM filters"""
        srm_kernels = get_srm_kernels()
        weight = np.zeros((30, 3, 5, 5), dtype=np.float32)
        for i, kernel in enumerate(srm_kernels):
            # Pad 3x3 kernels to 5x5
            if kernel.shape[0] == 3:
                padded = np.zeros((5, 5), dtype=np.float32)
                padded[1:4, 1:4] = kernel
                kernel = padded
            # Apply same kernel to all color channels
            for c in range(3):
                weight[i, c, :, :] = kernel

        self.srm_conv.weight.data = torch.from_numpy(weight)

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Preprocessing with SRM filters
        x = self.srm_conv(x)

        # TLU (Truncated Linear Unit) - helps with convergence
        x = torch.clamp(x, -self.tlu_threshold, self.tlu_threshold)

        # Absolute value layer (common in steganalysis)
        x = torch.abs(x)

        # Learnable preprocessing
        x = self.prep_bn(self.prep_conv(x))
        x = F.relu(x)

        # Feature extraction stages
        x = self.stage1(x)

        x = self.stage2_conv(x)
        x = self.stage2_res(x)
        x = self.pool2(x)

        x = self.stage3_conv(x)
        x = self.stage3_res(x)
        x = self.pool3(x)

        x = self.stage4_conv(x)
        x = self.stage4_res(x)
        x = self.pool4(x)

        # Inception attention block (near the end for best performance)
        x = self.inception_attention(x)

        # Final refinement
        x = self.final_conv(x)

        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

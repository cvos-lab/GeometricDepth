# HiDNet + Attention Gates (AG)
# ===========================================
# ✅ Adds AG to skip connections to suppress irrelevant encoder features
# Other features (CBAM, DropBlock, Residual, Dilation) remain unchanged
# ===========================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# DropBlock
class DropBlock2D(nn.Module):
    def __init__(self, block_size=5, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        for s in x.shape[2:]:
            gamma *= s / (s - self.block_size + 1)
        mask = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) < gamma).float()
        mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        return x * mask * (mask.numel() / mask.sum())

# CBAM (Channel + Spatial Attention)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(alpha, inplace=True),
            CBAM(out_channels)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
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

class HiDNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, dropout=0.2, alpha=0.3):
        super().__init__()

        def block(x_in, x_out):
            return ResidualCBAMBlock(x_in, x_out, alpha)

        self.enc1 = block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = block(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(base_filters * 16, base_filters * 16, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(alpha, inplace=True),
            CBAM(base_filters * 16)
        )
        self.dropblock = DropBlock2D(block_size=5, drop_prob=dropout)

        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.ag4 = AttentionGate(base_filters * 8, base_filters * 8, base_filters * 4)
        self.dec4 = block(base_filters * 16, base_filters * 8)

        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.ag3 = AttentionGate(base_filters * 4, base_filters * 4, base_filters * 2)
        self.dec3 = block(base_filters * 8, base_filters * 4)

        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.ag2 = AttentionGate(base_filters * 2, base_filters * 2, base_filters)
        self.dec2 = block(base_filters * 4, base_filters * 2)

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.ag1 = AttentionGate(base_filters, base_filters, base_filters // 2)
        self.dec1 = block(base_filters * 2, base_filters)

        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.dropblock(self.bottleneck(p4))

        g4 = self.upconv4(b)
        a4 = self.ag4(g4, e4)
        d4 = self.dec4(torch.cat([g4, a4], dim=1))

        g3 = self.upconv3(d4)
        a3 = self.ag3(g3, e3)
        d3 = self.dec3(torch.cat([g3, a3], dim=1))

        g2 = self.upconv2(d3)
        a2 = self.ag2(g2, e2)
        d2 = self.dec2(torch.cat([g2, a2], dim=1))

        g1 = self.upconv1(d2)
        a1 = self.ag1(g1, e1)
        d1 = self.dec1(torch.cat([g1, a1], dim=1))

        out = self.final_conv(d1)
        out = torch.sigmoid(out)
        return out

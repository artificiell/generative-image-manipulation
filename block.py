import torch
import torch.nn as nn

class SingleDownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm = True):
        super(SingleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.bn is not None:
            return self.relu(self.bn(self.conv(x)))
        return self.relu(self.conv(x))

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm = True):
        super(DoubleConvBlock, self).__init__()
        self.dconvb1 = SingleConvBlock(in_channels, out_channels, batch_norm)
        self.dconvb2 = SingleConvBlock(out_channels, out_channels, batch_norm)
    def forward(self, x):
        return self.dconvb2(self.dconvb1(x))
    
class SingleUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(SingleUpConvBlock, self).__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SingleConvBlock(in_channels, out_channels)

    def forward(self, x, skip_x):
        return self.conv(torch.cat([self.up(x), skip_x], dim=1))

class DoubleUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(DoubleUpConvBlock, self).__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, skip_x):
        return self.conv(torch.cat([self.up(x), skip_x], dim=1))

import torch
import torch.nn as nn

class SingleConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, batch_norm=True):
        super(SingleConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(output_dim) if batch_norm else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.bn is not None:
            return self.relu(self.bn(self.conv(x)))
        return self.relu(self.conv(x))

class DoubleConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, batch_norm=True):
        super(DoubleConvBlock, self).__init__()
        self.dconvb1 = SingleConvBlock(input_dim, output_dim, kernel_size, stride, padding, batch_norm)
        self.dconvb2 = SingleConvBlock(output_dim, output_dim, kernel_size, stride, padding, batch_norm)

    def forward(self, x):
        return self.dconvb2(self.dconvb1(x))

class UpSingleConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=2, stride=2, padding=0, up_mode='transpose'):
        super(UpSingleConvBlock, self).__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SingleConvBlock(input_dim, output_dim) 

    def forward(self, x, skip_x, custom_padding=False):
        x = self.up(x)
        
        if custom_padding:
            d_x, d_y = skip_x.size()[3] - x.size()[3], skip_x.size()[2] - x.size()[2] # Input is CHW
            x = nn.functional.pad(x, [d_x // 2, d_x - d_x // 2, d_y // 2, d_y - d_y // 2])

        y = torch.cat([skip_x, x], dim=1)
        return self.conv(y)

class UpDoubleConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=2, stride=2, padding=0, up_mode='transpose'):
        super(UpDoubleConvBlock, self).__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvBlock(input_dim, output_dim) 

    def forward(self, x, skip_x, custom_padding=False):
        x = self.up(x)

        if custom_padding:
            d_x, d_y = skip_x.size()[3] - x.size()[3], skip_x.size()[2] - x.size()[2] # Input is CHW
            x = nn.functional.pad(x, [d_x // 2, d_x - d_x // 2, d_y // 2, d_y - d_y // 2])

        y = torch.cat([skip_x, x], dim=1)
        return self.conv(y)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients, kernel_size=1, stride=1, padding=0):
        super(AttentionBlock, self).__init__()

        # Gate sequence (based on the number of feature maps in previous layer)
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        # Skip connection sequence (based on the number of feature, corresponding encoder layer, transferred via skip connection)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        # Output activation 
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_x):
        psi = self.psi(self.relu(self.W_gate(gate) + self.W_x(skip_x)))
        return skip_x * psi


class UpAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=2, stride=2, padding=0, up_mode='upsample'):
        super(UpAttentionBlock, self).__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sconvb = SingleConvBlock(input_dim, output_dim) 
        self.att = AttentionBlock(F_g = output_dim, F_l = output_dim, n_coefficients = output_dim // 2)
        self.dconvb = SingleConvBlock(input_dim, output_dim)
        
    def forward(self, x, skip_x, custom_padding=False):
        y = self.sconvb(self.up(x))
        if custom_padding:
            d_x, d_y = skip_x.size()[3] - y.size()[3], skip_x.size()[2] - y.size()[2] # Input is CHW
            y = nn.functional.pad(y, [d_x // 2, d_x - d_x // 2, d_y // 2, d_y - d_y // 2])
        skip_x = self.att(y, skip_x)
        y = torch.cat([skip_x, y], dim=1)
        return self.dconvb(y)

class ResidualConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=2, padding=1):
        super(ResidualConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=padding), # stride = default (1)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)
                                                
class ResidualInputBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1):
        super(ResidualInputBlock, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        )
        self.input_skip = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        return self.input_layer(x) + self.input_skip(x)
    
class UpResidualConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=2, padding=2):
        super(UpResidualConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.res_conv = nn.ConvTranspose2d(input_dim + output_dim, input_dim, kernel_size=kernel_size, stride=stride // 2, padding=padding // 2)

    def forward(self, x, skip_x):
        y = self.up(x)
        if custom_padding:
            d_x, d_y = skip_x.size()[3] - y.size()[3], skip_x.size()[2] - y.size()[2] # Input is CHW
            y = nn.functional.pad(y, [d_x // 2, d_x - d_x // 2, d_y // 2, d_y - d_y // 2])
        y = torch.cat([y, skip_x], dim=1)
        return self.res_conv(y)

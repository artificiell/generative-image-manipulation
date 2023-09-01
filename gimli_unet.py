import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from block import SingleConvBlock, SingleUpConvBlock

# Generator model
class generator(nn.Module):
    def __init__(self, num_channels = 3, mlp_nfilters = 32, mlp_hidden = 256, height_hidden = 8, width_hidden = 12):
        super(generator, self).__init__()        
        self.mlp_nfilters = mlp_nfilters

        # Joint pool layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Image encoder pathway
        self.encoder1 = SingleConvBlock(num_channels, 64)
        self.encoder2 = SingleConvBlock(64, 128)
        self.encoder3 = SingleConvBlock(128, 256)
        self.encoder4 = SingleConvBlock(256, 512)
                        
        # Linear transform (y = xA^T + b)
        self.g = nn.Sequential(
            nn.Linear(1412, mlp_hidden),
            nn.ReLU(True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(True),
            nn.Linear(mlp_hidden, height_hidden * width_hidden * self.mlp_nfilters),
            nn.ReLU(True)
        )
        x_coords = torch.linspace(-1, 1, width_hidden)
        y_coords = torch.linspace(-1, 1, height_hidden)
        self.x_grid, self.y_grid = torch.meshgrid(x_coords, y_coords)
        
        # Image decoder pathway
        self.decoder1 = DoubleUpConvBlock(1024, 512)
        self.decoder2 = DoubleUpConvBlock(512, 256)
        self.decoder3 = DoubleUpConvBlock(256, 128)
        self.decoder4 = DoubleUpConvBlock(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, num_channels, kernel_size=1)
        
    def forward(self, x, sen_embed):

        # Image encoder
        phi_im_enc1 = self.encoder1(x)
        phi_im_enc2 = self.encoder2(self.pool(phi_im_enc1))
        phi_im_enc3 = self.encoder3(self.pool(phi_im_enc2))
        phi_im = self.encoder4(self.pool(phi_im_enc3))
        batch_size, n_channel, conv_h, conv_w = phi_im.size()
        n_pair = conv_h * conv_w
        
        x_grid = self.x_grid.reshape(1, 1, conv_h, conv_w).repeat(batch_size, 1, 1, 1)
        y_grid = self.y_grid.reshape(1, 1, conv_h, conv_w).repeat(batch_size, 1, 1, 1)
        coord_tensor = torch.cat((x_grid, y_grid), dim=1).to(phi_im.device)
        
        im_enc_with_coords = torch.cat([phi_im, coord_tensor], 1)
        im_enc_i = im_enc_with_coords.unsqueeze(1).repeat(1, n_pair, 1, 1)
        im_enc_j = im_enc_with_coords.unsqueeze(2).repeat(1, 1, n_pair, 1)
        im_enc_stacked = torch.cat([im_enc_i, im_enc_j], 3)

        phi_s = sen_embed.unsqueeze(2).unsqueeze(3).repeat(1, n_pair, 1, n_pair)
        sen_embed_stacked = sen_embed.unsqueeze(1).unsqueeze(2).repeat(1, n_pair, n_pair, 1)
        
        # Relational module
        phi = torch.cat([im_enc_stacked, sen_embed_stacked], 3)
        phi = phi.view(batch_size * (n_pair**2), 1412)
        phi = self.g(phi)
        phi = phi.view(batch_size, n_pair**2, n_pair * self.mlp_nfilters).sum(1)
        phi = phi.view(batch_size, self.mlp_nfilters, conv_h, conv_w)
        
        # Decoder
        phi_im_dec1 = self.encoder1(phi, phi_im)
        phi_im_enc2 = self.encoder2(phi_im_dec1, phi_im_enc3)
        phi_im_enc3 = self.encoder2(phi_im_dec2, phi_im_enc2)
        phi_im_enc4 = self.encoder2(phi_im_dec3, phi_im_enc1)
        y = self.outconv(phi_im_enc4)

        return y, phi, phi_im, phi_s
    

import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

from block import SingleConvBlock, DoubleConvBlock, UpAttentionBlock, UpSingleConvBlock, UpDoubleConvBlock

# Generator model
class generator(nn.Module):
    def __init__(self, num_channels = 3, mlp_nfilters = 32, mlp_hidden = 256, height_hidden = 7, width_hidden = 11):
        super(generator, self).__init__()        
        self.mlp_nfilters = mlp_nfilters

        # Joint pool layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Image encoder pathway
        self.encoder1 = DoubleConvBlock(num_channels, 64)
        self.encoder2 = DoubleConvBlock(64, 128)
        self.encoder3 = DoubleConvBlock(128, 256)
        self.encoder4 = DoubleConvBlock(256, 512)

        self.bottleneck = DoubleConvBlock(512, 1024)
        self.preconv = DoubleConvBlock(2050, 1024)
        
        # Linear projection
        self.projection = nn.Sequential(
            nn.Linear(384, mlp_hidden),
            nn.ReLU(True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(True),
            nn.Linear(mlp_hidden, 1024),
            nn.ReLU(True)
        )
        self.x_grid, self.y_grid = torch.meshgrid(
            torch.linspace(-1, 1, width_hidden),
            torch.linspace(-1, 1, height_hidden)
        )
        
        # Image decoder pathway
        self.decoder1 = UpAttentionBlock(1024, 512)
        self.decoder2 = UpAttentionBlock(512, 256)
        self.decoder3 = UpAttentionBlock(256, 128)
        self.decoder4 = UpAttentionBlock(128, 64)
        
        # Output layer
        self.outconv = nn.Conv2d(64, num_channels, kernel_size=1)

        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

                
    def forward(self, x, sen_embed):

        # Image encoder
        phi_im_enc1 = self.encoder1(x)
        phi_im_enc2 = self.encoder2(self.pool(phi_im_enc1))
        phi_im_enc3 = self.encoder3(self.pool(phi_im_enc2))
        phi_im_enc4 = self.encoder4(self.pool(phi_im_enc3))
        phi_im = self.bottleneck(self.pool(phi_im_enc4))

        # Cast all pairs against each other
        batch_size, n_channel, conv_h, conv_w = phi_im.size()
        x_grid = self.x_grid.reshape(1, 1, conv_h, conv_w).repeat(batch_size, 1, 1, 1)
        y_grid = self.y_grid.reshape(1, 1, conv_h, conv_w).repeat(batch_size, 1, 1, 1)
        coord_tensor = torch.cat((x_grid, y_grid), dim=1).cuda()
        phi_im_coords = torch.cat([phi_im, coord_tensor], dim=1)

        # Sentence embedding
        phi_s = sen_embed.view(-1, sen_embed.shape[-1])
        projected = self.projection(sen_embed)

        # Relational module
        phi = torch.cat([phi_im_coords, projected.view(batch_size, 1024, 1, 1).expand(-1, -1, 7, 11)], dim=1)
        phi_pre = self.preconv(phi)
        
        # Decoder
        phi_im_dec1 = self.decoder1(phi_pre, phi_im_enc4, custom_padding=True)
        phi_im_dec2 = self.decoder2(phi_im_dec1, phi_im_enc3, custom_padding=True)
        phi_im_dec3 = self.decoder3(phi_im_dec2, phi_im_enc2)
        phi_im_dec4 = self.decoder4(phi_im_dec3, phi_im_enc1)
        y = self.outconv(phi_im_dec4)
        
        return y, phi, phi_im, phi_s
    

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

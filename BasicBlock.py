"""
Basic Blocks for the cycle GAN. `UnetGenerator` and `PathcDiscriminator`

Author: Rayykia

Discription
===========
* [batch, 3, 256, 256]->``UnetGenerator``->[batch, 3, 256, 256]
* [batch, 3, 256, 256]*2->`` PatchDiscriminator``->[batch, 1, 60, 60]
"""

# Loss:
# 1. adversarial
# 2. cycle consistency loss
# 3. identity loss

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import itertools

__all__ = ['UnetGenerator', 'PatchDiscriminator']

class DownsampleBlock(nn.Module):
    """Construct a downsample block for UNet.

    Attrs:

        input_nc (int):
            the number of channels in the input image
        output_nc (int):
            the number of channels in the ouptut image
        use_in (bool):
            if use instance normalization after relu, nor used in ``the last layer of generator`` and
            ``the first layer in the discriminator``
    """
    def __init__(self, input_nc: int, output_nc: int, use_in: bool = True,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model = [nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1), 
                               nn.LeakyReLU( inplace=True))]
        if use_in:
            model +=[nn.InstanceNorm2d(output_nc)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor):
        """Standard forward."""
        return self.model(x)


class UpsampleBlock(nn.Module):
    """Construct a upsample blcok for UNet.

    Attrs:

        input_nc (int):
            the number of channels in the input image
        output_nc (int):
            the number of channels in the output image
        use_dropour (bool):
            if use dropout, used in the ``first three layers of the generator``
    """
    def __init__(self, input_nc: int, output_nc: int, use_dropout: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        conv_relu = [nn.Sequential(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  nn.LeakyReLU(0.02, True))]
        norm_layer = [nn.InstanceNorm2d(output_nc)]
        if use_dropout:
            model = conv_relu + norm_layer + [nn.Dropout2d()]
        else:
            model = conv_relu + norm_layer
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor):
        """Standard forward."""
        return self.model(x)


class UnetGenerator(nn.Module):
    """Construct a UNet-based generator. ``7 downsample blocks`` and ``6 upsample blocks``

    Attrs:
        
        input_nc (int):
            the channels of the input image, defult ``3``
        output_nc (int):
            the channels of the ouptut image, defult ``3``
    """
    def __init__(self, input_nc: int = 3, output_nc: int = 3,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)               
        # downsample
        self.down1 = DownsampleBlock(input_nc, 64)              # 64, 128, 128 
        self.down2 = DownsampleBlock(64, 128)                   # 128, 64, 64   -> up5
        self.down3 = DownsampleBlock(128, 256)                  # 256, 32, 32   -> up4
        self.down4 = DownsampleBlock(256, 512)                  # 512, 16, 16   -> up3
        self.down5 = DownsampleBlock(512, 512)                  # 512, 8, 8     -> up2
        self.down6 = DownsampleBlock(512, 512)                  # 512, 4, 4     -> up1
        self.down7 = DownsampleBlock(512, 512)
        # upsample
        self.up1 = UpsampleBlock(512, 512, True)
        self.up2 = UpsampleBlock(1024, 512, True)                # 512, 8, 8     + down5
        self.up3 = UpsampleBlock(1024, 512, True)               # 512, 16, 16   + down4
        self.up4 = UpsampleBlock(1024, 256)                     # 256, 32, 32   + down3
        self.up5 = UpsampleBlock(512, 128)                      # 128, 64, 64   + down2
        self.up6 = UpsampleBlock(256, 64)                       # 64, 128, 128  + down1
        # output
        self.last = nn.Sequential(nn.ConvTranspose2d(128, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  nn.Tanh())

    def forward(self, x: Tensor):
        """Standard forward."""
        # downsample
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        # upsample
        x7 = self.up1(x7)
        x7 = torch.cat([x7, x6], dim=1)
        x7 = self.up2(x7)
        x7 = torch.cat([x7, x5], dim=1)
        x7 = self.up3(x7)
        x7 = torch.cat([x7, x4], dim=1)
        x7 = self.up4(x7)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.up5(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up6(x7)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.last(x7)
        return x7

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, use_dropout=True, n_blocks=9, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        norm_layer =nn.InstanceNorm2d
        use_bias = True

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class PatchDiscriminator(nn.Module):
    """Create a PatchGAN discriminator. patch size: ``60x60``

    Input:

        ``annotation`` + ``generated iamge / real image``

    Attrs:

        input_nc (int):
            the channels of the input image, defult ``3``   
    """
    def __init__(self, input_nc: int=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(DownsampleBlock(input_nc, 64, use_in=False),
                                   DownsampleBlock(64, 128),
                                   nn.Conv2d(128, 256, kernel_size=3, padding=2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.InstanceNorm2d(256),
                                   nn.Conv2d(256, 512, kernel_size=3, padding=2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.InstanceNorm2d(256),
                                   nn.Conv2d(512, 1, kernel_size=3, padding=2))


    def forward(self, img: Tensor):
        """Standard forward."""
        return torch.sigmoid(self.model(img))


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=2):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)




# Test the constructed model.   
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # batch, channel, width, height 8, 3, 256, 256
    test_input = torch.randn((8, 3, 256, 256)).to(device)    

    gen = ResnetGenerator(3, 3).to(device)
    output = gen(test_input)
    if list(output.shape) == [8, 3, 256, 256]:
        print('GENERATOR: PASS')
    else:
        print('GENERATOR: FAIL')
    # batch, channel, width, height
    test_input2 = torch.randn((8, 3, 256, 256)).to(device)  
    dis = PatchDiscriminator().to(device)
    output = dis(test_input)
    print("output size:", list(output.shape))
    # if list(output.shape) == [8, 1, 60, 60]:
    #     print('DISCRIMINATOR: PASS')
    # else:
    #     print('DISCRIMINATOR: FAIL')
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Backbone_conv(nn.Module):
    """
    Backbone_conv: Builds a convolutional backbone with modulable levels and layers.

    Arguments:
        - input_channels (int): Number of input channels.
        - output_channels (int): Number of output channels after the last level.
        - img_size (int): Image size.
        - num_levels (int): Number of levels in the backbone.
        - num_layers (int): Number of layers per level.

    Returns:
        - torch.Tensor: The output feature map after passing through the convolutional layers.
    """
    def __init__(self, input_channels, output_channels, img_size, num_levels, num_layers):
        super(Backbone_conv, self).__init__()
        lst_convs = []
        in_channels = input_channels
        out_channels = 64//2**(num_levels-1)
        for i in range(num_levels): # creates the levels
            lst_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            lst_convs.append(nn.BatchNorm2d(out_channels))
            lst_convs.append(nn.LeakyReLU())
            for j in range(num_layers-1):   # creates the layers per level
                lst_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
                lst_convs.append(nn.BatchNorm2d(out_channels))
                lst_convs.append(nn.LeakyReLU())
            lst_convs.append(nn.MaxPool2d(2))

            in_channels = out_channels
            out_channels *= 2

        self.convs = nn.ModuleList(nn.Sequential(*lst_convs))
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class ResFC(nn.Module):
    """
    ResFC: A residual fully connected network with global average pooling and dropout layers.

    Arguments:
    - input_channels (int): Number of input channels for the convolution.
    - output_channels (int): Number of output channels after the fully connected layers.
    - img_size (int): Input image size.

    Returns:
    - torch.Tensor: Output tensor after fully connected layers, residual connection included.
    """
    def __init__(self, input_channels, output_channels, img_size, dropout_frac):
        super(ResFC, self).__init__()
        self.output_channels = output_channels
        self.relu = nn.LeakyReLU()
        self.do = nn.Dropout(dropout_frac)

        # global averaging
        self.conv13 = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        self.bn13 = nn.BatchNorm2d(output_channels)
        self.gap = nn.AvgPool2d(int(img_size))
        #self.gap = nn.AvgPool2d(64)

        # Fully connected layer
        self.linear1 = nn.Linear(256, 128, bias=False)
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.linear3 = nn.Linear(64, 32, bias=False)
        self.linear4 = nn.Linear(32, output_channels, bias=False)

    def forward(self, x):
        batch_size = x.size()[0]
        y = x
        x = self.relu(self.bn13(self.conv13(x)))  # B x O x N/8 x N/8
        y = self.gap(y)  # B x 256 x 1 x 1
        y = y.reshape((batch_size, 256))  # B x 256
        y = self.relu(self.do(self.linear1(y)))  # B x 128
        y = self.relu(self.do(self.linear2(y)))  # B x 64
        y = self.relu(self.do(self.linear3(y)))  # B x 32
        y = self.relu(self.do(self.linear4(y)))  # B x O
        x = self.gap(x)  # B x O x 1 x 1 x 1
        x = x.reshape((batch_size, self.output_channels))  # B x O
        return x + y


class ASPPConv(nn.Sequential):
    """
    ASPPConv: A 3x3 convolution module for Atrous Spatial Pyramid Pooling (ASPP) with dilation.

    Arguments:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - dilation (int): Dilation rate for the convolution.

    Returns:
    - torch.Tensor: Output of the convolution with dilation and batch normalization.
    """
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """
    ASPPPooling: Global pooling layer for ASPP with interpolation to match input size.

    Arguments:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.

    Returns:
    - torch.Tensor: Output tensor after pooling, interpolation, and convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    ASPP: Atrous Spatial Pyramid Pooling module to extract multi-scale features with different dilation rates.

    Arguments:
    - in_channels (int): Number of input channels.
    - atrous_rates (list of int): Dilation rates for the different branches in ASPP.

    Returns:
    - torch.Tensor: Output tensor after ASPP processing.
    """
    def __init__(self, in_channels, atrous_rates, dropout_frac):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_frac))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    

class ResFC_gs(nn.Module):
    """
    ResFC: A residual fully connected network with global average pooling and dropout layers.

    Arguments:
    - input_channels (int): Number of input channels for the convolution.
    - output_channels (int): Number of output channels after the fully connected layers.
    - img_size (int): Input image size.

    Returns:
    - torch.Tensor: Output tensor after fully connected layers, residual connection included.
    """
    def __init__(self, input_channels, img_size, dropout_frac):
        super(ResFC_gs, self).__init__()
        self.relu = nn.LeakyReLU()
        self.do = nn.Dropout(dropout_frac)
        self.bn = nn.BatchNorm1d(32)

        # global averaging
        self.conv13 = nn.Conv2d(input_channels, 32, kernel_size=1, bias=False)
        self.bn13 = nn.BatchNorm2d(32)
        self.gap = nn.AvgPool2d(int(img_size))
        #self.gap = nn.AvgPool2d(64)

        # Fully connected layer
        self.linear1 = nn.Linear(256, 128, bias=False)
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.linear3 = nn.Linear(64, 32, bias=False)
        #self.linear4 = nn.Linear(32, output_channels, bias=False)

    def forward(self, x):
        batch_size = x.size()[0]
        y = x
        y = self.gap(y)  # B x 256 x 1 x 1
        y = y.reshape((batch_size, 256))  # B x 256
        y = self.relu(self.do(self.linear1(y)))  # B x 128
        y = self.relu(self.do(self.linear2(y)))  # B x 64
        y = self.relu(self.do(self.linear3(y)))  # B x 32
        #y = self.relu(self.do(self.linear4(y)))  # B x O
        x = self.relu(self.bn13(self.conv13(x)))  # B x 32 x N/8 x N/8
        x = self.gap(x)  # B x 32 x 1 x 1 x 1
        x = x.reshape((batch_size, 32))  # B x 32
        z = self.relu(self.bn(x + y))
        return z


class MLP_GlobStats(nn.Module):
    def __init__(self, input_channels):
        super(MLP_GlobStats, self).__init__()
        self.do = nn.Dropout(0.0)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(input_channels, 128, bias=False)
        #self.linear2 = nn.Linear(256, 128, bias=False)
        self.linear3 = nn.Linear(128, 64, bias=False)
        self.linear4 = nn.Linear(64, 32, bias=False)
        self.residual = nn.Linear(input_channels,32, bias=False)

    def forward(self, x):
        y = self.residual(x)
        x = self.relu(self.do(self.linear1(x)))
        #x = self.relu(self.do(self.linear2(x)))
        x = self.relu(self.do(self.linear3(x)))
        x = self.relu(self.bn(self.do(self.linear4(x))))
        return self.bn(x + y)
    

class MLP_Merge(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MLP_Merge, self).__init__()
        self.do = nn.Dropout(0.0)
        self.relu = nn.LeakyReLU()
        self.linear1 = nn.Linear(2 * input_channels, 128, bias=False)
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.linear3 = nn.Linear(64, 32, bias=False)
        self.linear4 = nn.Linear(32, output_channels, bias=False)
        self.residual = nn.Linear(2 * input_channels, output_channels, bias=False)

    def forward(self, x, y):
        z = torch.cat((x,y), dim=1)
        z_res = self.residual(z)
        z = self.relu(self.do(self.linear1(z)))
        z = self.relu(self.do(self.linear2(z)))
        z = self.relu(self.do(self.linear3(z)))
        z = self.relu(self.do(self.linear4(z)))
        return z + z_res
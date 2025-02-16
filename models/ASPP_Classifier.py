import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.model_modules import *


class ASPP_Classifier(nn.Module):
    """
    ASPP_Multi_Classifier: A classification network combining a backbone, ASPP module, and MLP head.

    Args:
        - input_channels (int): Number of input channels for the network.
        - output_channels (int): Number of output channels/classes for classification.
        - img_size (int): Input image size.
        - batch_size (int, optional): Batch size for training/inference, default is 16.
        - aspp_dilate (list of int, optional): Dilation rates for the ASPP module, default is [12, 24, 36].
        - backbone (str, optional): Type of backbone, default is "convslayers".
        - bb_layers (int, optional): Number of layers in each backbone level, default is 1.
        - bb_levels (int, optional): Number of backbone levels, default is 3.

    Returns:
        - torch.Tensor: Softmax output representing class probabilities.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        img_size,
        mode='multi',
        aspp_atrous_rates=[12, 24, 36],
        bb_layers=1,
        bb_levels=3,
        dropout_frac=0.3,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.do = nn.Dropout(p=dropout_frac)
        self.mode = mode

        # security
        assert isinstance(input_channels, int)
        assert isinstance(output_channels, int)
        assert isinstance(img_size, int)
        assert mode in ['multi', 'binary']
        all(isinstance(item, int) for item in aspp_atrous_rates)

        # if binary classification
        if mode == "binary":
            output_channels = 2

        # === Backbone ===
        self.backbone = Backbone_conv(
            input_channels=input_channels,
            output_channels=64,
            img_size=img_size,
            num_levels=bb_levels,
            num_layers=bb_layers,
            )

        # === ASPP module ===
        self.aspp = ASPP(64, aspp_atrous_rates, dropout_frac)

        # === MLP ===
        self.mlp = ResFC(
            256, output_channels, int(img_size / 2 ** (bb_levels)), dropout_frac
        )

    def forward(self, x):
        x = self.backbone(x)  # B x 64 x N/8 x N/8
        x = self.aspp(x)  # B x 256 x N/8 x N/8
        x = self.mlp(x)  # B x O

        if self.mode == 'multi':
            return self.softmax(x)
        elif self.mode == 'binary':
            return self.sigmoid(x)


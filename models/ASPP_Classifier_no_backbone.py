import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50

sys.path.insert(0,'D:\GitHubProjects\STDL_Classifier')
from models.model_modules import *

class ASPP_no_backbone(nn.Module):
    """
    ASPP_no_backbone: An alternative version without backbone.

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
    def __init__(self,input_channels, output_channels, img_size, mode='multi',  batch_size=16, aspp_atrous_rates=[12, 24, 36], backbone="convslayers", bb_layers=1, bb_levels=3):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.do = nn.Dropout(p=0.3)
        self.mode = mode

        # security
        assert isinstance(input_channels, int)
        assert isinstance(output_channels, int)
        assert isinstance(img_size, int)
        assert (backbone in ["convslayers"])
        assert mode in ['multi', 'binary']
        all(isinstance(item, int) for item in aspp_atrous_rates)

        # if binary classification
        if mode == "binary":
            output_channels = 2

        # === Backbone ===
        if backbone == "convslayers":
            self.backbone = Backbone_conv(
                input_channels=input_channels, 
                output_channels=64, 
                img_size=img_size, 
                num_levels=bb_levels,
                num_layers=bb_layers,
                )

        # === ASPP module ===
        self.aspp = ASPP(input_channels, aspp_atrous_rates)

        # === MLP ===
        self.mlp = ResFC(256, output_channels, img_size)

    def forward(self,x):
        #x = self.backbone(x) # B x 64 x N/8 x N/8
        print(x.size()) 
        x = self.aspp(x) # B x 256 x N/8 x N/8
        print(x.size())
        x = self.mlp(x) # B x O
        print(x.size())
        
        if self.mode == 'multi':
            return self.softmax(x)
        elif self.mode == 'binary':
            return self.sigmoid(x)



if __name__ == "__main__":
    import sys
    from torchvision import transforms, utils
    sys.path.insert(0,'D:\GitHubProjects\STDL_Classifier')
    from src.dataset import GreenRoofsDataset
    from src.dataset_utils import Normalize, ToTensor

    #database
    norm_boundaries = np.array([[0,255],[0,255],[0,255],[0,255],[-1,1],[0,255*3]])
    transform_composed = transforms.Compose([Normalize(norm_boundaries), ToTensor()])

    roof_dataset = GreenRoofsDataset(root_dir="./data/test/dataset_test",
                                    mode='train',
                                    data_frac=0.05,
                                    train_frac= 0.8,
                                    transform=transform_composed)
    image, label = [roof_dataset[0]['image'], roof_dataset[0]['label']]
    output_channels = len(roof_dataset.class_names)

    # dataloader
    batch_size = 4
    aspp_dilates = [8, 16, 24]
    img_size = 512
    input_channels = 6
    output_channels = 6
    dataloader = torch.utils.data.DataLoader(roof_dataset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=4,
                                )
    
    #tests model
    model = ASPP_Classifier(input_channels=input_channels,
                                   output_channels=output_channels,
                                    img_size=img_size,
                                    batch_size=batch_size,
                                    bb_layers=4,
                                    bb_levels=3
                                    ).double()
    
    num_param_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {num_param_model}")

    """for data in dataloader:
        images = data['image']
        labels = data['label'][0]
        print(f"labels: {labels}")
        quit()
        res = model(images)"""
    
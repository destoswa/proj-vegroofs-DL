import numpy as np
import pandas as pd
import torch


class Normalize(object):
    """
    Normalize the different layers' values between 0 and 1.

    Args:
        - boundaries (np.ndarray): A numpy array of shape (n, 2) containing 
        - the minimum and maximum values for each channel.

    Methods:
        __call__(sample: dict): Applies the normalization to the 'image' 
        in the sample, based on the boundaries for each channel.

    Returns:
        - dict: A dictionary with the normalized 'image' and the unchanged 'label'.
    """
    def __init__(self, boundaries):
        assert isinstance(boundaries, np.ndarray)
        assert boundaries.shape[1] == 2
        self.boundaries = boundaries

    def __call__(self, sample):
        (image, gs), label = sample['image'], sample['label']
        for i in range(image.shape[0]):
            min = self.boundaries[i,0]
            max = self.boundaries[i,1]
            image[i,...] = (image[i,...] - min)/(max - min)

            # normalize global stats
            gs[i*5] = (gs[i*5] - min)/(max - min)
            gs[i*5+1] = (gs[i*5+1] - min)/(max - min)
            gs[i*5+2] = (gs[i*5+2] - min)/(max - min)
            gs[i*5+3] = (gs[i*5+3] - min)/(max - min)
            gs[i*5+4] = (gs[i*5+4] - min)/(max - min)
        return {"image":[image,gs], "label":label}
        

class ToTensor(object):
    """
    Convert numpy ndarrays in a sample to PyTorch tensors.

    Methods:
        __call__(sample: dict): Converts the 'image' in the sample from 
        a numpy array to a PyTorch tensor, leaving the 'label' unchanged.

    Returns:
        - dict: A dictionary with the 'image' as a tensor and the unchanged 'label'.
    """
    def __call__(self, sample):
        (image, gs), label = sample['image'], sample['label']

        return {'image': [torch.from_numpy(image), torch.from_numpy(np.array(gs))], 
                'label': label}


class Crop(object):
    pass


if __name__ == "__main__":
    pass

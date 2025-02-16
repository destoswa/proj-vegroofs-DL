import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from src.dataset_utils import Normalize, ToTensor


class GreenRoofsDataset(Dataset):
    def __init__(self, root_dir, mode, data_frac=1.0, train_frac=0.8, transform=None):
        """
        Initialize the GreenRoofsDataset.

        Args:
            - root_dir (str): Path to the directory containing the dataset files.
            - mode (str): Mode of the dataset, either 'train' or 'val'.
            - data_frac (float, optional): Fraction of the total dataset to use. Default is 1.0.
            - train_frac (float, optional): Fraction of the dataset to use for training. Default is 0.8.
            - transform (callable, optional): Optional transform to apply to the samples.

        Returns:
            - None
        """
        self.mode = mode
        csv_file = os.path.join(root_dir, 'dataset.csv')
        class_name_file = os.path.join(root_dir, 'class_names.csv')

        # test parameters
        assert mode in ['train', 'val', 'inf']
        if mode != 'inf':
            assert os.path.exists(class_name_file)
        assert os.path.exists(csv_file)
        assert 0.0 <= train_frac <= 1.0

        # load samples
        if mode != 'inf':
            dataset = (
                pd.read_csv(csv_file, sep=';')
                .groupby('label')
                .sample(frac=data_frac, random_state=42)
            )

            # extract the samples that are not data augmentation
            rawset = dataset.loc[
                dataset['EGID'].astype('string')
                == dataset['original_egid'].astype('string'),
                :,
            ]
            # merge rawset and dataset with an indicator showing which ones are matching
            augmentedset = dataset.merge(
                rawset,
                on=['EGID', 'class', 'file_src', 'label', 'original_egid'],
                how='left',
                indicator=True,
            )
            # remove the matching ones to create augmented set
            augmentedset = (
                augmentedset[augmentedset['_merge'] == 'left_only']
                .drop('_merge', axis=1)
                .reset_index(drop=True)
            )

            # get training frac of each class in the rawset
            trainset = rawset.groupby('label').sample(frac=train_frac, random_state=42)
            # merge training set and rawset with an indicator showing which ones are matching
            valset = rawset.merge(
                trainset,
                on=['EGID', 'class', 'file_src', 'label', 'original_egid'],
                how='left',
                indicator=True,
            )
            # remove the matching ones to create validation set
            valset = (
                valset[valset['_merge'] == 'left_only']
                .drop('_merge', axis=1)
                .reset_index(drop=True)
            )
            # add augmented version to trainset
            trainset = pd.concat(
                [
                    trainset,
                    augmentedset.loc[
                        augmentedset['original_egid'].isin(
                            trainset['original_egid'].tolist()
                        )
                    ],
                ]
            )

            self.data = trainset if mode == 'train' else valset

            # load class names
            self.class_names = pd.read_csv(class_name_file, sep=';').cat.values

        else:
            self.data = pd.read_csv(csv_file, sep=';').sample(
                frac=data_frac, random_state=42
            )

        # other parameters
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int or torch.Tensor): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image and its associated label.
                  Keys are 'image' and 'label'.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = (
            os.path.join(self.root_dir, self.data.iloc[idx, 1])
            if self.mode == 'inf'
            else os.path.join(self.root_dir, self.data.iloc[idx, 2])
        )
        with open(img_name, 'rb') as input_file:
            img_arr = pickle.load(input_file)
        egid = self.data.iloc[idx, 0]

        sample = {}
        if self.mode != 'inf':
            label = self.data.iloc[idx, 3]
            sample = {'image': img_arr, 'label': [label, egid]}
        else:
            sample = {'image': img_arr, 'label': egid}

        if self.transform:
            sample = self.transform(sample)
        return sample

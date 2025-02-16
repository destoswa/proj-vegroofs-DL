import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# for testing purposes
import sys
sys.path.insert(0,'D:\GitHubProjects\STDL_Classifier')
sys.path.insert(0,'/mnt/data-volume-01/destouch/proj_vegroofs_DL/')

from src.dataset_utils import Normalize, ToTensor

class GreenRoofsDataset(Dataset):

    def __init__(self, root_dir, mode, data_frac=1.0, train_frac=0.8, transform=None, with_gs=False):
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
        self.with_gs = with_gs
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
            dataset = pd.read_csv(csv_file, sep=';').groupby('label').sample(frac=data_frac, random_state=42)
            rawset = dataset.loc[dataset['EGID'].astype('string') == dataset['original_egid'].astype('string'),:]   # extract the samples that are not data augmentation
            augmentedset = dataset.merge(rawset, on=['EGID','class','file_src','label','original_egid'], how='left', indicator=True)  # merge rawset and dataset with an indicator showing which ones are matching
            augmentedset = augmentedset[augmentedset['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)   # remove the matching ones to create augmented set

            trainset = rawset.groupby('label').sample(frac=train_frac, random_state=42) # get training frac of each class in the rawset
            valset = rawset.merge(trainset, on=['EGID','class','file_src','label','original_egid'], how='left', indicator=True)  # merge training set and rawset with an indicator showing which ones are matching
            valset = valset[valset['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)   # remove the matching ones to create validation set
            trainset = pd.concat([trainset, augmentedset.loc[augmentedset['original_egid'].isin(trainset['original_egid'].tolist())]])  # add augmented version to trainset
            
            self.data = trainset if mode == 'train' else valset

            # load class names
            self.class_names = pd.read_csv(class_name_file, sep=';').cat.values
        else:
            self.data = pd.read_csv(csv_file, sep=';').sample(frac=data_frac, random_state=42)

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
        
        img_name = os.path.join(self.root_dir, self.data.iloc[idx,1]) if self.mode == 'inf' else os.path.join(self.root_dir, self.data.iloc[idx,2])
        #egid = img_name.replace('\\', '/').split('/')[-1].split('_')[0].split('.')[0]
        gs_name = '/'.join(img_name.replace('\\', '/').split('/')[:-1]) + '/' + str(self.data.iloc[idx].original_egid) + '_global_stats.pickle'
        os_arr = np.array([])
        with open(img_name, 'rb') as input_file:
            img_arr = pickle.load(input_file)
        if self.with_gs:
            with open(gs_name, 'rb') as input_file:
                os_arr = pickle.load(input_file)

            
        egid = self.data.iloc[idx,0]
        
        sample = {}
        if self.mode != 'inf':
            label = self.data.iloc[idx,3]
            sample = {'image': [img_arr, os_arr] , 'label': [label, egid]}
        else:
            sample = {'image': [img_arr, os_arr], 'label': egid}


        if self.transform:
            sample = self.transform(sample)
        return sample
    

if __name__ == '__main__':
    # testing
    norm_boundaries = np.array([[0,255],[0,255],[0,255],[0,255],[-1,1],[0,255*3]])
    transform_composed = transforms.Compose([Normalize(norm_boundaries), ToTensor()])
    dataset = pd.read_csv("./data/dataset/dataset.csv", sep=';')
    print(len(dataset))
    print(len(dataset[dataset.EGID == dataset.original_egid.astype('string')]))
    dataset_full = GreenRoofsDataset(root_dir="./data/dataset",
                                    mode='train',
                                    data_frac=1.0,
                                    train_frac= .7,
                                    transform=transform_composed)
    quit()
    dataset_train = GreenRoofsDataset(root_dir="./data/dataset",
                                    mode='train',
                                    data_frac=1.0,
                                    train_frac= 0.7,
                                    transform=transform_composed)
    dataset_val = GreenRoofsDataset(root_dir="./data/dataset",
                                    mode='val',
                                    data_frac=1.0,
                                    train_frac= 0.7,
                                    transform=transform_composed)
    
    egids = {"train":[], 'train_no_oa': [], 'val':[]}
    vals = {"train": [], 'train_no_oa': [],  'val': []}
    for x in dataset_train:
        egids['train'].append(str(x['label'][1]))
        vals['train'].append(x['label'][0])
        if str(x['label'][1]).split('_') == 1:
            egids['train_no_oa'].append(str(x['label'][1]))
            vals['train_no_oa'].append(str(x['label'][0]))
    for x in dataset_val:
        egids['val'].append(str(x['label'][1]))
        vals['val'].append(x['label'][0])
    pd.DataFrame({'egid': egids['train']}).to_csv('./test/test_egids_in_trainingset.csv', sep=';')
    num_overlapp = len([x for x in egids['train'] if x in egids['val']])
    print(f"Number of overlapping samples: {num_overlapp}")
    print(f"Number of training samples: {len(egids['train'])}")
    print(f"Number of training samples without data augmentation: {len(list(set(egids['train'])))}")
    print(f"Number of validation samples: {len(egids['val'])}")
    print("\nPer class count")
    for i in range(6):
        print('label ' + str(i))
        print(f"\t on training set: {vals['train'].count(i)}")
        print(f"\t on validation set: {vals['val'].count(i)}")
    print()
    quit()
    """#egids = []
    print(len(dataset_test))
    for el in dataset_test:
        egids['val'].append(el['label'][1])
    print(len(dataset_train))
    for el in dataset_train:
        egids['train'].append(el['label'][1])
    pd.DataFrame(egids['train'],columns=['egid']).to_csv('./data/test/test_train.csv',sep=';', index=None)
    pd.DataFrame(egids['val'],columns=['egid']).to_csv('./data/test/test_val.csv',sep=';', index=None)
"""
    count = np.zeros((6))
    print("___FULL___")
    for el in dataset_full:
        count[el['label'][0]] += 1
    print(count)
    count = np.zeros((6))
    print("___TRAIN___")
    for el in dataset_train:
        count[el['label'][0]] += 1
    print(count)
    count = np.zeros((6))
    print("___TEST___")
    for el in dataset_val:
        count[el['label'][0]] += 1
    print(count)

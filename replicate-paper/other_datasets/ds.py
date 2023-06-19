import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CxVAE_Dset(Dataset):
    def __init__(self, csv_file, root_dir='./', tfm=None):
        self.csv_file = pd.read_csv(csv_file)
        self.list_imgs = list(self.csv_file['Index'])
        self.list_lbls = list(self.csv_file['B_Lable'])
        self.root_dir = root_dir
        self.tfm = tfm
    
    def __len__(self):
        return len(self.list_imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.root_dir+self.list_imgs[idx])
        img = img.convert("L")
        img = np.array(img)
        
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        img = torch.from_numpy(np.dstack([img, img, img])).transpose(2,0).transpose(1,2)
        img = (img - img.min())/(img.max() - img.min())
#         img = img.unsqueeze(0)
        lbl = torch.tensor(self.list_lbls[idx])
        ##
        if self.tfm is not None:
            img = self.tfm(img)
        # print(img.shape, lbl)
        return img, lbl
    
class CxVAE_retino_Dset(Dataset):
    def __init__(self, csv_file, root_dir='./', tfm=None):
        self.csv_file = pd.read_csv(csv_file)
        self.list_imgs = list(self.csv_file['Index'])
        self.list_lbls = list(self.csv_file['B_Lable'])
        self.root_dir = root_dir
        self.tfm = tfm
    
    def __len__(self):
        return len(self.list_imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.root_dir+self.list_imgs[idx])
#         img = img.convert("L")
        img = np.array(img)
        
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        img = torch.from_numpy(img).transpose(2,0).transpose(1,2)
        img = (img - img.min())/(img.max() - img.min())
#         img = img.unsqueeze(0)
        lbl = torch.tensor(self.list_lbls[idx])
        ##
        if self.tfm is not None:
            img = self.tfm(img)
        # print(img.shape, lbl)
        return img, lbl
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

from PIL import Image
import seaborn as sns
import numpy as np
import pickle as pkl
import os


def normalize_image(img):
    return (img-img.min())/(img.max()-img.min())

def train(model, 
          optimizer,
          trainloader,
          testloader,
          device,
          n_epoch=25,
          eps=0.05):
    ''' train a model with cross entropy loss
    '''
    ce_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epoch):
        # train
        model.train()
        train_loss = 0
        train_zero_one_loss = 0
        for img, label in tqdm(trainloader):
            img, label = img.to(device), label.to(device)
            pred = model(img)
            optimizer.zero_grad()
            loss = ce_loss(pred, label)
            loss.backward()
            train_loss += loss.item()  
            train_zero_one_loss += (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()
            optimizer.step()
        # test error
        model.eval()
        test_zero_one_loss = 0
        with torch.no_grad():
            for img, label in tqdm(testloader):
                img, label = img.to(device), label.to(device)
                pred = model(img)
                test_zero_one_loss += (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()   
        print('Train Error: ', train_zero_one_loss/len(trainloader.dataset))
        print('Test Error: ', test_zero_one_loss/len(testloader.dataset))
        if test_zero_one_loss/len(testloader.dataset) < eps:
            break
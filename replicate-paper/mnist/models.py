import torch.nn as nn
from torch import optim
import torchvision.models as models


# VGG variant
def vgg_model_and_optimizer(device="cuda"):
    model = nn.Sequential(
        nn.Conv2d(1, 64, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, 1, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16384, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer


# LeNet variant
from MNISTModel import MNIST32Model


def lenet_model_and_optimizer(lr=1e-4):
    model = MNIST32Model()
    optimizer = optim.Adam(model.parameters(), lr)
    return model, optimizer


# simplenet
from Simplenet import SimplenetV1


def simplenet_model_and_optimizer(device="cuda"):
    model = SimplenetV1()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer


# MLP
def mlp_model_and_optimizer(device="cuda"):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 10),
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer


# Resnet18
def resnet18_and_optimizer(lr=1e-2):
    resnet18 = models.resnet18()
    resnet18.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )  # single input channel
    resnet18.fc = nn.Linear(512, 10)  # 10-class problem
    optimizer = optim.Adam(resnet18.parameters(), lr=lr)
    return resnet18, optimizer


if __name__ == "__main__":
    print("test passed")

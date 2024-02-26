"""
SimplerNetV1 in Pytorch.
The implementation is basded on : 
https://github.com/D-X-Y/ResNeXt-DenseNet
code initially taken from
https://github.com/Coderx7/SimpleNet_Pytorch/blob/master/models/simplenet.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SimplenetV1(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, simpnet_name="simplenet"):
        super(SimplenetV1, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        )

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

        self.classifier = nn.Linear(256, num_classes)
        self.drp = nn.Dropout(0.1)

    def forward(self, x):
        out = self.model(x)

        # Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = self.drp(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

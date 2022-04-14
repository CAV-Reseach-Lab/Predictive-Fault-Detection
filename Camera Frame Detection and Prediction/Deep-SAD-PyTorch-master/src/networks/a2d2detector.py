import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class DetectorNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.conv1 = nn.Conv2d(128, 256, 3, padding=2, bias=False)
        self.pool = nn.MaxPool2d(19, 1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=2, bias=False)
        self.pool2 = nn.MaxPool2d(5, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, self.rep_dim, bias=False)

    def forward(self, x):
        #x = x.view(-1, 128, 19, 30)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        print(x.shape)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        #x = x.view(int(x.size(0)), -1)
        print(x.shape)
        x = self.fc1(x)
        return x

class DetectorNetDecoder(BaseNet):

    def __init__(self):
        super().__init__()

        #not sure what layers I should put here.
    def forward(self, x):
        #put the layers in.
        return x

class DetectorNetAutoEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.encoder = DetectorNet()
        self.decoder = DetectorNetDecoder()

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x
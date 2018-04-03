import torch
import torch.nn as nn
import math
import pdb
from collections import OrderedDict


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class IdsiaNet(nn.Module):
    def __init__(self, num_classes=10):
        super(IdsiaNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 100, kernel_size=7, padding=0)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=4, padding=0)

        self.fc1 = nn.Linear(3*3*250, 300)
        self.fc2 = nn.Linear(300, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_once(self, x):
        x = self.conv1(x)   # 100x42x42
        x = self.relu(x)       # 100x42x42
        x1 = self.pool(x)   # 100x21x21

        x = self.conv2(x1)  # 150x18x18
        x = self.relu(x)       # 150x18x18
        x2 = self.pool(x)   # 150x9x9

        x = self.conv3(x2)  # 250x6x6
        x = self.relu(x)       # 250x6x6
        x3 = self.pool(x)   # 250x3x3

        xv = x3.view(-1, 3*3*250)
        xfc1 = self.relu(self.fc1(xv))
        output = self.fc2(xfc1)
        
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
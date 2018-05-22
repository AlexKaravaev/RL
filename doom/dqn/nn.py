
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# arch: {conv -> conv -> conv -> flatten -> fc -> fc -> q-values for actions}
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # 1 layer is conv with filters = 32, kernel = 8, strides = 2, padding = valid
        # :input 84x84x4
        # :output 20x20x32
        self.conv1 = nn.Conv2d(4, 32, stride = 2, kernel_size = 8)
        self.conv1_bn = nn.BatchNorm2d(32)

        # 2 layer is conv with filters = 64, kernel = 4, strides = 2, padding = valid
        # :input 20x20x32
        # :output 9x9x64
        self.conv2 = nn.Conv2d(32, 64, stride = 2, kernel_size = 4)
        self.conv2_bn = nn.BatchNorm2d(64)

        # 3 layer is conv with filters = 128, kernel = 4, strides = 2, padding = valid
        # :input 9x9x64
        # :output 3x3x128
        self.conv3 = nn.Conv2d(64, 128, stride = 2, kernel_size = 4)
        self.conv3_bn = nn.BatchNorm2d(128)

        # flatten layer
        # :input 3x3x128
        # :output 1152
        self.flatten = Flatten()

        # fully connected:
        # :input 1152
        # :output 512
        self.fc1 =  torch.nn.Linear(1152, 512)

        # fully connected: units = 3, input = 512
        # :input 512
        # :output 3
        self.fc2 = torch.nn.Linear(512, 3)

    def forward(self, x):
        # 3 convolutions with elu activation func
        x = F.elu(self.conv1_bn(self.conv1(x)))
        x = F.elu(self.conv2_bn(self.conv2(x)))
        x = F.elu(self.conv3_bn(self.conv3(x)))

        x = self.flatten(x)

        x = F.elu(self.fc1(x))

        x = self.fc2(x)

        return x



if __name__ == '__main__':
    print("|Testing|\n")
    net = CNN()
    print(net)

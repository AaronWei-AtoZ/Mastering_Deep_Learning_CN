"""
为单通道 28x28 图像设计分类网络（无需训练），完成nn. Module :
1. Conv2d (1->6, kernel_size=3, padding=1) + ReLU
2. MaxPool2d (2,2)
3. Conv2d (6→12, kernel_size=3, padding=1) + ReLU
4. MaxPool2d (2,2)
5. 展平后 Linear（12*7*7 -> 4） 输出4类
要求在 forward 中用注释写出每一步张量形状变化。
"""

import torch
import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 12, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.output = nn.Linear(588, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # 形状变为[batch_size, 6, 28, 28]
        x = self.pool1(x)
        # 形状变为[batch_size, 6, 14, 14]
        x = torch.relu(self.conv2(x))
        # 形状变为[batch_size, 12, 14, 14]
        x = self.pool2(x)
        # 形状变为[batch_size, 12, 7, 7]

        x = torch.flatten(x, start_dim=1)
        # 形状变为[batch_size, 588]
        x = self.output(x)
        # 形状变为[batch_size, 4]

        return x


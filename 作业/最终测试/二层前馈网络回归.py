"""
题目描述 使用 PyTorch 实现一个简单的二层前馈网络完成回归任务：
1. 固定随机种子为 42；
2. 构建网络：
•隐藏层：Linear（3->5）+ReLU
• 输出层：Linear（5->1）
3. 随机生成 50 条样本 × 形状（50,3）），标签严格按
4.
Y= 4X1+X2-3X3，yER0xL.
y = 4×1 + X2 - 3x3， YE R50×1.
5. 使用
MSELOSS 与 SGD（Lr=0.01），演示一次“前向一反向一更新”流程 （batch size=50）；
6.更新后，打印隐藏层权重矩阵形状。
"""

import torch
import torch.nn as nn
from torch.nn import MSELoss

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(3, 5)
        self.output = nn.Linear(5,1)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        x = self.output(x)
        return x

def train():
    model = Model()
    x = torch.randn((50, 3))
    y = (4 * x[:, 0] + x[:, 1] - 3 * x[:, 2]).view(-1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'损失为:{loss.item()}')
    print(f'隐藏层权重矩阵形状:{model.Linear1.weight.shape}')

if __name__ == '__main__':
    model = Model()
    train()
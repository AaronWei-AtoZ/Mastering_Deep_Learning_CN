"""
卷积层提取局部特征后 -> 特征图（Feature Map）N = (W - F + 2P) // s + 1
每个卷积核都是一个神经元，会改变通道数量
池化只对宽高做处理，不会改变通道数

*** 卷积层参数计算公式 = 输入通道数 * 卷积核尺寸(h*w) * 卷积核数量 + 卷积核数量
"""
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from pathlib import Path
from torchsummary import  summary

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODEL_PATH = BASE_DIR / 'model' / 'image_model.pth'

# 每批次训练的样本数
BATCH_SIZE = 8

# 1.准备数据
def create_dataset():
                                                 # 数据预处理 -> 张量数据
    train_dataset = CIFAR10(DATA_DIR, train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(DATA_DIR, train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset

# 2.搭建卷积神经网络
class ImageModel(nn.Module):
    # 初始化父类成员，搭建神经网络
    def __init__(self):
        # 初始化父类成员
        super().__init__()
        # 搭建神经网络
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        # 全连接层
        self.fc1 = nn.Linear(576, 120)     # ps 由于全连接只能处理二维数据，所以待会儿需要将特征图从三维转为二维
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)    # 因为最后只有10个label，所以最终输出为10

    def forward(self, x):
        # 第一块：卷积（加权求和） + 激活函数（激励层） + 池化（降维）
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # 第二块：卷积（加权求和） + 激活函数（激励层） + 池化（降维）
        x = self.pool2(torch.relu(self.conv2(x)))

        # 第三块：全连接 + 激活函数
        # 先将三维转二维 -> 数据展开处理 (8, 16, 6, 6) -> (8, 576)
        x = x.reshape(x.size(0), -1)    # 参1:样本数（行数），参2:列数（特征数），-1表示能转多少列转多少列，最少1列

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.output(x)          # 这里不需要激励层，因为后面用CrossEntropyLoss,会自动进行softmax计算

# 3.模型训练
def train(train_dataset):
    # 创建数据加载器
    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 创建模型对象
    model = ImageModel()
    # 创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 循环遍历epoch
    epochs = 20
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x, y in data_loader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            # print(torch.argmax(y_pred, dim=-1))     # 返回每个样本预测值中最大的分类，比如样本有十个预测值(因为最后是10分类)，然后比较得出最大的预测值，将该值对应到分类索引输出
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)        # 第一批平均损失 * 第一批样本个数 = 第一批总损失
            total_samples += len(y)

        print(f'epoch:{epoch + 1}, loss:{total_loss / total_samples}, acc:{total_correct / total_samples:.2f}, time:{time.time() -start:.2f}s')

    # 保存模型
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

# 4.模型测试
def evaluate(test_dataset):
    # 创建测试集 数据加载器
    data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 创建模型对象
    model = ImageModel()
    # 加载模型参数
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'未找到模型权重：{MODEL_PATH}。请先调用 train() 完成训练。')
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    # 统计预测正确的样本个数、总样本个数
    total_correct, total_samples = 0, 0

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            y_pred = model(x)
            # 因为训练的时候用了CrossEntropyLoss,所以搭建神经网络时没有加softmax()激活函数，所以这里用argmax()获取预测类别
            y_pred = torch.argmax(y_pred, dim=1)
            total_correct += (y_pred == y).sum().item()
            total_samples += len(y)

    print(f'acc:{total_correct / total_samples:.2f}')




if __name__ == '__main__':
    # 1.获取数据集
    train_dataset, test_dataset = create_dataset()
    # print(f'训练集:{train_dataset}')                 # 这个看到是总览
    # print(f'训练集:{train_dataset.data.shape}')        # 训练集:(50000, 32, 32, 3)
    # print(f'测试集:{test_dataset.data.shape}')         # 测试集:(10000, 32, 32, 3)
    # print(f'数据集类别:{train_dataset.class_to_idx}')   # 数据集类别:{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    # 查看数据集中的图像
    # plt.figure(figsize=(30, 30))
    # plt.imshow(train_dataset.data[613])               # 获取索引为 613 的图像，即第614张图
    # plt.title(train_dataset.data[613])                # 获取索引为 613 的图像的标签
    # plt.show()


    # 2.搭建神经网络
    model = ImageModel()
    # 查看模型参数
    # summary(model,(3, 32, 32), batch_size=BATCH_SIZE)


    # 3.模型训练（按需取消下一行注释）
    # train(train_dataset)

    # 4.模型评估：仓库不包含权重，请先完成训练
    if MODEL_PATH.exists():
        evaluate(test_dataset)
    else:
        print('未检测到模型权重。请先取消 train(train_dataset) 的注释并运行训练。')

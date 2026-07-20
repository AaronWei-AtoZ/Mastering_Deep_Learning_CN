"""
需求：小明创办了一家手机公司，他不知道如何估算手机产品的价格。
为了解决这个问题，他收集了多家公司的手机销售数据。
该数据为二手手机的各个性能的数据，最后根据这些性能得到4个价格区间，作为这些二手手机售出的价格区间。
主要包括：
battery-power：电池一次可储存的总能量，单位为毫安时
blue：是否有蓝牙
clock_speed：微处理器执行指令的速度
dual_sim：是否支持双卡
fc：前置摄像头百万像素
four_g：是否有4G
int_memory：内存(GB)
m_dep：移动深度(cm)
mobile_wt：手机重量
n_cores：处理器内核数
pc：主摄像头百万像素
px_height：像素分辦率高度
px_width：像素分辦率宽度
ram：随机存取存储器（兆字节）
sc_h：手机屏幕高度（cm）
SC_W： 手机屏幕宽度（cm）
talk_time：一次电池充电持续时间最长的时间three_g：是杏有3G
touch_screen：是否有触控屏
wifi：是否能连wifi
price_range：价格区间（0, 1, 2, 3）此处的区间指如500~1000为0，1001~1500为1这样的范围

要求：帮助小明找出手机的功能（例如：RAM等）与其售价之间的某种关系。可以使用机器学习的方法来解决这个问题，也可以构建一个全连接的网络。
esp：这里只需要找出价格区间，所以可以看作一个分类问题而非精准的预测问题。

深度学习建模步骤：
1.准备训练集数据
2.构建神经网络
3.模型训练
4.模型预测评估


调优思路：
    1.优化方法从SGD -> Adam
    2.学习率从0.001 -> 0.01
    3.对数据进行标准化
    4.增加网络的深度，神经元个数
    5.调整训练的轮数
"""
import torch                                            # pytorch框架，封装了张量的各种操作
from torch.utils.data import TensorDataset              # 数据集对象 对数据的操作为：数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader                 # 数据加载器
import torch.nn as nn                                   # 封装了神经网络的操作
import torch.optim as optim                             # 优化器
from sklearn.model_selection import train_test_split    # 划分测试集和训练集
import matplotlib.pyplot as plt                         # 画图
import numpy as np
import pandas as pd
import time
from pathlib import Path
from torchsummary import summary                        # 模型结构可视化

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / '手机价格预测.csv'
MODEL_PATH = BASE_DIR / 'model' / 'phone.pth'


# todo 1.准备训练集数据
def create_dataset():
    # 1.加载数据集
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'未找到数据文件：{DATA_PATH}。请根据同目录 DATA_SOURCE.md 的说明获取数据。'
        )
    data = pd.read_csv(DATA_PATH)    # data.shape -> (2000,21) 这里的数据有整数有小数，要做类型转换方便后续微分操作
    # 2.获取特征和标签
    x, y = data.iloc[:, :-1], data.iloc[:,-1]
    # 3.将特征转换为浮点型 符合后续计算数值要求
    x = x.astype(np.float32)
    # 4.切分训练集和测试集
    # stratify:样本的分布，即参考y的类别抽取数据
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # 5.将数据集封装为张量数据集  数据 -> Tensor -> 数据集 -> 数据加载器
    train_dataset = TensorDataset(
        torch.tensor(x_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long),
    )

    # 6.返回结果                         20(充当 输入特征数)  去重 4 (充当输出标签数)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# todo 2.构建神经网络
"""
构建全连接神经网络来进行手机价格分类，该网络主要由三个线性层来构建，使用relu激活函数。
网络共有 3 个全连接层, 具体信息如下:
第一层: 输入为维度为 20, 输出维度为: 128
第二层: 输入为维度为 128, 输出维度为: 256
第三层: 输入为维度为 256, 输出维度为: 4
"""
class PhonePriceModel(nn.Module):
    # 1.在init魔法方法中，初始化父类成员，搭建神经网络
    def __init__(self, input_dim, output_dim):
        # 1.1 初始化父类成员
        super().__init__()
        # 1.2 搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.output = nn.Linear(256, output_dim)

    # 2. 定义前向传播方法 forward()
    def forward(self, x):
        # 2.1 隐藏层1处理数据：加权求和 + 激活函数relu
        x = torch.relu(self.linear1(x))
        # 2.2 隐藏层2处理数据：加权求和 + 激活函数relu
        x = torch.relu(self.linear2(x))
        # 2.3 输出层：加权求和 + 激活函数softmax -> 这里只需要加权求和，后面用CrossEntropyLoss()替代softmax CrossEntropyLoss() = softmax() + 损失计算
        x = self.output(x)
        # 2.4 返回处理结果
        return x


# todo 3.模型训练
def train(train_dataset, input_dim,output_dim):
    # 1.创建数据加载器
                            # 数据集对象 1600条 每批的数据条数  是否打乱数据
    train_loader = DataLoader(train_dataset,batch_size=16, shuffle=True)
    # 2.创建神经网络模型对象
    model = PhonePriceModel(input_dim, output_dim)
    # 3.定义损失函数，因为是多分类，采用多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 4.创建优化器对象
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 5.模型训练
    # 5.1 定义变量，记录训练的总轮数
    epochs = 50
    # 5.2 开始每轮的训练
    for epoch in range(epochs):
        # 5.2.1 定义变量，记录每次训练的损失值、训练的批次数
        total_loss, batch_num = 0.0, 0
        # 5.2.2 记录开始训练的时间
        start = time.time()
        # 5.2.3 开始本轮的训练
        for x, y in train_loader:
            # 5.2.4 切换模型状态
            model.train()   # 训练模式
            # 5.2.5 模型预测
            y_pred = model(x)
            # 5.2.6 计算损失
            loss = criterion(y_pred, y)
            # 5.2.7 梯度清零、反向传播、优化参数
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            # 5.2.8 累加损失值
            total_loss += loss.item()   # 将本轮的每批次（16条）的平均损失累加起来
            batch_num += 1
        # 本轮训练结束，打印训练信息
        print(f'轮数:{epoch + 1}, loss:{total_loss / batch_num:.4f}, time:{time.time() - start:.2f}s')

    # 6. 多轮训练结束，保存模型参数
    # print(f'\n\n模型的参数信息是：{model.state_dict()}')
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

# todo 4.模型预测评估
def evaluation(test_dataset, input_dim, output_dim):
    # 1.创建神经网络分类对象
    model = PhonePriceModel(input_dim, output_dim)
    # 2.加载模型参数
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'未找到模型权重：{MODEL_PATH}。请先运行训练。')
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    # 3.创建测试集的数据加载器对象
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # 4.记录预测正确的样本数
    correct = 0
    # 5.从数据加载器中获取每批数据
    model.eval()    # 测试模式
    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            # 用argmax()获取最大值对应的下标，就是类别。
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()

    # 6. 准确率
    print(f'准确率:{correct / len(test_dataset):.4f}')


if __name__ == '__main__':
    # 准备数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # 构建模型
    model = PhonePriceModel(input_dim, output_dim)
    # 计算模型参数
    # 参一：模型对象；参二：输入数据的形状（批次大小，输入特征数）
    summary(model, input_size=(input_dim,))
    # 模型训练
    train(train_dataset, input_dim, output_dim)

    # 模型预测评估
    evaluation(test_dataset, input_dim, output_dim)

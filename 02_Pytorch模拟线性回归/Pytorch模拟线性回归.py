import torch
from torch.utils.data import TensorDataset, DataLoader  # 构造数据集对象、数据加载器
from torch import nn                                    # nn模块中有平方损失函数和假设函数
from torch import optim                                 # optim模块中有优化器函数
from sklearn.datasets import make_regression            # 创建线性回归模型数据集
import matplotlib.pyplot as plt

# 转换关系！！！ numpy对象 -> 张量Tensor -> 数据集对象TensorDataset -> 数据加载器DataLoader

# 1. 定义函数，创建线性回归样本数据
def create_dataset():
    x, y, coef = make_regression(
        n_samples=100,  # 100个样本点
        n_features=1,   # 1个特征
        noise=10,       # 噪声，噪声越大样本点约散
        coef=True,      # 权重
        bias=10,        # 偏置
        random_state=42
    )
    # print(type(x))      # <class 'numpy.ndarray'>
    # 在这里将x, y封装成张量对象
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)  # y怎么来的初始值？ -> y = wx +b

    return x, y, coef

# 2. 定义函数，表示模型训练
def model_train(x, y, coef):
    # 2.1 创建数据集对象，把tensor -> 数据集对象 -> 数据加载器
    dataset = TensorDataset(x, y)

    # 2.2 创建数据加载器对象       数据集对象   批次大小   是否随机打乱（训练集打乱，测试集不打乱）
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2.3 创建初始的线性回归模型
    model = nn.Linear(1, 1)

    # 2.4 创建损失函数对象
    criterion = nn.MSELoss()

    # 2.5 创建优化器对象       模型参数            学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 2.6 具体的训练过程
    # 2.6.1 定义变量，分别表示：训练轮数，每轮的平均损失，训练总损失，训练样本数
    epochs, loss_list = 100, []
    # 2.6.2 开始训练，按轮训练
    for epoch in range(epochs):
        total_loss, total_batches = 0.0, 0
        # 2.6.3 因为每轮 都是按批次训练， 所以从数据加载器中 获取 批次数据
        for train_x, train_y in dataloader: # 7批(16, 16, 16, 16, 16, 16, 4)
            # 2.6.4 模型的预测
            y_pred = model(train_x)
            # 2.6.5 计算损失
            loss = criterion(y_pred, train_y.reshape(-1,1)) # 因为真实值的形状是一行n列，要转换成n行1列。现在的y_pred是n行1列的。这样才能计算
            # 2.6.6 梯度计算总损失 和 样本批次树
            total_loss += loss.item()
            total_batches += 1
            # 2.6.7 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 2.6.8 把本轮的平均损失值添加到列表中
        average_loss = total_loss / total_batches
        loss_list.append(average_loss)
        print(f'轮数：{epoch + 1}, 平均损失值:{average_loss}')

    # 2.7 打印最终的训练结果
    print(f'{epochs}轮的平均损失分别为：{loss_list}')
    print(f'模型参数，权重:{model.weight}, 偏置:{model.bias}')

    # 2.8 绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title('loss line')
    plt.grid()
    plt.show()

    # 3. 绘制真实值和预测值的关系
    # 3.1 绘制样本点分布情况
    plt.scatter(x, y)
    # 3.2 绘制训练模型的预测值
    with torch.no_grad():
        y_pred = model(x).squeeze(1)
        y_true = x.squeeze(1) * coef + 10

    sorted_x, indices = torch.sort(x.squeeze(1))

    # 3.3 绘制预测值和真实值的折线图
    plt.plot(sorted_x, y_pred[indices], color='red', label='est value')
    plt.plot(sorted_x, y_true[indices], color='blue', label='true value')

    # 3.4 图例
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    x, y, coef = create_dataset()
    model_train(x, y, coef)

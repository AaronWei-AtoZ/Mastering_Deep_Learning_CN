# Mastering Deep Learning with PyTorch｜中文学习记录

这是我的 PyTorch 与深度学习课程学习仓库，记录从张量操作、线性回归，到 ANN、CNN 和基础 RNN 的代码实践。

> This repository documents my course-based PyTorch learning and implementation practice. It is a learning record rather than an original research project or a production-ready framework.

## 当前内容

| 模块 | 内容 | 代表代码 |
| --- | --- | --- |
| [01_Pytorch_Study](./01_Pytorch_Study) | 张量创建、数据类型、索引与维度操作 | [张量创建](./01_Pytorch_Study/Tensor_张量的基础创建和元素类型.py) · [张量基本操作](./01_Pytorch_Study/Tensor_张量的基本操作.py) |
| [02_Pytorch模拟线性回归](./02_Pytorch模拟线性回归) | `TensorDataset`、`DataLoader`、`nn.Linear`、MSE 与 SGD | [线性回归](./02_Pytorch模拟线性回归/Pytorch模拟线性回归.py) |
| [03_ANN全连接神经网络](./03_ANN全连接神经网络) | 网络搭建、参数初始化、Dropout、BatchNorm、手机价格四分类 | [手机价格分类](./03_ANN全连接神经网络/05_ANN手机价格预测case/手机价格分类案例.py) |
| [04_CNN卷积神经网络](./04_CNN卷积神经网络) | 卷积层、池化层、CIFAR-10 图像分类 | [CIFAR-10 分类](./04_CNN卷积神经网络/03_图像分类识别case/图像分类识别.py) |
| [05_RNN循环神经网络](./05_RNN循环神经网络) | 词嵌入、`nn.RNN` 与基础文本生成流程 | [歌词生成练习](./05_RNN循环神经网络/03_歌词生成case.py) |
| [作业/最终测试](./作业/最终测试) | Autograd、张量连续性、前馈网络、CNN 与 RNN 小练习 | 补充练习记录 |

未完成、尚未复核或不适合公开的文件没有纳入仓库。

## 代表性实践

### 1. PyTorch 线性回归

- 使用 `scikit-learn` 生成带噪声的回归数据；
- 完成 NumPy → Tensor → Dataset → DataLoader 的数据流程；
- 使用单层线性模型、均方误差损失和 SGD 训练；
- 可视化每轮损失及拟合结果。

### 2. ANN 手机价格区间分类

- 将 20 个手机配置特征映射到 4 个价格区间；
- 分层划分训练集与测试集；
- 构建 `20 → 128 → 256 → 4` 的全连接网络；
- 使用交叉熵损失、SGD、权重保存与测试集准确率评估。

### 3. CNN CIFAR-10 图像分类

- 使用两组卷积、ReLU 与最大池化层提取图像特征；
- 使用三层全连接网络完成 10 类分类；
- 使用交叉熵损失与 Adam；
- 包含训练、权重保存和测试集评估流程。

### 4. RNN 文本生成练习

- 使用 Jieba 分词并建立词表；
- 使用 `Dataset`/`DataLoader` 构造移位序列；
- 使用 Embedding、`nn.RNN` 和线性输出层预测下一个词；
- 语料和训练权重不随仓库分发。

## 思维导图

![深度学习思维导图](./images/深度学习思维导图_第一版.png)

## 环境

建议使用 Python 3.10 或更高版本：

```bash
pip install -r requirements.txt
```

不同案例使用相对独立的脚本。进入相应目录后运行，例如：

```bash
cd 02_Pytorch模拟线性回归
python Pytorch模拟线性回归.py
```

## 数据来源与仓库策略

| 案例 | 数据来源 | 仓库是否包含 |
| --- | --- | --- |
| 手机价格分类 | Kaggle [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) | 否。原页面许可证标记为 `Unknown`，请自行获取并阅读案例目录中的说明 |
| CIFAR-10 | [CIFAR-10 官方页面](https://cave.cs.toronto.edu/kriz/cifar.html)；代码通过 `torchvision` 下载 | 否。运行代码时下载到本地 |
| RNN 文本生成 | 课程配套的周杰伦歌词语料 | 否。歌词文本不在仓库重新分发；请换用自有或获授权语料 |

模型权重、原始数据、IDE 配置和运行缓存均不纳入版本控制。更详细的说明见 [DATA_SOURCES.md](./DATA_SOURCES.md)。

## 校验范围

- 已对公开脚本和 notebook 代码单元进行静态语法检查；
- 已修正明显的路径、损失记录、评估模式和批次处理问题；
- 没有重新训练模型，因此仓库不声称新的准确率或 benchmark 结果。

## 仓库定位

本仓库用于保存可验证的学习轨迹，重点展示我对 PyTorch 数据流程、训练循环、基础神经网络结构和模型评估的理解。代码主要来自课程练习与个人复现，不应被视为原创研究成果或成熟的软件工程项目。


# 数据来源与文件说明

本仓库只保存学习代码，不重新分发大型数据、来源授权不明确的数据、歌词文本或训练权重。

## 手机价格分类

- 来源：[Kaggle — Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- 原数据页面目前将许可证标记为 `Unknown`，因此 CSV 不随仓库上传。
- 下载训练数据后，将包含 `price_range` 标签的文件改名为 `手机价格预测.csv`，放到：
  `03_ANN全连接神经网络/05_ANN手机价格预测case/`。

## CIFAR-10

- 来源：[CIFAR-10 and CIFAR-100 datasets](https://cave.cs.toronto.edu/kriz/cifar.html)
- 代码使用 `torchvision.datasets.CIFAR10(..., download=True)` 自动下载。
- 本地数据会出现在 CNN 案例的 `data/` 目录，该目录已被 Git 忽略。

## RNN 文本生成

- 原练习使用课程配套的 `jaychou_lyrics.txt` 周杰伦歌词集合。
- 因歌词文本涉及版权，原始语料不随仓库上传。
- 请使用自有或获授权的 UTF-8 中文文本，并命名为 `jaychou_lyrics.txt`，放到：
  `05_RNN循环神经网络/data/`。

## 模型权重

`*.pth`、`*.pt`、`*.ckpt` 和 `model/` 均属于训练生成物，不纳入版本控制。完成本地训练后，脚本会自动创建相应目录并保存权重。


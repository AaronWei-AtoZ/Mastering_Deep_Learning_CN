"""
RNN层 作用：
    基于 上一次的隐藏状态 + 本次的输入 -> 本次的隐藏状态，本次的输出

本次的隐藏状态 = tanh(上次的隐藏状态加权求和 + 本次的输入 加权求和)
本次的输出 = 本次的隐藏状态加权求和 并选择概率最大的定位到词汇表，作为输出词汇
"""

import torch
import torch.nn as nn

# 创建RNN层
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)

# 定义变量，表示输入的x
# 参1:每个句子中词的个数（句子的长度）
# 参2:句子的数量 -> batch的大小
# 参3:词向量的维度
x = torch.randn(size=(5, 32, 128))

# 创建上一时刻的隐藏状态
# 参1:隐藏层的层数
# 参2:句子的数量
# 参3:隐藏状态向量维度
h0 = torch.randn(size=(1, 32, 256))

# 调用RNN处理，获取当前时刻的预测值 和 当前的隐藏状态
output, h1 = rnn(x, h0)
print(f'output:{output}, h1:{h1}')
print(f'output:{output.shape}, h1:{h1.shape}')

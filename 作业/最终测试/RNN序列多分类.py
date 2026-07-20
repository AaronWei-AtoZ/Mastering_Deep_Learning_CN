"""
请基于原始 nn.RNN 完成一个序列多分类模型：
1. 数据准备
• torch-manual_seed (42) ;
•随机生成输入 X，形状（16,10,12）；
• 随机生成标签 y，形状（16，），取值 0-4；
2. 模型结构
1. RNN: input_size=12, hidden_size=20, nonlinearity='tanh', batch_first=True ;
2. 取最后隐藏态 h_n 一（16,20），通过：
• Linear (20→10) + ReLU
• Dropout (p=0.5)
• Linear（10-5）+ Softmax（dim=1）一输出（16,5）；
3.参数初始化
• 对两层全连接权重使用 Xavier 正态初始化、偏置置零；
"""

import torch
import torch.nn as nn
import torch.nn.init as init

torch.manual_seed(42)


class rnn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=12,          # 输入的维度，要和每个词的表示维度一致
            hidden_size=20,         # 用来计算的神经元个数，不管输入是多少维的向量，最后都会压缩为 1个20维的向量
            nonlinearity='tanh',
            batch_first=True        # 用来规定，传入的张量的第0维就是批次大小，因为RNN默认会将批次放到第2个位置 (10, 16, 12)
        )
        self.fc1 = nn.Linear(in_features=20, out_features=10)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(in_features=10, out_features=5)

        self._init_weights()

    def forward(self, x):
        out, h_n = self.rnn(x)
        h_n = h_n.squeeze(0)

        x = torch.relu(self.fc1(h_n))
        x = self.dropout(x)
        x = torch.softmax(self.output(x), dim=1)    # 由于softmax之后的产出为 (batch_size, 5(最后线性变换维度))，按照dim=1就是横着计算每个句子在5个分类之中的概率占比

        return x

    def _init_weights(self):
        # fc1 层的权重和偏置
        init.xavier_normal_(self.fc1.weight)
        init.zeros_(self.fc1.bias)

        # fc2 层的权重和偏置
        init.xavier_normal_(self.output.weight)
        init.zeros_(self.output.bias)


if __name__ == '__main__':
   model = rnn_model()
   print(model)
   x = torch.randn(16, 10, 12)  # 16个句子 每个句子10个词 每个词由12维的向量表示 ｜ 这10个词就是代表了时间步，处理1个词会有大神经元里的20个小神经元一起处理，一次性会处理所有的词，那么就是有10个步骤
   y = torch.randint(0, 5, (16,))  # 标签必须是整数，(0,5)包左不包右，表示取值范围
   # print(f'x:{x}')
   # print(f'x:{x.shape}')
   # print(f'y:{y}')
   y_pred = model(x)
   print('-' * 50)

   print(f"输入 X 的形状: {x.shape}")
   print(f"输出 Predictions 的形状: {y_pred.shape}")

   # 验证 Softmax 概率逻辑 (取第0个样本，查看其5个概率之和是否约为 1.0)
   sample_0_prob_sum = torch.sum(y_pred[0]).item()
   print(f"\n第 0 个样本的类别概率输出: {y_pred[0].detach().numpy()}")
   print(f"该样本概率总和: {sample_0_prob_sum:.4f}")


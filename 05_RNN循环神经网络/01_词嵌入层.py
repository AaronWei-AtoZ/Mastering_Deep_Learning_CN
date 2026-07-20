"""
RNN(循环神经网络)，主要用于处理序列数据
序列数据：后边的数据对前边的数据有依赖
RNN组成：
    词嵌入层
    循环网络层
    输出层
词嵌入层：
    将词/文字或对应的 索引转化为词向量
"""

import torch
import torch.nn as nn
import jieba

# 演示把词/词索引 转换为 词向量
def transform():
    text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
    # 使用jieba进行分词
    words = jieba.lcut(text)
    print(words)

    # 创建词嵌入层
    embed = nn.Embedding(num_embeddings=len(words), embedding_dim=4 )

    # 获取每个词对象的下标索引  enumerate()返回类表中每个值 及其索引
    for i, word in enumerate(words):
        print(i, word)
        # 把词索引 i(张量形式) 转为词向量
        word2vec = embed(torch.tensor(i))   # 全随机
        print(word2vec)


if __name__ == '__main__':
    transform()

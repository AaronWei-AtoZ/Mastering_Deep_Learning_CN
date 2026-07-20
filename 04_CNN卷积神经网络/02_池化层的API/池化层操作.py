"""
池化层的作用：降维
分类：最大池化、平均池化
特点：池化不会改变通道数
"""
import torch
import torch.nn as nn

# 单通道池化
def single_pool():
    # 1.创建一个 1通道的 3*3矩阵
    inputs = torch.tensor([[
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]])

    # 2.创建最大池化层
    pool_max = nn.MaxPool2d(2, 1, 0)
    outputs1 = pool_max(inputs)
    print(outputs1)

    # 3.创建平均池化
    pool_avg = nn.AvgPool2d(2, 1, 0)
    outputs2 = pool_avg(inputs)
    print(outputs2)


# 多通道池化
def mul_pool():
    # 1.创建1个多通道的 3*3矩阵
    inputs = torch.tensor([[
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ],
    [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ],
    [
        [99, 11, 22],
        [33, 44, 55],
        [66, 77, 88]
    ]
    ])

    # 2.创建最大池化层
    pool_max = nn.MaxPool2d(2, 1, 0)
    outputs1 = pool_max(inputs)
    print(outputs1)

    # 3.创建平均池化
    pool_avg = nn.AvgPool2d(2, 1, 0)
    outputs2 = pool_avg(inputs)
    print(outputs2)



if __name__ == '__main__':
    single_pool()
    mul_pool()
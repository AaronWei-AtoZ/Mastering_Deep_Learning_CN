"""
卷积层API: nn.Conv2d()
卷积层的组成：
    卷积层：用于提取图像的局部特征，结合 “卷积核”（神经元）实现，处理后的结果为：特征图
    池化层：用于降维
    全连接层(fc, linear, Full Connected)：用于预测结果，并输出结果
卷积层特征图维度计算方式：
    N = ((W-F+2*P) / S) + 1
    W:图像大小
    F：卷积核大小
    P：填充大小
    S：步长
"""

import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 1. 加载图像，实现卷积、特征图可视化操作
def dm01():
    image_path = BASE_DIR / 'img.jpg'
    if image_path.exists():
        img = torch.tensor(plt.imread(image_path), dtype=torch.float32)
    else:
        # 仓库不依赖外部示例图片；缺少图片时使用固定随机张量演示 API。
        torch.manual_seed(42)
        img = torch.rand(640, 640, 3)
    # 打印图像信息
    # print(f'img:{img}, shape:{img.shape}')    # (640, 640, 3)

    # 将图像的形状从HWC改为CHW 要将通道放前边: 现将img封装为张量，再转换维度
    img2 = img.permute(2, 0, 1)                 # 用于转换维度的方法
    # print(f'img2:{img2}, shape:{img2.shape}') # (3, 640, 640)

    # 因为这里只有一张图，所以再增加一个维度，从 CHW -> (1, C, H, W) 1张3通道的640*640像素的图
    img3 = img2.unsqueeze(dim=0)
    # print(f'img3:{img3}, shape:{img3.shape}')  # (1, 3, 640, 640)

    # 创建卷积层对象，提取特征图
    conv = nn.Conv2d(3, 4, 3, 1, 0)

    # 卷积计算
    conv_img = conv(img3)

    # 打印卷积后的结果
    # print(f'conv_img:{conv_img}, shape:{conv_img.shape}')  # ([1, 4, 638, 638])

    img4 = conv_img[0]                          # CHW (4, 638, 638)

    # 可视化第1个通道的特征图，现在有4个通道。要可视化需要将CHW -> HWC
    img5 = img4.permute(1, 2, 0)
    feature1 = img5[:, :, 0].detach().numpy()   # 第1个通道的(638, 638)像素图
    plt.imshow(feature1)
    plt.show()


if __name__ == '__main__':
    dm01()

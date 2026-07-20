"""
在Pytorch中，数据以张量的形式储存，有0、1、2、3...多维张量。
简单理解0、1、2、3维张量 分别对应 标量、向量、矩阵、张量
张量中只能是数值，可以通过系列操作进行类型更改，下面会讲到

在本python文件中会介绍张量的创建形式、全0/1/指定值张量的创建、线性和随机张量的创建、元素类型转换和设置

以定义函数的方式讲解每个要点

多敲代码 背熟函数即可！！！

张量的创建有三种基础方式：
    1.使用torch.tensor()  根据指定“数据”创建张量    -> 好处是可以根据传入的数值自动判断数值类型
    2.使用torch.Tensor()  根据指定“数据”创建张量 + 可根据“形状”(size = x,y)创建
    3.使用指定数值类型的方式创建，如：torch.IntTensor, torch.FloatTensor, torch.LongTensor, torch.DoubleTensor
    *** 推荐优先使用 torch.tensor(..., dtype=...)；类型专用构造器的 dtype 已由构造器决定。

全0、1、指定张量的创建方式：
    1.全0 -> torch.zeros(x, y) / torch.zeros_like(参数)
    2.全1 -> torch.ones(x, y) / torch.ones_like(参数)
    3.全指定 -> torch.full(size(x, y), fill_value) / torch.full_like(参数, fill_value)

线性和随机张量的创建：
    torch.rand(x, y)       # 均匀分布的值在0～1间的x行y列二维张量
    torch.randn(x, y)      # 标准正态分布的x行y列二维张量
    torch.randint(x, y, size=(a,b))    # 以[x,y)为区间的a行b列的二维张量


总结：
    1.创建张量有三种基础方式
    2.全0、1、指定张量均可以采用给形状、传入参数的方式创建张量
    3.可使用 torch.arange 和 torch.linspace 生成数值序列
    4.随机张量有三种创建方式
    5.设定元素类型的时候，可以一开始指定元素类型，或采用类型转换的方式转变类型
    6.元素类型转换可使用 .to(dtype=...)、.type()，或在创建时传入 dtype

"""

import torch    # 首先导包 pytorch的包就是torch
import numpy as np
torch.manual_seed(42)   # pytorch中人工设定随机种子的方式，建议每次在导包后进行。


# 三种基础创建方式
def define_tensor_by_torch_tensor():    # 演示 torch.tensor 创建张量
    # 1.演示：torch.tensor()
    # 1.1 直接赋值创建
    t1 = torch.tensor([1, 3, 3])                # 1维张量 -> 向量
    t2 = torch.tensor([[4, 5, 6], [7, 8, 9]])   # 2维张量 -> 2行3列的矩阵
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')

    # 1.2 传入值创建
    data = [[1,3,3],[4,5,6],[7,8,9]]
    t3 = torch.tensor(data)
    print(f'张量t3的元素值是:{t3}, t3的类型是:{type(t3)}, 张量t3中的元素值类型是:{t3.dtype}')

    # 1.3 numpy转换的方式创建
    data = np.random.randint(0, 20, size=(4, 5))    # 参数解释 参一：最低值，参二：最高值，参三：x * y的矩阵 此处是4*5的矩阵
    t4 = torch.tensor(data)
    print(f'张量t4的元素值是:{t4}, t4的类型是:{type(t4)}, 张量t4中的元素值类型是:{t4.dtype}')

    # 1.4 尝试以形状的方式创建 -> 会报错
    # t5 = torch.tensor(3, 4)
    # print(t5)   # TypeError: tensor() takes 1 positional argument but 2 were given

    # 1.5 演示指定元素类型
    t6 = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.float)    # 从 int64 变为了 float32
    print(f'张量t6的元素值是:{t6}, t6的类型是:{type(t6)}, 张量t6中的元素值类型是:{t6.dtype}')

    print('-' * 100) # 这是一条分割线哦～～～

def define_tensor_by_torch_Tensor():    # 演示 torch.Tensor 创建张量  基本和 torch.tensor一致，加可以直接以形状的方式创建
    # 1.演示：torch.Tensor()
    # 1.1 直接赋值创建
    t1 = torch.Tensor([1, 3, 3])               # 1维张量 -> 向量
    t2 = torch.Tensor([[4, 5, 6], [7, 8, 9]])  # 2维张量 -> 2行3列的数组
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')

    # 1.2 传入值创建
    data = [[1, 3, 3], [4, 5, 6], [7, 8, 9]]
    t3 = torch.Tensor(data)
    print(f'张量t3的元素值是:{t3}, t3的类型是:{type(t3)}, 张量t3中的元素值类型是:{t3.dtype}')

    # 1.3 numpy转换的方式创建
    data = np.random.randint(0, 20, size=(4, 5))  # 参数解释 参一：最低值，参二：最高值，参三：x * y的矩阵 此处是4*5的矩阵
    t4 = torch.Tensor(data)
    print(f'张量t4的元素值是:{t4}, t4的类型是:{type(t4)}, 张量t4中的元素值类型是:{t4.dtype}')

    # 1.4 尝试以形状的方式创建 -> 这里就可以啦～～
    t5 = torch.Tensor(3, 4)  # 按形状创建未初始化张量；若需要确定初值，应使用 zeros/ones/full
    print(f'张量t5的元素值是:{t5}, t5的类型是:{type(t5)}, 张量t5中的元素值类型是:{t5.dtype}')

    print('-' * 100)  # 这是一条分割线哦～～～

def define_tensor_by_given_types():
    # 1.使用指定数值类型的方式创建，如：torch.IntTensor, torch.FloatTensor, torch.LongTensor, torch.DoubleTensor
    # *** 这类创建方式不能指定类型
    # 用数值创建的时候和torch.tensor、torch.Tensor一样，
    # 1.1 此处用torch.IntTensor演示
    t1 = torch.IntTensor([1, 3, 3])
    t2 = torch.IntTensor([[4, 5, 6], [7, 8, 9]])
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')

    # 1.2 演示用“形状”创建
    # 参数 -> 个数、行、列
    t3 = torch.IntTensor(3, 4, 5)  # 可能会看到用0填充，但是这是“垃圾值”，如果想创建全0张量，务必使用torch.zeros() / torch.zeros_like()
    print(f'张量t3的元素值是:{t3}, t3的类型是:{type(t3)}, 张量t3中的元素值类型是:{t3.dtype}')

    print('-' * 100)


# 全0/1/指定值张量的创建
def all_zeros_tensor():     # 创建全0张量
    # 1. 采用给形状的方式创建全0张量
    t1 = torch.zeros(3, 4)  # -> 创建一个三行四列(size = 3, 4)的二维张量
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')

    # 2. 采用传参的方式创建全0张量
    data = torch.tensor([[1, 2, 3], [3, 4, 4]])
    t2 = torch.zeros_like(data)  # 根据传入参数的 size(x行，y列)创建同样行列数的张量 **和数值无关
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')

    print('-' * 100)

def all_ones_tensor():      # 创建全1张量
    # 1. 采用给形状的方式创建全1张量
    t1 = torch.ones(3, 4)  # -> 创建一个三行四列(size = 3, 4)的二维张量
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')

    # 2. 采用传参的方式创建全1张量
    data = torch.tensor([[1, 2, 3], [3, 4, 4]])
    t2 = torch.ones_like(data)  # 根据传入参数的 size(x行，y列)创建同样行列数的张量 **和数值无关
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')

    print('-' * 100)

def all_full_tensor():      # 创建给指定值的张量
    # 1. 采用给形状+指定值的方式创建全指定值张量
    t1 = torch.full((3, 4), 255)  # -> 创建一个三行四列(size = 3, 4)，指定值为255的二维张量
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')

    # 2. 采用传参的方式创建全指定张量
    data = torch.tensor([[1, 2, 3], [3, 4, 4]])
    t2 = torch.full_like(data, 255)  # 根据传入参数的 size(x行，y列) 和 fill_value(指定值) 创建同样行列数的张量 **和数值无关
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')

    print('-' * 100)


# 线性和随机张量的创建
def linear_tensor():            # 创建线性张量
    t1 = torch.arange(1, 5)     # 生成[1, 5)范围的一维张量，即1、2、3、4
    t2 = torch.linspace(1, 20, 3)   # 以1为起点、20为终点，生成3个等间隔值
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')

    print('-' * 100)

def random_tensor():
    t1 = torch.rand(3, 4)       # 均匀分布的值在0～1间的三行四列二维张量
    t2 = torch.randn(3, 4)      # 标准正态分布的三行四列二维张量
    t3 = torch.randint(0, 5, size=(3,5))    # 以[0,5)为区间的三行五列的二维张量
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')
    print(f'张量t3的元素值是:{t3}, t3的类型是:{type(t3)}, 张量t3中的元素值类型是:{t3.dtype}')

    print('-' * 100)


# 元素类型转换和设置
def value_convert():
    """
    元素类型设置方式：1.在创建时直接指定；2.创建后通过函数转变
    :return: none
    """
    # 1. 创建时直接指定
    t1 = torch.tensor((3, 4), dtype=torch.float)
    print(f'张量t1的元素值是:{t1}, t1的类型是:{type(t1)}, 张量t1中的元素值类型是:{t1.dtype}')

    # 2. 在创建后通过函数改变
    # 2.1 使用 .type()函数
    t1 = torch.tensor((3, 4), dtype=torch.float)
    t2 = t1.type(torch.int32)
    print(f'张量t2的元素值是:{t2}, t2的类型是:{type(t2)}, 张量t2中的元素值类型是:{t2.dtype}')

    # 2.2 使用 half(), float(), double(), short(), int(), long()的方式转变，不建议，太多了
    # 解释：half() -> float16, float() -> float32, double() -> float64, short() -> int16, int() -> int32, long() -> int64
    t1 = torch.tensor((3, 4), dtype=torch.float)
    t3 = t1.short()
    print(f'张量t3的元素值是:{t3}, t3的类型是:{type(t3)}, 张量t3中的元素值类型是:{t3.dtype}')

    print('-' * 100)


if __name__ == '__main__':
    define_tensor_by_torch_tensor()     # 使用 torch.tensor 创建张量
    define_tensor_by_torch_Tensor()     # 使用 torch.Tensor 创建张量
    define_tensor_by_given_types()      # 使用指定数值类型的方式创建，如：torch.IntTensor, torch.FloatTensor, torch.LongTensor, torch.DoubleTensor
    all_zeros_tensor()                  # 创建全0张量
    all_ones_tensor()                   # 创建全1张量
    all_full_tensor()                   # 创建全指定值张量
    linear_tensor()                     # 创建线性张量
    random_tensor()                     # 创建随机张量
    value_convert()                     # 元素类型转换和设置+

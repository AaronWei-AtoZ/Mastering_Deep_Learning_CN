"""
1. 使用 torch.manual_seed（42） 固定随机种子；
2. 创建形状为（2，3）、dtype=torch.float64、并开启自动求导的随机张量 X；
3. 创建与 X 同形状、dtype=torch.float64、元素全为 2.0 的常数张量C（不需自动求导）；
4. 定义标量损失
L= mean(X×C) + sum(x^2)
L = mean(X×C) + sum(X2)
并对其调用 L.backward（）完成反向传播；
5. 打印X.grad
"""

import torch

torch.manual_seed(42)


x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
print(x)
C = torch.full_like(x, fill_value=2.0, dtype=torch.float64, requires_grad=False)
print(C)
L = torch.mean(x * C) + torch.sum(x ** 2)
L.backward()
print(x.grad)
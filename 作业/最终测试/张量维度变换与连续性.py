"""
请使用 PyTorch 编写一段代码，完成下列操作：
1. 固定随机种子为 42；
2. 创建一个形状为（2，3, 4）的随机张量 T（dtype=torch.float32，不需要自动求导）；
3. 对T 调用 permute（1，0, 2），得到张量 P；
4. 判断 P 是否是连续张量，如果不是，则用 .contiguous（）生成连续张量 P_contig；
5. 将 P_contig 重塑为形状（6，4），并打印其形状
"""

import torch

torch.manual_seed(42)

T = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=False)
print(f"张量 T 的形状: {T.shape}")

P = T.permute(1, 0, 2)
print(f"张量 P 的形状: {P.shape}")

print(f"执行 permute 后，张量 P 是否连续: {P.is_contiguous()}")

if not P.is_contiguous():
    P_contig = P.contiguous()
    print("已执行 .contiguous() 强制内存连续。")
else:
    P_contig = P

# 5. 将 P_contig 重塑为形状 (6, 4)，并打印其形状
P_reshaped = P_contig.view(6, 4)
print(f"最终重塑后的张量形状: {P_reshaped.shape}")

import torch
torch.manual_seed(42)
# 多维索引
t1 = torch.randint(1, 10, (1, 2, 3))

t2 = t1.unsqueeze(3)

print(f't2:{t2}, 形状:{t2.shape}')


print(t1)
print(t1[0, :, :])
print(t1[:, 0, :])
print(t1[:, :, 0])
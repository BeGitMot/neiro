import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print((x[x > int(input())]).sum())


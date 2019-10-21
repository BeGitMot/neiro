
'''
import torch

w = torch.tensor( [[1.,2.],[4.,5.]], requires_grad=True)

function = 10 * torch.log( ( w + 1. ) ).sum()

print (function)

function.backward()

print (function)

print(w-w.grad)
'''

import torch
x = torch.tensor([[1., 2.], [4., 5.]])
print (x.shape)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i, j] = x[i, j] - 10/(x[i, j] + 1)

print (x)

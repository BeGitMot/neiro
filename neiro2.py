import torch
import numpy as np

print (torch.cuda.is_available())
'''
x = np.array([[1, 2, 3, 4],[4, 3, 2, 1]])
x = torch.from_numpy(x)
x = x.numpy()
print (x)
'''
#x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#print((x[x > int(input())]).sum())





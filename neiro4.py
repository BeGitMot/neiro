import torch

x = torch.tensor(
    [[1.,  2.,  3.,  4.],
     [5.,  6.,  7.,  8.],
     [9., 10., 11., 12.]], requires_grad=True)

#######
device = torch.device('cuda:0'
                      if torch.cuda.is_available()
                      else 'cpu')
x = x.to(device)
#######

funct = 10 * (x ** 2).sum()

funct.backward()

print(x.grad, '<- gradient')

x.data -= 0.001 * x.grad
x.grad.zero_()

print (x)

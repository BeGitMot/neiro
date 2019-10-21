import torch

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)

function = (w + 7).log().log().prod()

function.backward()

print(w.grad)
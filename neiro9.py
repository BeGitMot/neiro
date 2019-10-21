import torch

w = torch.tensor([[1.]], requires_grad=True)
alpha = 1
optimizer =  torch.optim.SGD([w], lr=alpha)

for _ in range(500):
    # it's critical to calculate function inside the loop:
    function = w**2
    function.backward()
    optimizer.step()
    optimizer.zero_grad()
    print (w)

print(w)
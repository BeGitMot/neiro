import torch

x = torch.tensor([8., 8.], requires_grad=True)

optimizer = torch.optim.SGD([x], lr=0.001)

def function_parabola(variable):
    return 10 * (variable ** 2).sum()

def make_step(function, variable):
    result = function(variable)
    result.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(x)

for _ in range(500):
    make_step(function_parabola, x)

print(x)
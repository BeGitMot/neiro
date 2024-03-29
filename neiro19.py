import torch
import numpy as np

#seed = int(input())
seed = 11
np.random.seed(seed)
torch.manual_seed(seed)

NUMBER_OF_EXPERIMENTS = 200

class SimpleNet(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()

        self.activation = activation
        self.fc1 = torch.nn.Linear(1, 1, bias=False)  # one neuron without bias
        self.fc1.weight.data.fill_(1.)  # init weight with 1
        self.fc2 = torch.nn.Linear(1, 1, bias=False)
        self.fc2.weight.data.fill_(1.)
        self.fc3 = torch.nn.Linear(1, 1, bias=False)
        self.fc3.weight.data.fill_(1.)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

    def get_fc1_grad_abs_value(self):
        return torch.abs(self.fc1.weight.grad)

def get_fc1_grad_abs_value(net, x):
    output = net.forward(x)
    output.backward()  # no loss function. Pretending that we want to minimize output
                       # In our case output is scalar, so we can calculate backward
    fc1_grad = net.get_fc1_grad_abs_value().item()
    net.zero_grad()
    return fc1_grad


funcs = [torch.nn.ELU(), torch.nn.Hardtanh(), torch.nn.LeakyReLU(), torch.nn.LogSigmoid(), torch.nn.PReLU(), torch.nn.ReLU(), torch.nn.ReLU6(),\
         torch.nn.RReLU(), torch.nn.SELU(), torch.nn.CELU(), torch.nn.Sigmoid(), torch.nn.Softplus(), torch.nn.Softshrink(), torch.nn.Softsign(),\
         torch.nn.Tanh(), torch.nn.Tanhshrink(), torch.nn.Hardshrink()]

for f in funcs:

    activation =  f#torch.nn.LeakyReLU()

    net = SimpleNet(activation=activation)

    fc1_grads = []
    for x in torch.randn((NUMBER_OF_EXPERIMENTS, 1)):
        fc1_grads.append(get_fc1_grad_abs_value(net, x))

    print(f, np.mean(fc1_grads))
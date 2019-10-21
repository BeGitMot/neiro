import torch
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)

def myFunc(x):
    return torch.sin(x)

def getXTrain():
    return (torch.rand(100)*20.0 - 10.0)

def getYTrain(x):
    noise = torch.randn(x.shape) / 5.
    return myFunc(x) + noise

def getXValidation():
    return torch.linspace(-10, 10, 100)

def getYValidation(x):
    return myFunc(x)

def drawIt(x, y, color = 'r'):
    plt.plot(x.numpy(), y.numpy(), 'o', c=color);

x_train = getXTrain()
y_train = getYTrain(x_train)

x_validation = getXValidation()
y_validation = getYValidation(x_validation)

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

#drawIt(x_train, y_train, 'b')
drawIt(x_validation, y_validation, 'y')

class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)

        return x

sine_net = SineNet(10)

#print(sine_net)

def predict(net, x, y):
    y_pred = net.forward(x)
    drawIt(x, y_pred.data, 'r')

optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)

def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()

for epoch_index in range(2000):
    optimizer.zero_grad()
    y_pred = sine_net.forward(x_train)
    loss_val = loss(y_pred, y_train)
    loss_val.backward()
    optimizer.step()

predict(sine_net, x_validation, y_validation)

plt.show()

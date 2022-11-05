import torch

from torch import nn
from torch.nn import functional


def test():
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

    X = torch.rand(5, 20)
    return net(X)


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.aggr = nn.ReLU()
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(self.aggr(self.hidden(X)))


if __name__ == '__main__':
    X = torch.rand(2, 20)
    mlp = MLP()
    y = mlp.__call__(X)
    print(y)

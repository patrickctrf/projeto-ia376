import torch
from torch import nn


class SReLU(nn.Module):
    def __init__(self, threshold=0.8, *args, **kwargs):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):

        return torch.where(x > self.threshold, self.threshold + x * 1e-2, torch.where(x < -self.threshold, -self.threshold + x * 1e-2, x))


if __name__ == '__main__':
    activation = SReLU()

    x = activation(torch.tensor([-2.0]))
    print(x)
    x = activation(torch.tensor([-0.5]))
    print(x)
    x = activation(torch.tensor([0.0]))
    print(x)
    x = activation(torch.tensor([0.5]))
    print(x)
    x = activation(torch.tensor([2.0]))
    print(x)

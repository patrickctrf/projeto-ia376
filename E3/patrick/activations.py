import torch
from torch import nn


class SReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x_leaky_relu = self.leaky_relu(x)

        return torch.where(x_leaky_relu > 1.0, 1 + x_leaky_relu * 1e-2, x_leaky_relu)


if __name__ == '__main__':
    activation = SReLU()

    x = activation(torch.tensor([-1.0]))
    print(x)
    x = activation(torch.tensor([0.0]))
    print(x)
    x = activation(torch.tensor([0.5]))
    print(x)
    x = activation(torch.tensor([2.0]))
    print(x)

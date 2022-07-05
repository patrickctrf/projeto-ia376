from torch import nn


class ExpSquareLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, y, y_hat):
        return ((y - y_hat) ** 2).exp().mean()


class HyperbolicLoss(nn.Module):
    def __init__(self, epsilon=1e-6, *args, **kwargs):
        super().__init__()
        self.epsilon = epsilon + 1

    def forward(self, y, y_hat):
        return (1 / (self.epsilon - ((y - y_hat) ** 2))).mean()

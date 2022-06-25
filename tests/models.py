import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid

__all__ = ["Generator2DUpsampled", "Discriminator2D", "DummyGenerator"]


class DummyGenerator(nn.Module):
    def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()

        self.dummy_tensor = nn.Parameter(torch.rand((1, 2, 1024, 128), requires_grad=True))

    def forward(self, x):
        return self.dummy_tensor


class Generator2DUpsampled(nn.Module):
    def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()
        self.target_length = target_length

        n_filters = 256

        self.feature_generator = Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=(2, 16), stride=(1, 1), dilation=(1, 1), padding=(1, 15), bias=bias), nn.Tanh(),
            nn.Upsample(scale_factor=2.0, mode='nearest', align_corners=None),
            ResBlock(256, 256, kernel_size=3, stride=1, dilation=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2.0, mode='nearest', align_corners=None),
            ResBlock(256, 256, kernel_size=3, stride=1, dilation=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2.0, mode='nearest', align_corners=None),
            ResBlock(256, 256, kernel_size=3, stride=1, dilation=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2.0, mode='nearest', align_corners=None),
            ResBlock(256, 128, kernel_size=3, stride=1, dilation=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2.0, mode='nearest', align_corners=None),
            ResBlock(128, 64, kernel_size=3, stride=1, dilation=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2.0, mode='nearest', align_corners=None),
            ResBlock(64, 32, kernel_size=3, stride=1, dilation=1, padding=0, bias=bias),
            nn.Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias),
        )

    def forward(self, x):
        som_2_canais = self.feature_generator(x).transpose(2, 3)
        som_2_canais[:, 1, :, :] = torch.tanh(som_2_canais[:, 1, :, :]) * 3.1415926535
        return som_2_canais


class Discriminator2D(nn.Module):
    def __init__(self, seq_length=64000, n_input_channels=24,
                 kernel_size=7, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()

        n_output_channels = 256

        self.feature_extractor = Sequential(
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, stride=3, dilation=2, bias=bias, ), nn.Tanh(),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
        )

        self.mlp = nn.Sequential(
            Linear(2560, 1024, bias=bias),
            nn.Tanh(),
            Linear(1024, 1, bias=bias),
        )

        self.activation = Sigmoid()

    def forward(self, x):
        return self.activation(self.mlp(self.feature_extractor(x).flatten(start_dim=1)))


class ResBlock(nn.Module):
    def __init__(self, n_input_channels=6, n_output_channels=7,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        """
    ResNet-like block, receives as arguments the same that PyTorch's Conv1D
    module.
        """
        super(ResBlock, self).__init__()

        self.feature_extractor = \
            Sequential(
                nn.Conv2d(n_input_channels, n_output_channels, kernel_size,
                          stride, kernel_size // 2 * dilation, dilation,
                          groups, bias, padding_mode),
                nn.Tanh(),
                nn.Conv2d(n_output_channels, n_output_channels, kernel_size,
                          stride, kernel_size // 2 * dilation,
                          dilation, groups, bias, padding_mode),
            )

        self.skip_connection = \
            Sequential(
                nn.Conv2d(n_input_channels, n_output_channels, 1,
                          stride, padding, dilation, groups, bias, padding_mode)
            )

        self.activation = nn.Tanh()

    def forward(self, input_seq):
        return self.activation(self.feature_extractor(input_seq) + self.skip_connection(input_seq))


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

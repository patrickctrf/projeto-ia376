from torch import nn
from torch.nn import Sequential, Conv1d, AdaptiveAvgPool1d, Linear


class Generator1D(nn.Module):
    def __init__(self, noise_length=256, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()

        self.feature_generator = Sequential(
            ResBlock(n_input_channels=1 * n_input_channels,
                     n_output_channels=10 * n_input_channels,
                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                     bias=bias),
            ResBlock(n_input_channels=10 * n_input_channels,
                     n_output_channels=n_output_channels,
                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                     bias=bias),
        )

        # self.pooling = AdaptiveAvgPool1d(1)

        # self.linear = Linear(noise_length * n_output_channels, 64000)

    def forward(self, x):
        # print("\nx.shape: ", x.shape)
        #
        # features = self.feature_generator(x).flatten(start_dim=1)
        #
        # print("features.shape: ", features.shape)
        #
        # out = self.linear(features)
        #
        # print("out.shape: ", out.shape)
        #
        # return out

        return self.feature_generator(x).view(-1, 1, 64000)


class Discriminator1D(nn.Module):
    def __init__(self, seq_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()

        self.feature_extractor = Sequential(
            ResBlock(n_input_channels=n_input_channels,
                     n_output_channels=n_output_channels,
                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                     bias=bias),
            ResBlock(n_input_channels=n_output_channels,
                     n_output_channels=n_output_channels,
                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                     bias=bias),
            ResBlock(n_input_channels=n_output_channels,
                     n_output_channels=n_output_channels,
                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                     bias=bias),
            ResBlock(n_input_channels=n_output_channels,
                     n_output_channels=n_output_channels,
                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                     bias=bias),
            ResBlock(n_input_channels=n_output_channels,
                     n_output_channels=n_output_channels,
                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                     bias=bias),
        )

        # self.pooling = AdaptiveAvgPool1d(1)

        self.linear = Linear(seq_length * n_output_channels, 1)

    def forward(self, x):

        return self.linear(self.feature_extractor(x).flatten(start_dim=1))


class ResBlock(nn.Module):
    def __init__(self, n_input_channels=6, n_output_channels=7,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='replicate'):
        """
    ResNet-like block, receives as arguments the same that PyTorch's Conv1D
    module.
        """
        super(ResBlock, self).__init__()

        self.feature_extractor = \
            Sequential(
                Conv1d(n_input_channels, n_output_channels, kernel_size,
                       stride, kernel_size // 2 * dilation, dilation,
                       groups, bias, padding_mode),
                nn.LeakyReLU(),
                nn.BatchNorm1d(n_output_channels),
                Conv1d(n_output_channels, n_output_channels, kernel_size,
                       stride, kernel_size // 2 * dilation,
                       dilation, groups, bias, padding_mode),
                nn.PReLU(num_parameters=n_output_channels),
                nn.BatchNorm1d(n_output_channels)
            )

        self.skip_connection = \
            Sequential(
                Conv1d(n_input_channels, n_output_channels, (1,),
                       stride, padding, dilation, groups, bias, padding_mode)
            )

    def forward(self, input_seq):
        return self.feature_extractor(input_seq) + self.skip_connection(input_seq)

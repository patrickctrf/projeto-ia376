from torch import nn
from torch.nn import Sequential, Conv1d, Linear, AdaptiveAvgPool1d


class Generator1DUpsampled(nn.Module):
    def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()
        self.target_length = target_length

        n_filters = 32

        self.feature_generator = Sequential(
            nn.Upsample(size=(32000,), mode='linear', align_corners=True),
            nn.Conv1d(n_input_channels, 256, kernel_size=(7,), stride=(1,), dilation=(1,), bias=False), nn.PReLU(256), nn.BatchNorm1d(256, affine=False),
            nn.Conv1d(256, 256, kernel_size=(7,), stride=(1,), dilation=(1,), bias=False), nn.PReLU(256), nn.BatchNorm1d(256, affine=False),
            nn.Upsample(size=(64012,), mode='linear', align_corners=True),
            nn.Conv1d(256, 256, kernel_size=(7,), stride=(1,), dilation=(1,), bias=False), nn.PReLU(256), nn.BatchNorm1d(256, affine=False),
            nn.Conv1d(256, n_output_channels, kernel_size=(7,), stride=(1,), dilation=(1,), bias=False),
        )

    def forward(self, x):
        return self.feature_generator(x).view(-1, 1, self.target_length)


class Discriminator1D(nn.Module):
    def __init__(self, seq_length=64000, n_input_channels=24, n_output_channels=1024,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()

        self.feature_extractor = Sequential(
            nn.Conv1d(1, 256, kernel_size=(7,), stride=(3,), dilation=(1,), bias=True), nn.PReLU(256), nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=(7,), stride=(3,), dilation=(1,), bias=True), nn.PReLU(256), nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=(7,), stride=(3,), dilation=(1,), bias=True), nn.PReLU(256), nn.BatchNorm1d(256),
            nn.Conv1d(256, n_output_channels, kernel_size=(7,), stride=(3,), dilation=(1,), bias=True), nn.PReLU(n_output_channels), nn.BatchNorm1d(n_output_channels),
        )

        self.pooling = AdaptiveAvgPool1d(1)

        self.linear = Linear(n_output_channels, 1)

    def forward(self, x):
        return self.linear(self.pooling(self.feature_extractor(x)).flatten(start_dim=1))


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


class Generator1DTransposed(nn.Module):
    def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()
        self.target_length = target_length

        n_filters = 32

        self.feature_generator = Sequential(
            nn.ConvTranspose1d(1 * n_filters, 1 * n_filters, kernel_size=(4,), stride=(1,), padding=(1,), bias=False),  # L=2
            nn.PReLU(1 * n_filters), nn.BatchNorm1d(1 * n_filters, affine=False),
            nn.ConvTranspose1d(1 * n_filters, 2 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=4
            nn.PReLU(2 * n_filters), nn.BatchNorm1d(2 * n_filters, affine=False),
            nn.ConvTranspose1d(2 * n_filters, 2 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=8
            nn.PReLU(2 * n_filters), nn.BatchNorm1d(2 * n_filters, affine=False),
            nn.ConvTranspose1d(2 * n_filters, 3 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=16
            nn.PReLU(3 * n_filters), nn.BatchNorm1d(3 * n_filters, affine=False),
            nn.ConvTranspose1d(3 * n_filters, 3 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=32
            nn.PReLU(3 * n_filters), nn.BatchNorm1d(3 * n_filters, affine=False),
            nn.ConvTranspose1d(3 * n_filters, 4 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=64
            nn.PReLU(4 * n_filters), nn.BatchNorm1d(4 * n_filters, affine=False),
            nn.ConvTranspose1d(4 * n_filters, 4 * n_filters, kernel_size=(5,), stride=(2,), padding=(3,), bias=False),  # L=125
            nn.PReLU(4 * n_filters), nn.BatchNorm1d(4 * n_filters, affine=False),
            nn.ConvTranspose1d(4 * n_filters, 5 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=250
            nn.PReLU(5 * n_filters), nn.BatchNorm1d(5 * n_filters, affine=False),
            nn.ConvTranspose1d(5 * n_filters, 5 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=500
            nn.PReLU(5 * n_filters), nn.BatchNorm1d(5 * n_filters, affine=False),
            nn.ConvTranspose1d(5 * n_filters, 6 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=1000
            nn.PReLU(6 * n_filters), nn.BatchNorm1d(6 * n_filters, affine=False),
            nn.ConvTranspose1d(6 * n_filters, 6 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=2000
            nn.PReLU(6 * n_filters), nn.BatchNorm1d(6 * n_filters, affine=False),
            nn.ConvTranspose1d(6 * n_filters, 7 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=4000
            nn.PReLU(7 * n_filters), nn.BatchNorm1d(7 * n_filters, affine=False),
            nn.ConvTranspose1d(7 * n_filters, 7 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=8000
            nn.PReLU(7 * n_filters), nn.BatchNorm1d(7 * n_filters, affine=False),
            nn.ConvTranspose1d(7 * n_filters, 8 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=16000
            nn.PReLU(8 * n_filters), nn.BatchNorm1d(8 * n_filters, affine=False),
            nn.ConvTranspose1d(8 * n_filters, 8 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=32000
            nn.PReLU(8 * n_filters), nn.BatchNorm1d(8 * n_filters, affine=False),
            nn.ConvTranspose1d(8 * n_filters, 9 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=False),  # L=64000
            nn.PReLU(9 * n_filters), nn.BatchNorm1d(9 * n_filters, affine=False),
            nn.Conv1d(9 * n_filters, 1, (1,), bias=False),  # L=64000
        )

    def forward(self, x):
        return self.feature_generator(x).view(-1, 1, self.target_length)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

#
# class Generator1D(nn.Module):
#     def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
#                  kernel_size=7, stride=1, padding=0, dilation=1,
#                  bias=False):
#         super().__init__()
#         self.target_length = target_length
#
#         self.feature_generator = Sequential(
#             ResBlock(n_input_channels=1 * n_input_channels,
#                      n_output_channels=10 * n_input_channels,
#                      kernel_size=kernel_size, stride=stride, dilation=dilation,
#                      bias=bias),
#             ResBlock(n_input_channels=10 * n_input_channels,
#                      n_output_channels=n_output_channels,
#                      kernel_size=kernel_size, stride=stride, dilation=dilation,
#                      bias=bias),
#         )
#
#     def forward(self, x):
#         return self.feature_generator(x).view(-1, 1, self.target_length)
#
#
# class Discriminator1D(nn.Module):
#     def __init__(self, seq_length=64000, n_input_channels=24, n_output_channels=64,
#                  kernel_size=7, stride=1, padding=0, dilation=1,
#                  bias=False):
#         super().__init__()
#
#         self.feature_extractor = Sequential(
#             ResBlock(n_input_channels=n_input_channels,
#                      n_output_channels=n_output_channels,
#                      kernel_size=kernel_size, stride=stride, dilation=dilation,
#                      bias=bias),
#             ResBlock(n_input_channels=n_output_channels,
#                      n_output_channels=n_output_channels,
#                      kernel_size=kernel_size, stride=stride, dilation=dilation,
#                      bias=bias),
#         )
#
#         # self.pooling = AdaptiveAvgPool1d(1)
#
#         self.linear = Linear(seq_length * n_output_channels, 1)
#
#     def forward(self, x):
#         return self.linear(self.feature_extractor(x).flatten(start_dim=1))


# class Discriminator1D(nn.Module):
#     def __init__(self, seq_length=64000, n_input_channels=24, n_output_channels=64,
#                  kernel_size=7, stride=1, padding=0, dilation=1,
#                  bias=False):
#         super().__init__()
#
#         self.feature_extractor = Sequential(
#             ResBlock(n_input_channels=n_input_channels,
#                      n_output_channels=n_output_channels,
#                      kernel_size=kernel_size, stride=stride, dilation=dilation,
#                      bias=bias),
#             ResBlock(n_input_channels=n_output_channels,
#                      n_output_channels=n_output_channels,
#                      kernel_size=kernel_size, stride=stride, dilation=dilation,
#                      bias=bias),
#         )
#
#         # self.pooling = AdaptiveAvgPool1d(1)
#
#         self.linear = Linear(seq_length * n_output_channels, 1)
#
#     def forward(self, x):
#         return self.linear(self.feature_extractor(x).flatten(start_dim=1))

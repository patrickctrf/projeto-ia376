from torch import nn
from torch.nn import Sequential, Conv1d, Linear, AdaptiveAvgPool1d, Sigmoid, AdaptiveMaxPool1d

__all__ = ["Generator2DUpsampled", "Generator1DUpsampled", "Generator1DTransposed", "Discriminator2D", "Discriminator1D"]


class Generator2DUpsampled(nn.Module):
    def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()
        self.target_length = target_length

        n_filters = 256

        self.feature_generator = Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(2, 9), bias=bias), nn.Tanh(),
            nn.Upsample(size=(4, 32), mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Upsample(size=(8, 64), mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Upsample(size=(16, 128), mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Upsample(size=(32, 256), mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Upsample(size=(64, 512), mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Upsample(size=(128, 1024), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias), nn.Tanh(),
            nn.Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding='same', bias=bias),
        )

    def forward(self, x):
        return self.feature_generator(x).transpose(2, 3)


class Discriminator2D(nn.Module):
    def __init__(self, seq_length=64000, n_input_channels=24,
                 kernel_size=7, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()

        n_output_channels = 256

        self.feature_extractor = Sequential(
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, stride=3, dilation=2, bias=bias), nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias), nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias), nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias), nn.Tanh(),
        )

        self.pooling = nn.AdaptiveMaxPool2d(1)

        self.linear = Linear(n_output_channels, 1, bias=bias)

        self.activation = Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(self.pooling(self.feature_extractor(x)).flatten(start_dim=1)))


class Discriminator1D(nn.Module):
    def __init__(self, seq_length=64000, n_input_channels=24,
                 kernel_size=7, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()

        n_output_channels = 32

        self.feature_extractor = Sequential(
            # nn.BatchNorm1d(n_input_channels, affine=False),
            nn.Conv1d(n_input_channels, n_output_channels, kernel_size=(3,), stride=(3,), dilation=(2,), bias=bias), nn.Tanh(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(n_output_channels, n_output_channels, kernel_size=(3,), stride=(1,), dilation=(1,), bias=bias), nn.Tanh(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(n_output_channels, n_output_channels, kernel_size=(3,), stride=(1,), dilation=(1,), bias=bias), nn.Tanh(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(n_output_channels, n_output_channels, kernel_size=(3,), stride=(1,), dilation=(1,), bias=bias), nn.Tanh(),
        )

        self.pooling = AdaptiveMaxPool1d(1)

        self.linear = Linear(n_output_channels, 1, bias=bias)

        self.activation = Sigmoid()

        self.aux = Sequential(
            nn.Flatten(),
            Linear(64000, 10),
            nn.LeakyReLU(),
            Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.activation(self.linear(self.pooling(self.feature_extractor(x)).flatten(start_dim=1)))
        # return self.aux(x)


class Generator1DTransposed(nn.Module):
    def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()
        self.target_length = target_length

        n_filters = 32

        self.feature_generator = Sequential(
            nn.ConvTranspose1d(n_input_channels, 1 * n_filters, kernel_size=(4,), stride=(1,), padding=(1,), bias=bias),  # L=2
            nn.Tanh(),
            nn.ConvTranspose1d(1 * n_filters, 2 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=4
            nn.Tanh(),
            nn.ConvTranspose1d(2 * n_filters, 2 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=8
            nn.Tanh(),
            nn.ConvTranspose1d(2 * n_filters, 3 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=16
            nn.Tanh(),
            nn.ConvTranspose1d(3 * n_filters, 3 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=32
            nn.Tanh(),
            nn.ConvTranspose1d(3 * n_filters, 4 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=64
            nn.Tanh(),
            nn.ConvTranspose1d(4 * n_filters, 4 * n_filters, kernel_size=(5,), stride=(2,), padding=(3,), bias=bias),  # L=125
            nn.Tanh(),
            nn.ConvTranspose1d(4 * n_filters, 5 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=250
            nn.Tanh(),
            nn.ConvTranspose1d(5 * n_filters, 5 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=500
            nn.Tanh(),
            nn.ConvTranspose1d(5 * n_filters, 6 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=1000
            nn.Tanh(),
            nn.ConvTranspose1d(6 * n_filters, 6 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=2000
            nn.Tanh(),
            nn.ConvTranspose1d(6 * n_filters, 7 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=4000
            nn.Tanh(),
            nn.ConvTranspose1d(7 * n_filters, 7 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=8000
            nn.Tanh(),
            nn.ConvTranspose1d(7 * n_filters, 8 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=16000
            nn.Tanh(),
            nn.ConvTranspose1d(8 * n_filters, 8 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=32000
            nn.Tanh(),
            nn.ConvTranspose1d(8 * n_filters, 9 * n_filters, kernel_size=(4,), stride=(2,), padding=(1,), bias=bias),  # L=64000
            nn.Tanh(),
            nn.Conv1d(9 * n_filters, 1, (1,), bias=bias),  # L=64000
        )

    def forward(self, x):
        return self.feature_generator(x).view(-1, 1, self.target_length)


class ResLayer(nn.Module):
    def __init__(self, n_input_channels=6,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        """
    ResNet-like block, receives as arguments the same that PyTorch's Conv1D
    module.
        """
        super(ResLayer, self).__init__()

        self.conv = Conv1d(n_input_channels, n_input_channels, kernel_size,
                           stride, 'same', dilation,
                           groups, bias, padding_mode)

    def forward(self, input_seq):
        return self.conv(input_seq) + input_seq


class Generator1DUpsampled(nn.Module):
    def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()
        self.target_length = target_length

        n_filters = 256

        self.feature_generator = Sequential(
            nn.Conv1d(n_input_channels, n_filters, kernel_size=(3,), stride=(1,), dilation=(1,), padding=(2,), bias=bias), nn.Tanh(),
            nn.Upsample(size=(8,), mode='linear', align_corners=True),
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=1, bias=bias), nn.Tanh(),
            nn.Upsample(size=(32,), mode='linear', align_corners=True),
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=2, bias=bias), nn.Tanh(),
            nn.Upsample(size=(128,), mode='linear', align_corners=True),
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=2, bias=bias), nn.Tanh(),
            nn.Upsample(size=(1024,), mode='linear', align_corners=True),
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=2, bias=bias), nn.Tanh(),
            nn.Upsample(size=(4096,), mode='linear', align_corners=True),
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=2, bias=bias), nn.Tanh(),
            nn.Upsample(size=(16384,), mode='linear', align_corners=True),
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=2, bias=bias), nn.Tanh(),
            nn.Upsample(size=(64000,), mode='linear', align_corners=True),
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=1, bias=bias), nn.Tanh(),
            nn.Upsample(size=(64000,), mode='linear', align_corners=True),
            nn.Conv1d(n_filters, n_output_channels, kernel_size=(1,), stride=(1,), dilation=(1,), padding=(0,), bias=bias),
        )

    def forward(self, x):
        return self.feature_generator(x).view(-1, 1, self.target_length)


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

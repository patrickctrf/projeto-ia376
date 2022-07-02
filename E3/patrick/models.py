import torch
from torch import nn
from torch.nn import Sequential, Conv1d, Linear, Sigmoid, AdaptiveMaxPool1d

__all__ = ["Generator2DUpsampled", "Generator1DUpsampled", "Generator1DTransposed", "Discriminator2D", "Discriminator1D"]

from activations import BnActivation, LnActivation


class _AttentionLayer(torch.nn.Module):

    def __init__(self, embedding_dim: int, max_seq_length: int = 300):
        """
        Implements the Self-attention, decoder-only."

        Args:
            max_seq_length (int): Size of the sequence to consider as context for prediction.
            embedding_dim (int): Dimension of the embedding layer for each word in the context.
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # hidden_size for MLP
        hidden_size = 2048

        # Linear projections
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_0 = nn.Linear(embedding_dim, embedding_dim)

        # output MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            LnActivation(hidden_size),
            # nn.Dropout(0.1),
            nn.Linear(hidden_size, embedding_dim),
        )

        self.activation = nn.LeakyReLU()

        # cast to probabilities
        self.softmax = nn.Softmax(dim=-1)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Matriz triangular de mascara, convertida para Booleano
        # Onde vale 1, o valor deve ser substituida por um valor negativo alto no tensor de scores.
        self.casual_mask = torch.ones((max_seq_length, max_seq_length), ).triu(diagonal=1) == 1.0

    def forward(self, x_embeddings):
        k = self.w_k(x_embeddings)
        v = self.w_v(x_embeddings)
        q = self.w_q(x_embeddings)

        scores = torch.matmul(q, k.transpose(1, 2))

        probabilities = self.softmax(scores)

        e = self.w_0(self.norm1(x_embeddings + torch.matmul(probabilities, v)))

        logits = self.mlp(e)

        return self.norm2(self.activation(logits + e))


class TransformerDiscriminator(torch.nn.Module):

    def __init__(self, dim: int, n_layers: int = 2, max_seq_length: int = 300, input_size: int = 6, output_size: int = 7):
        """
        Implements the Self-attention, decoder-only."

        Args:
            input_size (int): Size of the input features.
            max_seq_length (int): Size of the sequence to consider as context for prediction.
            dim (int): Dimension of the embedding layer for each word in the context.
            n_layers (int): number of self-attention layers.
        """
        # Escreva seu código aqui.
        super().__init__()
        embedding_dim = dim
        self.embedding_dim = embedding_dim

        # tokens (words indexes) embedding and positional embedding
        self.c_embedding = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            LnActivation(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.p_embedding = nn.Embedding(max_seq_length, embedding_dim)

        self.attention = nn.Sequential(*[_AttentionLayer(embedding_dim=embedding_dim) for _ in range(n_layers)])

        self.CLS = nn.Parameter(torch.randn((1, 1, input_size,)))

        self.dense_network = nn.Linear(embedding_dim, output_size)

    def forward(self, inputs):
        # Precisa adicionar o token CLS no inicio de cada sequencia
        inputs = torch.cat((self.CLS.repeat(inputs.shape[0], 1, 1), inputs), dim=1)

        positional_indexes = torch.arange(inputs.shape[1], device=inputs.device).view(1, -1)

        input_embeddings = self.c_embedding(inputs)

        positional_embeddings = self.p_embedding(positional_indexes.repeat(inputs.shape[0], 1))

        x_embeddings = positional_embeddings + input_embeddings

        logits = self.attention(x_embeddings)

        return torch.sigmoid(self.dense_network(logits)[:, 0])


class TransformerGenerator(torch.nn.Module):

    def __init__(self, dim: int, n_layers: int = 2, max_seq_length: int = 300, input_size: int = 6, output_size: int = 7):
        """
        Implements the Self-attention, decoder-only."

        Args:
            input_size (int): Size of the input features.
            max_seq_length (int): Size of the sequence to consider as context for prediction.
            dim (int): Dimension of the embedding layer for each word in the context.
            n_layers (int): number of self-attention layers.
        """
        # Escreva seu código aqui.
        super().__init__()
        embedding_dim = dim
        self.embedding_dim = embedding_dim

        # hidden_size for MLP
        hidden_size = 2048

        # tokens (words indexes) embedding and positional embedding
        self.c_embedding = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            LnActivation(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.p_embedding = nn.Embedding(max_seq_length, embedding_dim)

        self.attention = nn.Sequential(*[_AttentionLayer(embedding_dim=embedding_dim) for _ in range(n_layers)])

        self.dense_network = nn.Linear(embedding_dim, output_size)

    def forward(self, inputs):
        positional_indexes = torch.arange(inputs.shape[1], device=inputs.device).view(1, -1)

        input_embeddings = self.c_embedding(inputs)

        positional_embeddings = self.p_embedding(positional_indexes.repeat(inputs.shape[0], 1))

        x_embeddings = positional_embeddings + input_embeddings

        logits = self.attention(x_embeddings)

        som_2_canais = self.dense_network(logits).view(-1, 2, 128, 1024)
        som_2_canais[:, 1, :] = torch.tanh(som_2_canais[:, 1, :]) * 3.1415926535897
        som_2_canais[:, 0, :] = torch.tanh(som_2_canais[:, 0, :]) * 11.85 - 7.35
        return som_2_canais.transpose(2, 3)


class Generator2DUpsampled(nn.Module):
    def __init__(self, noise_length=256, target_length=64000, n_input_channels=24, n_output_channels=64,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 bias=False):
        super().__init__()
        self.target_length = target_length

        self.feature_generator = Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=(2, 16), stride=(1, 1), dilation=(1, 1), padding=(1, 15), bias=bias), BnActivation(256),
            ResBlock(256, 256, kernel_size=3, stride=1, dilation=1, padding=0, bias=bias),
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

        self.activation = nn.Tanh()

    def forward(self, x):
        som_2_canais = self.feature_generator(x).transpose(2, 3)
        som_2_canais[:, 1, :, :] = self.activation(som_2_canais[:, 1, :, :]) * 3.1415926535897
        som_2_canais[:, 0, :, :] = self.activation(som_2_canais[:, 0, :, :]) * 11.85 - 7.35
        return som_2_canais


class Discriminator2D(nn.Module):
    def __init__(self, seq_length=64000, n_input_channels=24,
                 kernel_size=7, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()

        n_output_channels = 256

        self.feature_extractor = Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=1, stride=3, dilation=3, bias=bias, ), BnActivation(32),
            ResBlock(32, 64, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(64, 128, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(128, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
            nn.AvgPool2d(2, 2),
            ResBlock(n_output_channels, n_output_channels, kernel_size=3, stride=1, dilation=1, bias=bias),
        )

        self.mlp = nn.Sequential(
            Linear(2560, 1024, bias=bias),
            LnActivation(1024),
            Linear(1024, 1, bias=bias),
        )

        self.activation = Sigmoid()

    def forward(self, x):
        return self.activation(self.mlp(self.feature_extractor(x).flatten(start_dim=1)))


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
            nn.Tanh(),
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
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=1, padding=1, bias=bias), nn.Tanh(),
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
            ResLayer(n_filters, kernel_size=3, stride=1, dilation=1, padding=1, bias=bias), nn.Tanh(),
            nn.Upsample(size=(64000,), mode='linear', align_corners=True),
            nn.Conv1d(n_filters, n_output_channels, kernel_size=(1,), stride=(1,), dilation=(1,), padding=(0,), bias=bias),
        )

    def forward(self, x):
        return self.feature_generator(x).view(-1, 1, self.target_length)


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
                BnActivation(n_output_channels),
                nn.Conv2d(n_output_channels, n_output_channels, kernel_size,
                          stride, kernel_size // 2 * dilation,
                          dilation, groups, bias, padding_mode),
            )

        self.skip_connection = \
            Sequential(
                nn.Conv2d(n_input_channels, n_output_channels, 1,
                          stride, padding, dilation, groups, bias, padding_mode)
            )

        self.activation = BnActivation(n_output_channels)

    def forward(self, input_seq):
        return self.activation(self.feature_extractor(input_seq) + self.skip_connection(input_seq))


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

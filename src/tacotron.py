import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Prenet(nn.Module):
    def __init__(self, in_features, out_features=[256, 128], dropout=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_features] + out_features[:-1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(in_dim, out_dim)
                for (in_dim, out_dim) in zip(in_sizes, out_features)
            ]
        )
        self.dropout = dropout

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=self.dropout, training=True)
        return x


class BatchNormConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, activation=None
    ):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.batch_norm(x)


class Highway(nn.Module):
    "paper : https://arxiv.org/pdf/1505.00387"

    def __init__(self, in_features, out_features):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_features, out_features, bias=False)
        self.T = nn.Linear(in_features, out_features, bias=False)

    def forward(self, inputs):
        H = F.relu(self.H(inputs))
        T = torch.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
    - 1-d convolution banks
    - Highway networks + residual connections
    - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [
                BatchNormConv1d(
                    in_dim,
                    in_dim,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    activation=self.relu,
                )
                for k in range(1, K + 1)
            ]
        )
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [
                BatchNormConv1d(
                    in_size, out_size, kernel_size=3, stride=1, padding=1, activation=ac
                )
                for (in_size, out_size, ac) in zip(in_sizes, projections, activations)
            ]
        )

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList([Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(
            self.conv1d_banks
        ), f"The input dimension {x.size(-1)} if different to the length of conv * input _dim wich is {self.in_dim*len(self.conv1d_banks)}"

        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


class Encoder(nn.Module):
    def __init__(self, in_dim):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_dim, out_features=[256, 128])
        self.cbhg = CBHG(in_dim=128, K=16, projections=[128, 128])

    def forward(self, inputs):
        # (B, T_in, in_dim)
        x = self.prenet(inputs)
        # (B, T_in, in_dim*2)
        outputs = self.cbhg(x)
        return outputs


if __name__ == "__main__":
    # Test Encoder
    encoder = Encoder(80)
    inputs = torch.randn(4, 100, 80)
    outputs = encoder(inputs)
    print(outputs.size())  # torch.Size([4, 100, 256])

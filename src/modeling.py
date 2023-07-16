import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from torch.autograd import Variable


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',.!?;: "

lookup = {s: i for i, s in enumerate(alphabet)}


class EncoderPrenet(nn.Module):
    def __init__(self, input_dim, output_dim_1, output_dim_2, dropout=0.5):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Linear(input_dim, output_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim_1, output_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        torch.nn.init.xavier_uniform_(
            self.lin[0].weight, gain=torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.xavier_uniform_(
            self.lin[3].weight, gain=torch.nn.init.calculate_gain("relu")
        )

    def forward(self, x):
        return self.lin(x)


class CEmbedding(nn.Module):
    def __init__(self, vocab_size, out_embd):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, out_embd)

        torch.nn.init.xavier_uniform_(
            self.embed.weight, gain=torch.nn.init.calculate_gain("relu")
        )

    def forward(self, text):
        return self.embed(text)


class Conv1dBank(nn.Module):
    """
    Conv1D bank: K=16, conv-k-128-ReLU Max pooling: stride=1, width=2

    arguements:
    -----------
        in_channels: input channels
        out_channels: output channels
        kernel_size: kernel size
    return:
    -------
        conv1d: 1D convolutional layer


    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1),
        )

        torch.nn.init.xavier_uniform_(
            self.conv[0].weight, gain=torch.nn.init.calculate_gain("relu")
        )

    def forward(self, x):
        return self.conv(x)


class Conv1dProj(nn.Module):
    def __init__(self, input_channels, projecions=[128, 128], kernel_size=3):
        super().__init__()
        self.poj = nn.Sequential(
            nn.Conv1d(input_channels, projecions[0], kernel_size),
            nn.ReLU(),
            nn.Conv1d(projecions[0], projecions[1], kernel_size),
            nn.ReLU(),
            # nn.Linear(projecions[1], input_channels),
        )
        torch.nn.init.xavier_uniform_(
            self.poj[0].weight, gain=torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.xavier_uniform_(
            self.poj[2].weight, gain=torch.nn.init.calculate_gain("relu")
        )

    def forward(self, x):
        proj = self.poj(x)
        proj = nn.Linear(proj.shape[2], x.shape[1])(proj)
        return proj


class Highway(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.H = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

        nn.init.xavier_uniform_(self.H[0].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.H[2].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.H[4].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.H[6].weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        return self.H(x)


class CBHG(nn.Module):
    def __init__(
        self,
        in_channels=[30, 128],
        out_channels=128,
        projecions=[128, 128],
        kernel_size=[16, 3],
    ):
        super().__init__()
        self.conv1dBank = Conv1dBank(in_channels[0], out_channels, kernel_size[0])
        self.conv1dProj = Conv1dProj(in_channels[1], projecions, kernel_size[1])
        self.highway = Highway(in_channels[1], out_channels)

        self.gru_bidirectional = nn.GRU(
            out_channels, out_channels, bidirectional=True, batch_first=True
        )

    def forward(self, x):
        conv = self.conv1dBank(x)
        proj = self.conv1dProj(conv)
        highway = self.highway(proj)
        gru_output, _ = self.gru_bidirectional(highway)
        return gru_output





class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim_1,
        output_dim_2,
        in_channels=[30, 128],
        out_channels=128,
        projecions=[128, 128],
        kernel_size=[16, 3],
    ):
        super().__init__()
        self.pre_net = EncoderPrenet(input_dim, output_dim_1, output_dim_2)
        self.cbhg = CBHG(in_channels, out_channels, projecions, kernel_size)

    def forward(self, x):
        prenet = self.pre_net(x)
        cbhg = self.cbhg(prenet)
        return cbhg
    

### Decoder 



class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(output_dim, 1)

        self.tanh = nn.Tanh()

        self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)

    def forward(self, encoder_output, decoder_hidden = None):
        # encoder_output: [batch_size, seq_len, hidden_size]
        # decoder_hidden: [batch_size, 1, hidden_size]

        # [batch_size, seq_len, hidden_size]
        encoder_output = self.W(encoder_output)

        # [batch_size, 1, hidden_size]
        decoder_hidden = self.W(decoder_hidden)

        # [batch_size, seq_len, 1]
        attn_score = self.V(self.tanh(encoder_output + decoder_hidden))

        # [batch_size, seq_len]
        attn_score = attn_score.squeeze(-1)

        # [batch_size, seq_len]
        attn_weight = F.softmax(attn_score, dim=-1)

        # [batch_size, 1, seq_len]
        attn_weight = attn_weight.unsqueeze(1)

        # [batch_size, 1, hidden_size]
        context = torch.bmm(attn_weight, encoder_output)

        # [batch_size, 1, hidden_size]
        decoder_output, decoder_hidden = self.rnn(context, decoder_hidden)

        return decoder_output, decoder_hidden, attn_weight
    




class DecoderPrenet(EncoderPrenet):
    def __init__(self, input_dim, output_dim_1, output_dim_2):
        super().__init__(input_dim, output_dim_1, output_dim_2)

    def forward(self, x):
        return super().forward(x)
    






    
 



if __name__ == "__main__":

    MAX_LEN = 30

    text = "Hello World"

    vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',.!?;: "

    vocab_size = len(vocab)

    input_id = [lookup[s] for s in text if s in alphabet]
    input_id = input_id + [0] * (MAX_LEN - len(input_id))

    input_id = torch.tensor(input_id).unsqueeze(0)

    embedding = CEmbedding(vocab_size, out_embd=256)

    encoder = Encoder(input_dim=256, output_dim_1=128, output_dim_2=128)

    output = embedding(input_id)

 
    output = encoder(output)

    attention = Attention(256, 256)

    decoder_hidden = torch.zeros(1, 256)

    context = attention(output, decoder_hidden)

    print(context)


    


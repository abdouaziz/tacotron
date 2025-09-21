import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import BahdanauAttention, AttentionWrapper, get_mask_from_lengths
from torch.autograd import Variable

from dataset import TTSDataset , TTSCollator






class Embedding(nn.Module):
    def __init__(self, n_symbols, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(n_symbols, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim ** -0.5)
        
    def forward(self, x):
        return self.embedding(x)
    

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        
        nn.init.xavier_uniform_(
            self.linear.weight, gain=nn.init.calculate_gain(w_init_gain)
        )
        
    def forward(self, x):
        return self.linear(x)
    
class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, bias=True, w_init_gain="linear"):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )
        
    def forward(self, x):
        return self.conv(x)
    

class Encoder(nn.Module):
    def __init__(self, char_dim, embedding_dim=512, conv_channels=512, conv_kernel_size=5, conv_layers=3, lstm_hidden_size=256):
        super(Encoder, self).__init__()
        self.embedding = Embedding(char_dim, embedding_dim)
        self.convs = nn.ModuleList()
        for _ in range(conv_layers):
            conv_layer = ConvNorm(
                in_channels=embedding_dim,
                out_channels=conv_channels,
                kernel_size=conv_kernel_size,
                stride=1,
                padding=(conv_kernel_size - 1) // 2,
                w_init_gain="relu",
            )
            self.convs.append(nn.Sequential(conv_layer, nn.BatchNorm1d(conv_channels), nn.ReLU()))
        
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
    def forward(self, x, input_lengths=None):
        x = self.embedding(x).transpose(1, 2)  # (B, embedding_dim, T)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(x)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs  # (B, T, 2 * lstm_hidden_size)
    
    

if __name__=="__main__":


    from torch.utils.data import DataLoader

    ds = TTSDataset(name_or_path="abdouaziiz/alffa", split="train+validation+test")

    #train_sampler = BatchSampler(ds, batch_size=1 )

    loader = DataLoader(ds,batch_size=1, collate_fn=TTSCollator())
    
    text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask = next(iter(loader)) 

    print(text_padded.shape , mel_padded.shape)


    model = Encoder(
        char_dim=42
    )

    output = model(text_padded)

    print(output)

 
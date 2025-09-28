import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import TTSDataset , TTSCollator 
from torch.utils.data import DataLoader

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
        
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(w_init_gain))
        
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
       
        return outputs  



class Prenet(nn.Module):
    def __init__(self,in_dim, out_dim=[256 ,128]  , dropout_p=0.5):
        super(Prenet,self).__init__()

        self.dropout = dropout_p
        
        in_dim = [in_dim] +out_dim[:-1]

        self.layers = nn.ModuleList([
            nn.Linear(in_features=in_dim , out_features=out_dim) for (in_dim , out_dim) in zip(in_dim , out_dim)
        ])

        self.relu = nn.ReLU() 

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(self.relu(layer(x)), p=self.dropout)
        return x
    

class LocationLayer(nn.Module):
    
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.conv = ConvNorm(
            in_channels=2, 
            out_channels=attention_n_filters, 
            kernel_size=attention_kernel_size, 
            padding="same",
            bias=False
        )
        self.proj = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_weights):
        attention_weights = self.conv(attention_weights).transpose(1,2)
        attention_weights = self.proj(attention_weights)
        return attention_weights


class LocalSensitiveAttention(nn.Module):
    def __init__(self, attention_dim, decoder_hidden_size,encoder_hidden_size, attention_n_filters, attention_kernel_size):
        super(LocalSensitiveAttention, self).__init__()

        self.in_proj = LinearNorm(decoder_hidden_size, attention_dim, bias=True, w_init_gain="tanh")
        self.enc_proj = LinearNorm(encoder_hidden_size, attention_dim, bias=False, w_init_gain="tanh")
        
        self.what_have_i_said = LocationLayer(
            attention_n_filters, 
            attention_kernel_size, 
            attention_dim
        )

        self.energy_proj = LinearNorm(attention_dim, 1, bias=False, w_init_gain="tanh")

        self.reset()

    def reset(self):
        self.enc_proj_cache = None

    def _calculate_alignment_energies(self,mel_input,encoder_output,cumulative_attention_weights, mask=None):

        ### Take our previous step of the mel sequence and project it (B x 1 x attention_dim)
        mel_proj = self.in_proj(mel_input).unsqueeze(1)

        ### Take our entire encoder output and project it (B x encoder_len x attention_dim)
        if self.enc_proj_cache is None:
            self.enc_proj_cache = self.enc_proj(encoder_output)

        ### Look at our attention weight history to understand where the model has already placed attention 
        cumulative_attention_weights = self.what_have_i_said(cumulative_attention_weights)

        ### Broadcast sum the single mel timestep over all of our encoder timesteps (both attention weight features and encoder features)
        ### And scale with tanh to get scores between -1 and 1, and project to a single value to comput energies
        energies = self.energy_proj(torch.tanh(mel_proj + self.enc_proj_cache + cumulative_attention_weights)).squeeze(-1)
        
        ### Mask out pad regions (dont want to weight pad tokens from encoder)
        if mask is not None:
            energies = energies.masked_fill(mask.bool(), -float("inf"))
        
        return energies
    
    def forward(self, mel_input, encoder_output, cumulative_attention_weights, mask=None):

        ### Compute energies ###
        energies = self._calculate_alignment_energies(mel_input,encoder_output,cumulative_attention_weights,mask)
        
        ### Convert to Probabilities (relation of our mel input to all the encoder outputs) ###
        attention_weights = F.softmax(energies, dim=1)

        ### Weighted average of our encoder states by the learned probabilities 
        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)

        return attention_context, attention_weights


 
if __name__=="__main__":



    dataset = TTSDataset(name_or_path="abdouaziiz/alffa")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        collate_fn=TTSCollator()
    )

    text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask  = next(iter(dataloader))

    encoder = Encoder(char_dim=42)

    output1 = encoder(text_padded)

    print(output1.shape)

    prenet = Prenet(in_dim=512)

    output= prenet(output1)

    print(output.shape)


 
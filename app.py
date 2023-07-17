
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim, hidden_dim, attention_dim)
        self.gru = nn.GRUCell(embedding_dim + hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_outputs, decoder_hidden, decoder_input):
        """
        encoder_outputs: [batch_size, seq_len, enc_dim]
        decoder_hidden: [batch_size, dec_dim]
        decoder_input: [batch_size]
        """
        embedded = self.embedding(decoder_input)
        context, attention = self.attention(encoder_outputs, decoder_hidden)
        decoder_input = torch.cat([embedded, context], dim=1)
        decoder_hidden = self.gru(decoder_input, decoder_hidden)
        output = F.log_softmax(self.fc(decoder_hidden), dim=1)
        return output, decoder_hidden, attention
    
class Prenet(nn.Module):
    def __init__(self, input_dim, prenet_dim, prenet_layers):
        super().__init__()
        self.input_dim = input_dim
        self.prenet_dim = prenet_dim
        self.prenet_layers = prenet_layers

        self.layers = nn.ModuleList()
        for i in range(prenet_layers):
            self.layers.append(nn.Linear(input_dim, prenet_dim))
            input_dim = prenet_dim

    def forward(self, x):
        for i in range(self.prenet_layers):
            x = F.dropout(F.relu(self.layers[i](x)), p=0.5, training=True)
        return x







class Tacotron(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        attention_dim,
        prenet_dim,
        prenet_layers,
        max_len,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_dim=embedding_dim,
            output_dim_1=hidden_dim,
            output_dim_2=hidden_dim,
            in_channels=[30, 128],
            out_channels=128,
            projecions=[128, 128],
            kernel_size=[16, 3],
        )
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, attention_dim)
        self.max_len = max_len
        self.prenet = Prenet(embedding_dim, prenet_dim, prenet_layers)

    def forward(self, encoder_inputs, decoder_inputs):
        """
        encoder_inputs: [batch_size, seq_len, embedding_dim]
        decoder_inputs: [batch_size, seq_len]
        """
        encoder_outputs = self.encoder(encoder_inputs)
        decoder_hidden = torch.zeros(encoder_outputs.shape[0], encoder_outputs.shape[2])
        decoder_outputs = []
        for i in range(decoder_inputs.shape[1]):
            decoder_input = decoder_inputs[:, i]
            decoder_output, decoder_hidden, attention = self.decoder(
                encoder_outputs, decoder_hidden, decoder_input
            )
            decoder_outputs.append(decoder_output)
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs
    




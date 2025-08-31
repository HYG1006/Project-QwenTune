import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_pe(embedding: torch.Tensor):
    seq_len, bs, embed_dim = embedding.shape
    position_embedding = torch.zeros(seq_len, embed_dim).to(embedding.device)
    position = torch.arange(0, seq_len, 1, dtype=torch.float32).unsqueeze(1)
    rest = torch.exp(-torch.arange(2, embed_dim+2, 2, dtype=torch.float32) * math.log(10000) / embed_dim)
    position_embedding[:, 0::2] = torch.sin(position * rest)
    position_embedding[:, 1::2] = torch.cos(position * rest)
    position_embedding = position_embedding.unsqueeze(1).repeat(1, bs, 1)
    pe_added_embedding = embedding + position_embedding
    return pe_added_embedding

# class MHA(nn.Module):
#     def __init__(self, dim, num_head, dropout=0.0):
#         super().__init__()
#         self.dim = dim
#         self.d_k = dim // num_head
#         self.num_head = num_head

#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.output_layer = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor, mask=None):
#         seq_len, bs, embed_dim = x.shape
#         q = self.q_proj(x).view(seq_len, bs, self.num_head, self.d_k).permute(2, 1, 0, 3)
#         k = self.k_proj(x).view(seq_len, bs, self.num_head, self.d_k).permute(2, 1, 0, 3)
#         v = self.v_proj(x).view(seq_len, bs, self.num_head, self.d_k).permute(2, 1, 0, 3)
#         attention_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.d_k)
#         if mask is not None:
#             attention_score = attention_score.masked_fill(mask == float('-inf'), float('-inf'))
#         attention = torch.softmax(attention_score, dim=-1)
#         attention = self.dropout(attention)
#         output = torch.matmul(attention, v) # (n_head, bs, seq_len, d_k)
#         output = output.permute(2, 1, 0, 3).contiguous().view(seq_len, bs, self.dim)
#         output = self.output_layer(output)
#         return output

# class FFN(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.0):
#         super().__init__()
#         self.linear1 = nn.Linear(dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(hidden_dim, dim)
    
#     def forward(self, x: torch.Tensor):
#         return self.linear2(self.dropout(F.relu(self.linear1(x))))

# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_head, hidden_dim, dropout=0.0):
#         super().__init__()
#         self.attention = MHA(dim, num_head, dropout)
#         self.ffn = FFN(dim, hidden_dim, dropout)
#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
    
#     def forward(self, x: torch.Tensor, mask=None):
#         attention_output = self.attention(x, mask=mask)
#         x = x + self.dropout(attention_output)
#         x = self.norm1(x)
#         x = x + self.dropout(self.ffn(x))
#         x = self.norm2(x)
#         return x

class LMModel_transformer(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, dim=256, nhead=8, num_layers = 4, dropout=0.2, hidden_size=512):
        super(LMModel_transformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, dim)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you Transformer model here. You can add additional parameters to the function.
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=nhead,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        ########################################

        self.decoder = nn.Linear(dim, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, none_variable=None):
        #print(input.device)
        embeddings = self.drop(self.encoder(input))

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        L = embeddings.size(0)
        src_mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(input.device.type)
        src = embeddings * math.sqrt(self.dim)
        #TODO: use your defined transformer, and use the mask.
        x = add_pe(src)
        output = self.transformer_encoder(x, mask=src_mask)
        ########################################
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), None

class MyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj_layers = nn.ModuleList()
        self.hidden_proj_layers = nn.ModuleList()
        for layer in range(num_layers):
            dim = input_dim if layer == 0 else hidden_dim
            self.input_proj_layers.append(nn.Linear(dim, hidden_dim))
            self.hidden_proj_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
    def forward(self, x, hidden=None):
        # x: (seq_len, bs, input_dim)
        seq_len, batch_size, _ = x.shape
        if hidden is None:
            hidden = x.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        h_prev = hidden
        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            h_current = []
            for layer in range(self.num_layers):
                input_proj = self.input_proj_layers[layer](input_t)
                hidden_proj = self.hidden_proj_layers[layer](h_prev[layer])
                h = torch.tanh(input_proj + hidden_proj)
                h_current.append(h)
                input_t = h
            h_prev = torch.stack(h_current, dim=0)
            outputs.append(h_current[-1])
        output = torch.stack(outputs, dim=0)
        # output: (seq_len, bs, hidden_dim)
        return output, h_prev


class LMModel_RNN(nn.Module):
    """
    RNN-based language model:
    1) Embedding layer
    2) Vanilla RNN network
    3) Output linear layer
    """
    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_RNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        # Construct your RNN model here.
        self.rnn = MyRNN(input_dim=dim, hidden_dim=hidden_size, num_layers=num_layers)
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.dim = dim
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)
        ########################################
        # TODO: use your defined RNN network
        output, hidden = self.rnn(embeddings, hidden)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden




class LMModel_LSTM(nn.Module):
    """
    LSTM-based language model:
    1) Embedding layer
    2) LSTM network
    3) Output linear layer
    """
    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        # Construct your LSTM model here.
        self.lstm = nn.LSTM(input_size = dim,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = dropout,
                            batch_first = False)

        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)

        ########################################
        # TODO: use your defined LSTM network
        output, hidden = self.lstm(embeddings, hidden)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden

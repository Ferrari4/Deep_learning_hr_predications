import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, value, key, query):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        values = value.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        a = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(a / (self.head_dim ** (0.5)), dim=-1)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim)

        out = self.fc_out(out)
        residual = query + self.attention_dropout(out)
        norm_out = self.attention_norm(residual)
        return norm_out

class PositionEmbedding(nn.Module):
    def __init__(self, token_len, input_len, embed_len):
        super(PositionEmbedding, self).__init__()
        self.token_len = token_len
        self.sin_lookup(embed_len, token_len)
    
    def sin_lookup(self, embed_len, token_len):
        table = np.zeros((token_len, embed_len))
        for i in range(token_len):          
            for j in range(embed_len):        
                angle = i / np.power(10000, 2 * (j // 2) / embed_len)
                if j % 2 == 0:
                    table[i, j] = np.sin(angle)  
                else:
                    table[i, j] = np.cos(angle) 
        table = torch.FloatTensor(table).unsqueeze(0)
        self.register_buffer("table", table)

    def forward(self, input_vector):
        assert input_vector.shape[1] == self.token_len, \
            f"expected {self.token_len} tokens, got {input_vector.shape[1]}"
        return input_vector + self.table

class VectorEmbedding(nn.Module):
    def __init__(self, token_len, input_len, embed_len):
        super(VectorEmbedding, self).__init__()
        self.embed = nn.Linear(input_len, embed_len)
        self.token_len = token_len
    
    def forward(self, input):
        input_vector = self.embed(input)
        return input_vector
    
class output_net(nn.Module):
    def __init__(self, embed_len, hidden_len, dropout):
        super().__init__()
        self.w_1 = nn.Linear(embed_len, hidden_len) 
        self.w_2 = nn.Linear(hidden_len, embed_len)
        self.layer_norm = nn.LayerNorm(embed_len, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_len, heads, hidden_len, dropout):
        super(EncoderLayer, self).__init__()
        self.atten = MultiHeadAttention(embed_len, heads, dropout)
        self.out_net = output_net(embed_len, hidden_len, dropout)

    def forward(self, input_vector):
        attention = self.atten(input_vector, input_vector, input_vector)
        output = self.out_net(attention)
        return output

class Encoder(nn.Module):
    def __init__(self, n_layers, heads, hidden_len, embed_len, device, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(n_layers):
            layer = EncoderLayer(embed_len, heads, hidden_len, dropout)
            self.layer_stack.append(layer)
        self.device = device

    def forward(self, input_vector):
        x = self.dropout(input_vector)
        for layer in self.layer_stack:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_len, heads, hidden_len, dropout):
        super(DecoderLayer, self).__init__()
        self.atten = MultiHeadAttention(embed_len, heads, dropout)
        self.out_net = output_net(embed_len, hidden_len, dropout)

    def forward(self, input_encode_vector, input_future_vector):
        attention = self.atten(input_encode_vector, input_encode_vector, input_future_vector)
        output = self.out_net(attention)
        return output

class Decoder(nn.Module):
    def __init__(self, n_layers, heads, hidden_len, embed_len, device, dropout=0.1):
        super(Decoder, self).__init__()
        self.layer_stack = nn.ModuleList()
        for _ in range(n_layers):
            layer = DecoderLayer(embed_len, heads, hidden_len, dropout)
            self.layer_stack.append(layer)
        self.device = device

    def forward(self, encoder_output, query):
        for layer in self.layer_stack:
            query = layer(encoder_output, query)
        return query

class WristForecaster(nn.Module):
    def __init__(self, history_len=40, horizon=16, embed_len=32,
                 heads=4, hidden_len=64, n_layers=2):
        super().__init__()
        self.horizon = horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = VectorEmbedding(history_len, 3, embed_len)
        self.position = PositionEmbedding(history_len, 3, embed_len)

        self.robot_embedding = VectorEmbedding(horizon, 3, embed_len)
        self.robot_position = PositionEmbedding(horizon, 3, embed_len)

        self.encoder = Encoder(n_layers, heads, hidden_len, embed_len, self.device, dropout=0.1)
        self.decoder = Decoder(n_layers, heads, hidden_len, embed_len, self.device, dropout=0.1)

        self.forecast_head = nn.Linear(embed_len, 6)

    def forward(self, history, robot_history):
        batch_size = history.shape[0]
        robot_batch_size = robot_history.shape[0]
        assert batch_size == robot_batch_size

        embedded = self.embedding(history)
        postion_embed = self.position(embedded)
        encoded = self.encoder(postion_embed) # [B, token/history=40, D=32]

        embedded_robot = self.robot_embedding(robot_history)
        position_embed_robot =  self.robot_position(embedded_robot) # [B, 16, D=32]

        decoded = self.decoder(encoded, position_embed_robot)  # [B, token/history=16, D=32]
        
        prediction = self.forecast_head(decoded)
        prediction = prediction.reshape(batch_size, self.horizon, 6)

        mean_offsets = prediction[..., :3]
        raw_variance = prediction[..., 3:]

        mean = history[:, -1:] + mean_offsets
        variance = F.softplus(raw_variance) + 1e-6

        return mean, variance
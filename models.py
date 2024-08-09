import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch_geometric.utils import scatter


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

class Numerator(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_length, dropout=0.1):
        super(Numerator, self).__init__()
        # self.input_layer = nn.Linear(6, d_model)
        self.path_encoding_layer = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.path_reweighting_layer = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.scoring_layer1 = nn.Linear(d_model, 4096)
        self.scoring_layer2 = nn.Linear(4096, 1)
        self.value_encoding_layer = nn.Linear(64, d_model)
        # self.output_layer_1 = nn.Linear(d_model, 4096)
        # self.output_layer_2 = nn.Linear(4096, 1)
        # self.path_length_encoding_layer = nn.Linear(1, d_model)
        self.output_layer_1 = nn.Linear(d_model, int(d_model * 2))
        self.output_layer_2 = nn.Linear(int(d_model * 2), 1)

        
    def forward(self, src, value, reg_token_indices, path_counts):
        value_embedding = self.value_encoding_layer(value)
        value_embedding = value_embedding
        path_embedding = self.path_encoding_layer(src)
        path_embedding = path_embedding[torch.arange(reg_token_indices.shape[0]), reg_token_indices]
        value_embedding = path_embedding * value_embedding
        output = self.output_layer_1(value_embedding)
        output = self.output_layer_2(F.relu(output))
        query_counts = torch.bincount(path_counts)[0]
        # path_length_embedding = self.path_length_encoding_layer(reg_token_indices.unsqueeze(-1).float())
        path_embedding = path_embedding.reshape(-1, query_counts, path_embedding.shape[-1])
        # path_length_embedding = path_length_embedding.reshape(-1, query_counts, path_length_embedding.shape[-1])
        # path_embedding = path_embedding + path_length_embedding
        path_scores = self.path_reweighting_layer(path_embedding)
        path_scores = self.scoring_layer1(path_scores)
        path_scores = self.scoring_layer2(F.relu(path_scores))
        path_scores = F.softmax(path_scores.squeeze(), dim=1)
        # output = output[:, reg_token_indices]
        return output, path_scores.reshape(-1, 1)
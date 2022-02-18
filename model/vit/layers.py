import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, in_features:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        # in_features and head for Multi-Head Attention
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(in_features) # LayerNorm is BatchNorm for NLP
        self.msa = MultiHeadAttention(in_features, head=head, dropout=dropout)
        self.norm2 = nn.LayerNorm(in_features)
        # Position-wise Feed-Forward Networks with GELU activation functions
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, in_features),
            nn.GELU(),
        )
        # Position-wise Feed-Forward Networks (same as one in NLP Transformer model)
        '''
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, in_features)
        )
        # MLP using GELU
        '''

    def forward(self, input):
        out = self.msa(self.norm1(input)) + input # add residual connection
        out = self.mlp(self.norm2(out)) + out # add another residual connection
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, features:int, head:int=8, dropout:float=0.):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.features = features
        self.sqrt_d = self.features**0.5

        self.q = nn.Linear(features, features)
        self.k = nn.Linear(features, features)
        self.v = nn.Linear(features, features)

        self.o = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        b, n, f = input.size()
        q = self.q(input).view(b, n, self.head, self.features//self.head).transpose(1,2)
        k = self.k(input).view(b, n, self.head, self.features//self.head).transpose(1,2)
        v = self.v(input).view(b, n, self.head, self.features//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

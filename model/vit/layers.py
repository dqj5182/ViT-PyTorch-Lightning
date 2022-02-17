import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, features:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(features)
        self.msa = MultiHeadAttention(features, head=head, dropout=dropout)
        self.norm2 = nn.LayerNorm(features)
        self.mlp = nn.Sequential(
            nn.Linear(features, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        out = self.msa(self.norm1(input)) + input
        out = self.mlp(self.norm2(out)) + out
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

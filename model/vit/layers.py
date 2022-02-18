import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module): # Done
    def __init__(self, input_dim:int, mlp_hidden_dim:int, num_head:int=8, dropout:float=0.):
        # input_dim and head for Multi-Head Attention
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim) # LayerNorm is BatchNorm for NLP
        self.msa = MultiHeadSelfAttention(input_dim, num_head=num_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(input_dim)
        # Position-wise Feed-Forward Networks with GELU activation functions
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, input_dim),
            nn.GELU(),
        )
        # Position-wise Feed-Forward Networks (same as one in NLP Transformer model)
        '''
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, input_dim)
        )
        # MLP using GELU
        '''

    def forward(self, input):
        out = self.msa(self.norm1(input)) + input # add residual connection
        out = self.mlp(self.norm2(out)) + out # add another residual connection
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim:int, num_head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.input_dim = input_dim
        self.sqrt_dim = self.input_dim**0.5

        # Query
        self.q = nn.Linear(input_dim, input_dim)
        # Key
        self.k = nn.Linear(input_dim, input_dim)
        # Value
        self.v = nn.Linear(input_dim, input_dim)

        self.o = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        b, n, f = input.size()
        q = self.q(input).view(b, n, self.num_head, self.input_dim//self.num_head).transpose(1,2)
        k = self.k(input).view(b, n, self.num_head, self.input_dim//self.num_head).transpose(1,2)
        v = self.v(input).view(b, n, self.num_head, self.input_dim//self.num_head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_dim, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

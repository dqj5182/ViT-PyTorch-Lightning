import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module): # Done
    def __init__(self, img_size:int, patch_size:int, in_chans:int=3, emb_dim:int=48):
        """
        img_size: 1d size of each image (32 for CIFAR-10)
        patch_size: 1d size of each patch (img_size/num_patch_1d, 4 in this experiment)
        in_chans: input channel (3 for RGB images)
        emb_dim: flattened length for each token (or patch)
        """
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, 
            emb_dim, 
            kernel_size = patch_size, 
            stride = patch_size
        )

    def forward(self, x):
        with torch.no_grad():
            # x: [batch, in_chans, img_size, img_size]
            x = self.proj(x) # [batch, embed_dim, # of patches in a row, # of patches in a col], [batch, 48, 8, 8] in this experiment
            x = x.flatten(2) # [batch, embed_dim, total # of patches], [batch, 48, 64] in this experiment
            x = x.transpose(1, 2) # [batch, total # of patches, emb_dim] => Transformer encoder requires this dimensions [batch, number of words, word_emb_dim]
        return x


class TransformerEncoder(nn.Module): # Done
    def __init__(self, input_dim:int, mlp_hidden_dim:int, num_head:int=8, dropout:float=0.):
        # input_dim and head for Multi-Head Attention
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim) # LayerNorm is BatchNorm for NLP
        self.msa = MultiHeadSelfAttention(input_dim, n_heads=num_head)
        self.norm2 = nn.LayerNorm(input_dim)
        # Position-wise Feed-Forward Networks with GELU activation functions
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, input_dim),
            nn.GELU(),
        )

    def forward(self, input):
        out = self.msa(self.norm1(input)) + input # add residual connection
        out = self.mlp(self.norm2(out)) + out # add another residual connection
        return out


class MultiHeadSelfAttention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim: int
        The input and out dimension of per token features

    n_heads: int
        Number of attention heads
    
    qkv_bias: bool
        If True then we include bias to the query, key, and value projections

    attn_p: float
        Dropout probability applied to the query, key, and value tensors

    proj_p: float
        Droupout probability applied to the output tensor

    Attributes
    ----------
    scale: float
        Normalizing constant for the dot product
    
    qkv: nn.Linear
        Linear projection for the query, key, and value
    
    proj: nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space

    attn_drop, proj_drop: nn.Linear
        Dropout Layers
    """
    def __init__(self, dim:int, n_heads:int=8, qkv_bias:bool=True, attn_p:float=0., proj_p:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads # Preparing for concat
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        """Run forward pass

        Parameters
        ----------
        x: torch.Tensor
            Shape '(n_samples, n_patches + 1, dim)'

        Returns
        -------
        torch.Tensor
            Shape '(n_samples, n_patches + 1, dim)'
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim: # To make sure concatnated dim (output dim) is same as input dim
            raise ValueError
        
        # Linear Layers for Query, Key, Value
        qkv = self.qkv(x) # (n_samples, n_patches+1, 3*dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) # (n_samples, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, n_samples, n_heads, n_patches+1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (n_samples, n_heads, n_patches+1, head_dim)

        # Scaled Dot-Product Attention
        k_t = k.transpose(-2, -1) # K Transpose: (n_samples, n_heads, head_dim, n_patches+1)
        dot_product = (q @ k_t)*self.scale # Query, Key Dot Product with Scale Factor: (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = dot_product.softmax(dim=-1) # Softmax: (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_drop(attn) # Attention Dropout: (n_samples, n_heads, n_patches+1, n_patches+1)
        weighted_avg = attn @ v # (n_samples, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (n_samples, n_patches+1, n_heads, head_dim)

        # Concat and Last Linear Layer
        weighted_avg = weighted_avg.flatten(2) # Concat: (n_samples, n_patches+1, dim)
        x = self.proj(weighted_avg) # Last Linear Layer: (n_samples, n_patches+1, dim)
        x = self.proj_drop(x) # Last Linear Layer Dropout: (n_samples, n_patches+1, dim)

        return x









"""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim:int, num_head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.input_dim = input_dim
        self.sqrt_dim = self.input_dim**0.5 # sqrt(d_k): scaling factor to prevent gradient vanishing

        # Linear Layer for Value
        self.ll_value = nn.Linear(input_dim, input_dim)
        # Linear Layer for Key
        self.ll_key = nn.Linear(input_dim, input_dim)
        # Linear Layer for Query
        self.ll_query = nn.Linear(input_dim, input_dim)

        # Last Linear Layer
        self.lll = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout) # Dropout for Last Linear Layer

    def forward(self, input):
        batch, num_token, emb_dim = input.size() # batch: batch, num_token: number of words (or patches), emb_dim: embedding (feature) dimensions, [b, n, f]
        
        # Linear Layer for Value, Key, Query
        query = self.ll_query(input).view(batch, num_token, self.num_head, self.input_dim//self.num_head).transpose(1,2) # [batch, self.num_head, num_token, self.input_dim//self.num_head]
        key = self.ll_key(input).view(batch, num_token, self.num_head, self.input_dim//self.num_head).transpose(1,2) # [batch, self.num_head, num_token, self.input_dim//self.num_head]
        value = self.ll_value(input).view(batch, num_token, self.num_head, self.input_dim//self.num_head).transpose(1,2) # [batch, self.num_head, num_token, self.input_dim//self.num_head]

        # Scaled Dot-Product Attention
        attn_score = F.softmax(torch.einsum("bhtf, bhjf->bhtj", query, key)/self.sqrt_dim, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", attn_score, value) #(b,n,h,f//h)

        # Concat and Linear
        out = self.dropout(self.lll(attn.flatten(2)))
        return out
"""

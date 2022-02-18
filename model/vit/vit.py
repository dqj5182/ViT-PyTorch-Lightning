import torch
import torch.nn as nn

from model.vit.layers import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, num_patch_1d:int=8, dropout:float=0., num_layers:int=7, hidden_dim:int=384, mlp_hidden_dim:int=384*4, num_head:int=8, is_cls_token:bool=True):
        super(ViT, self).__init__()
        self.is_cls_token = is_cls_token
        self.num_patch_1d = num_patch_1d # number of patches in one row (or col), 3 in Figure 1 of the paper, 8 in this experiment
        self.patch_size = img_size//self.num_patch_1d # 1d size (size of row or col) of each patch, 16 for ImageNet in the paper, 4 in this experiment
        flattened_patch_dim = (img_size//self.num_patch_1d)**2*3 # 48 # Flattened vec length for each patch (4 x 4 x 3, each side is 4 and 3 color scheme)
        num_tokens = (self.num_patch_1d**2)+1 if self.is_cls_token else (self.num_patch_1d**2) # number of total patches + 1 (class token), 10 in the paper, 65 in this experiment

        # Linear Projection of Flattened Patches
        self.lpfp = nn.Linear(flattened_patch_dim, hidden_dim)

        # Patch + Position Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) if is_cls_token else None # learnable classification token with dim [1, 1, 384]
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)) # learnable positional embedding with dim [1, 65, 384]
        
        # Transformer Encoder
        enc_list = [TransformerEncoder(hidden_dim, mlp_hidden_dim=mlp_hidden_dim, dropout=dropout, num_head=num_head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list) # * should be adeed if given regular python list to nn.Sequential
        
        # MLP Head
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes) # for cls_token
        )

    def forward(self, x):
        out = self._to_words(x)
        out = self.lpfp(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.num_patch_1d**2 ,-1)
        return out

import torch
import torch.nn as nn

from model.vit.layers import PatchEmbedding, TransformerEncoder

class ViT(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, num_patch_1d:int=8, dropout:float=0., num_enc_layers:int=7, hidden_dim:int=384, mlp_hidden_dim:int=384*4, num_head:int=8, is_cls_token:bool=True):
        super(ViT, self).__init__()
        self.is_cls_token = is_cls_token
        self.num_patch_1d = num_patch_1d # number of patches in one row (or col), 3 in Figure 1 of the paper, 8 in this experiment
        self.patch_size = img_size//self.num_patch_1d # 1d size (size of row or col) of each patch, 16 for ImageNet in the paper, 4 in this experiment
        flattened_patch_dim = (img_size//self.num_patch_1d)**2*3 # 48 # Flattened vec length for each patch (4 x 4 x 3, each side is 4 and 3 color scheme)
        num_tokens = (self.num_patch_1d**2)+1 if self.is_cls_token else (self.num_patch_1d**2) # number of total patches + 1 (class token), 10 in the paper, 65 in this experiment

        # Divide each image into patches
        self.images_to_patches = PatchEmbedding(
                                    img_size=img_size, 
                                    patch_size=img_size//num_patch_1d
                                )

        # Linear Projection of Flattened Patches
        self.lpfp = nn.Linear(flattened_patch_dim, hidden_dim) # 48 x 384

        # Patch + Position Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) if is_cls_token else None # learnable classification token with dim [1, 1, 384]. 1 in 2nd dim because there is only one class per each image not each patch
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)) # learnable positional embedding with dim [1, 65, 384]
        
        # Transformer Encoder
        enc_list = [TransformerEncoder(hidden_dim, mlp_hidden_dim=mlp_hidden_dim, dropout=dropout, num_head=num_head) for _ in range(num_enc_layers)] # num_enc_layers is L in Transformer Encoder at Figure 1
        self.enc = nn.Sequential(*enc_list) # * should be adeed if given regular python list to nn.Sequential
        
        # MLP Head (Standard Classifier)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Images into Patches (including flattening)
        out = self.images_to_patches(x)
        # Linear Projection on Flattened Pathces
        out = self.lpfp(out)
        # Add Class Token and Positional Embedding
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        # Transformer Encoder
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        # MLP Head
        out = self.mlp_head(out)
        return out
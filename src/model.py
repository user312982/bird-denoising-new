import torch
import torch.nn as nn
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTVS_Encoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head=dim//heads, mlp_dim=mlp_dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        return self.transformer(x)

class ViTVS_Decoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels=1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        patch_dim = channels * patch_size ** 2

        self.decoder_projection = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim)
        )
        self.to_pixels = nn.Linear(patch_dim, patch_size * patch_size)

    def forward(self, x):
        x = self.decoder_projection(x)
        x = self.to_pixels(x)
        # Unfold 1D back to 2D Image
        h = w = self.image_size // self.patch_size
        x = rearrange(x, 'b (h w) (p1 p2) -> b 1 (h p1) (w p2)', h=h, w=w, p1=self.patch_size, p2=self.patch_size)
        return x

class ViTVS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ViTVS_Encoder(
            image_size=config.IMAGE_SIZE, patch_size=config.PATCH_SIZE, 
            dim=config.DIM, depth=config.DEPTH, heads=config.HEADS, mlp_dim=config.MLP_DIM
        )
        self.decoder = ViTVS_Decoder(image_size=config.IMAGE_SIZE, patch_size=config.PATCH_SIZE, dim=config.DIM)

    def forward(self, img):
        encoded = self.encoder(img)
        decoded = self.decoder(encoded)
        # Sigmoid for binary mask probabilities [0, 1]
        return torch.sigmoid(decoded)

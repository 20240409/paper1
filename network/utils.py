import torch
from torch import nn
import numpy as np


def expand_with_masked_zeros(x_encoded, masked_len=40):
    B, kept_len, D = x_encoded.shape

    masked_zeros = torch.zeros((B, masked_len, D), device=x_encoded.device, dtype=x_encoded.dtype)

    x_dec_input = torch.cat([masked_zeros, x_encoded], dim=1)

    return x_dec_input
def one_dimension_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb = np.zeros((pos.shape[0], embed_dim))
    emb[:, 0::2] = np.sin(out)
    emb[:, 1::2] = np.cos(out)
    return emb


def two_dimension_pos_embed(embed_dim, coordination, cls_token=False):
    assert embed_dim % 2 == 0

    emb_x = one_dimension_pos_embed(embed_dim // 2, coordination[0])

    emb_y = one_dimension_pos_embed(embed_dim // 2, coordination[1])

    pos_embed = np.concatenate([emb_x, emb_y], axis=1)
    if cls_token:
        pos_embed = np.concatenate([pos_embed, np.zeros([1, embed_dim])], axis=0)
    return pos_embed

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_mask=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_mask = attn_mask

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, act_layer=nn.GELU):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_features = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class Layer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(input_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, attn_mask=None):
        x = self.norm1(x)
        x = x + self.attn(x, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


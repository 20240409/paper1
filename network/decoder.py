from typing import Optional
import torch
from torch import nn
import numpy as np
from network.utils import Layer, two_dimension_pos_embed, expand_with_masked_zeros


class Decoder(nn.Module):
    def __init__(
            self,
            channel_num: int = 23,
            origin_channel: int = 17,
            encoder_dim: int = 16,
            depth: int = 2,
            num_heads: int = 8,
            mlp_ratio: float = 4.,
            regions: int = 6,
            attn_mask: Optional[torch.Tensor] = None,
            pe_coordination: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.origin_channel = origin_channel
        self.embed_dim = encoder_dim
        self.pe_coordination = pe_coordination
        self.regions = regions
        self.channel_num = channel_num

        self.attn_mask = attn_mask
        self.pos_embed = torch.zeros(self.channel_num, encoder_dim, requires_grad=False)

        norm_layer = nn.LayerNorm

        self.liner1 = nn.Linear(self.channel_num, origin_channel)
        self.liner2 = nn.Linear(encoder_dim, 5)
        self.blocks = nn.ModuleList(
            Layer(encoder_dim, num_heads, mlp_ratio=mlp_ratio,
                  norm_layer=norm_layer) for _ in range(depth))

        self.norm = norm_layer(encoder_dim)
        self.fc_norm = norm_layer(encoder_dim)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = two_dimension_pos_embed(self.embed_dim, self.pe_coordination, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.tensor):
        x = expand_with_masked_zeros(x, masked_len=self.origin_channel)
        self.pos_embed = self.pos_embed.to(x.device)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x, self.attn_mask.to(x.device))

        x = self.liner2(x)
        x = x.permute(0, 2, 1)
        x = self.liner1(x)
        x = x.permute(0, 2, 1)

        return x

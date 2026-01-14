from typing import Optional
import torch
from torch import nn
import numpy as np
from network.utils import Layer,two_dimension_pos_embed
from network.RegionAggregator import RegionAggregator

class Encoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 5,
        channel_num: int = 23,
        encoder_dim: int = 16,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio : float = 4.,
        regions:int=6,
        func_area: Optional[list] = None,
        aggregation_type: Optional[str] = None,
        attn_mask: Optional[torch.Tensor]= None,
        pe_coordination: Optional[np.ndarray]= None,
        is_reconstruction:Optional[bool]=True
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.regions=regions
        self.pe_coordination =pe_coordination
        self.is_reconstruction=is_reconstruction
        self.attn_mask = attn_mask
        self.patch_embed = nn.Linear(in_chans, encoder_dim, bias=True)

        if is_reconstruction:
            self.pos_embed = torch.zeros(channel_num , encoder_dim,requires_grad=True)
        else:
            self.pos_embed = torch.zeros(channel_num + 1, encoder_dim,requires_grad=True)
        norm_layer = nn.LayerNorm
        self.cls_token = nn.parameter.Parameter(torch.zeros(1,1,encoder_dim),requires_grad=True)
        torch.nn.init.normal_(self.cls_token)

        self.blocks = nn.ModuleList(
            Layer(encoder_dim,num_heads,mlp_ratio=mlp_ratio,
                norm_layer=norm_layer) for _ in range(depth))

        self.aggregation_method = RegionAggregator(func_area=func_area,aggregation_type=aggregation_type,encoder_dim=5)
        self.norm = norm_layer(encoder_dim)
        self.fc_norm = norm_layer(encoder_dim)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = two_dimension_pos_embed(self.encoder_dim, self.pe_coordination,cls_token=not self.is_reconstruction)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed))

        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x:torch.tensor):
        if self.aggregation_method is not None:
            x=self.aggregation_method(x)
        x = self.patch_embed(x)
        self.pos_embed=self.pos_embed.to(x.device)
        if self.is_reconstruction:
            x = x + self.pos_embed[:, :]

        else:
            x = x + self.pos_embed[:-1, :]
            cls = torch.repeat_interleave(self.cls_token + self.pos_embed[-1:, :], x.shape[0], 0)
            x = torch.cat([x, cls], dim=1)

        for blk in self.blocks:
            x = blk(x,self.attn_mask.to(x.device))

        if self.is_reconstruction:
            return x[:, -(self.regions):]
        return x[:, -(self.regions + 1):]



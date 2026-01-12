import torch
from typing import Optional
from network.encoder import Encoder
import numpy as np
from torch import nn

class Stage2(nn.Module):
    def __init__(self,
                 channel_num=21,
                 encoder_dim=16,
                 depth: int = 6,
                 num_heads: int = 4,
                 regions=6,
                 func_area: list = None,
                 aggregation_type: str = None,
                 attn_mask: Optional[torch.Tensor] = None,
                 pe_coordination: Optional[np.ndarray] = None,
                 num_class=3):
        super().__init__()
        self.channel_num = channel_num
        self.attn_mask = attn_mask
        self.pe_coordination = pe_coordination
        self.depth = depth
        self.encoder_dim = encoder_dim
        self.regions = regions
        self.num_class = num_class

        self.encoder = Encoder(pe_coordination=pe_coordination, attn_mask=attn_mask, encoder_dim=encoder_dim,
                               depth=depth, num_heads=num_heads, regions=regions, channel_num=channel_num,
                               is_reconstruction=False, func_area=func_area, aggregation_type=aggregation_type)

        self.fc_norm1 = nn.BatchNorm1d(self.encoder_dim * (self.regions + 1))
        self.fc_out1 = nn.Linear(self.encoder_dim * (self.regions + 1), 64)
        self.fc_norm2 = nn.BatchNorm1d(64)
        self.fc_out2 = nn.Linear(64, num_class)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.fc_norm1(x)
        x = self.fc_out1(x)
        x = self.fc_norm2(x)
        x = self.fc_out2(x)

        return x
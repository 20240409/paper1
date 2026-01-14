from typing import Optional
import torch
from torch import nn
from network.encoder import Encoder
from network.decoder import Decoder
import numpy as np

"""
The following code shows the framework of the first stage in our training method,Note that the classifier used in the first stage is different from the one used in the second stage.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Classifier(nn.Module):
    def __init__(self,
                 encoder_dim=16,
                 regions=6,
                 num_class=3):
        super().__init__()

        self.fc_norm1 = nn.BatchNorm1d(encoder_dim * (regions))
        self.fc_out1 = nn.Linear(encoder_dim * (regions), 64)
        self.fc_norm2 = nn.BatchNorm1d(64)
        self.fc_out2 = nn.Linear(64, num_class)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.fc_norm1(x)
        x = self.fc_out1(x)
        x = self.fc_norm2(x)
        x = self.fc_out2(x)
        return x


class Stage1(nn.Module):
    def __init__(self,
                 channel_num: int = 23,
                 origin_channel: int = 17,
                 encoder_dim: int = 16,
                 num_heads: int = 8,
                 regions: int = 6,
                 depth: int = 4,
                 func_area: list = None,
                 aggregation_type: str = None,
                 attn_mask: Optional[torch.Tensor] = None,
                 pe_coordination: Optional[np.ndarray] = None,
                 ):
        super().__init__()
        self.attn_mask = attn_mask
        self.pe_coordination = pe_coordination

        self.encoder = Encoder(attn_mask=attn_mask, pe_coordination=pe_coordination, encoder_dim=encoder_dim,
                               depth=depth, num_heads=num_heads, regions=regions,
                               channel_num=channel_num, is_reconstruction=True, func_area=func_area,
                               aggregation_type=aggregation_type)
        self.classifier = Classifier(encoder_dim=encoder_dim, regions=regions, num_class=3).to(device)

        self.decoder = Decoder(attn_mask=attn_mask, pe_coordination=pe_coordination, encoder_dim=encoder_dim,
                               origin_channel=origin_channel,
                               depth=depth, num_heads=num_heads, channel_num=channel_num)

    def forward(self, x):
        x = self.encoder(x)
        y = self.pre_tune(x)
        x = self.decoder(x)

        return [y, x]



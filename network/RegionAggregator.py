import torch
from torch import nn
import numpy as np

class RegionAggregator(nn.Module):
    def __init__(self, func_area, embed_dim, aggregation_type="prototype-attention"):
        super().__init__()
        self.func_area = func_area
        self.aggregation_type = aggregation_type
        self.embed_dim = embed_dim
        self.region_nodes = len(func_area)
        self.raw_nodes = sum(len(g) for g in func_area)

        if aggregation_type == "prototype-attention":
            self.region_prototypes = nn.Parameter(
                torch.randn(self.region_nodes, embed_dim)
            )

        elif aggregation_type == "self-attention":
            self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, data: torch.Tensor):
        B, total_nodes, C = data.shape
        raw_nodes = self.raw_nodes
        region_nodes = self.region_nodes
        device = data.device

        x_raw = data[:, :raw_nodes, :]
        x_region_init = data[:, raw_nodes:, :]

        x_region_updated = torch.zeros_like(x_region_init, device=device)


        if self.aggregation_type == "prototype-attention":
            for r, idx_list in enumerate(self.func_area):
                idx = torch.tensor(idx_list, device=device)
                x_region = x_raw[:, idx, :]
                proto = self.region_prototypes[r:r + 1, :]
                sim = torch.matmul(x_region, proto.t()) / np.sqrt(C)
                attn = torch.softmax(sim, dim=1)
                region_feat = torch.sum(attn * x_region, dim=1, keepdim=True)
                x_region_updated[:, r:r + 1, :] = region_feat
                out = torch.cat([x_raw, x_region_updated], dim=1)
                return out

        elif self.aggregation_type == "self-attention":
            for r, idx_list in enumerate(self.func_area):
                idx = torch.tensor(idx_list, device=device)
                x_region = x_raw[:, idx, :]
                q = self.W_q(x_region)
                k = self.W_k(x_region)
                v = self.W_v(x_region)
                attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(C)
                attn = torch.softmax(attn, dim=-1)
                region_feat = torch.matmul(attn, v).mean(dim=1, keepdim=True)
                x_region_updated[:, r:r + 1, :] = region_feat
                out = torch.cat([x_raw, x_region_updated], dim=1)
                return out

        elif self.aggregation_type=="mean":
            return data
        else:
            raise ValueError(f"Invalid aggregation_type")
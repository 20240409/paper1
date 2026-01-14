from typing import Optional
import torch
import numpy as np
import scipy.io as sio
from scipy.stats import rankdata
from torch.utils.data import Dataset

"""
This dataset class initializes three distinct datasets, namely stage1, stage2, and test, designed for the first-stage, second-stage, and testing, respectively.
This class automatically aggregates the raw nodes into region-level nodes using the default mean aggregation.
"""
class SEEDVIG(Dataset):
    def __init__(
            self,
            prefix: str = "./data/", # data storage path
            normalize: Optional[str] = None,
            subject_idx: int = 1,
            dataset_name: Optional[str] = None, # dataset_name: stage1 or stage2 or test
            mask_type: Optional[str] = None, # masking methods: node or band or random
            rand_list: Optional[list] = None, # shuffling order of samples
            func_areas: Optional[list] = None
    ):
        super().__init__()
        self.regions = len(func_areas)
        self.mask_type = mask_type
        self.normalize = normalize
        self.dataset_name = dataset_name
        self.subject_idx = subject_idx
        self.func_areas = func_areas

        if rand_list is None:
            rand_list = list(range(885))
        self.rand_list = rand_list

        file = prefix
        data_path = file + "DE_{}.mat"

        self.data = np.array(
            [
                sio.loadmat(data_path.format(self.subject_idx))["DE_feature"]
            ]
        )
        self.data = self.data[:, :, rand_list, :]
        self.data = self.data.transpose([0, 2, 1, 3])

        if normalize is not None:
            self._normalize(normalize)
        self.data = self.data.transpose([0, 2, 1, 3])

        self.get_coordination()

        label_path = file + "label_{}.mat"
        if self.dataset_name == "stage1":
            self.label_1 = self.data.transpose([0, 2, 1, 3])
            self.label_2 = np.array([
                sio.loadmat(label_path.format(subject_idx))["de_labels"]
            ])
            self.label_2 = self.label_2[:, rand_list, :]

        region_data = self.build_region_nodes(self.data, self.func_areas)
        region_data = region_data.transpose([0, 2, 1, 3])

        self.data = self.data.transpose([0, 2, 1, 3])
        if mask_type:
            self.data = self.node_mask(self.data, mask_type=self.mask_type)

        self.data = np.concatenate([self.data, region_data], axis=2)
        
        if self.dataset_name == "stage2" or self.dataset_name == "test":
            self.label = np.array([
                sio.loadmat(label_path.format(subject_idx))["de_labels"]
            ])
            self.label = self.label[:, rand_list, :]
            
        self._split(self.dataset_name)
        
        if self.dataset_name == "stage1":
            self.data = self.data.reshape(self.data.shape[0] * self.data.shape[1], self.data.shape[2],
                                          self.data.shape[3])
            self.label_1 = self.label_1.reshape(self.label_1.shape[0] * self.label_1.shape[1], self.label_1.shape[2],
                                            self.label_1.shape[3])
            self.label_2 = self.label_2.flatten()
            self.label_2 = np.digitize(self.label_2, [0.35, 0.7])

        if dataset_name == "stage2" or self.dataset_name == "test":
            self.data = self.data.reshape(self.data.shape[0] * self.data.shape[1], self.data.shape[2],
                                          self.data.shape[3])
            self.label = self.label.flatten()
            self.label = np.digitize(self.label, [0.35, 0.7])

    def full_data(self, partial_data, all_channels, partial_channels):
        N, _, T, F = partial_data.shape
        full_data = np.zeros((N, len(all_channels), T, F), dtype=partial_data.dtype)

        for i, ch in enumerate(partial_channels):
            if ch in all_channels:
                idx = all_channels.index(ch)
                full_data[:, idx, :, :] = partial_data[:, i, :, :]
            else:
                print(f"channel{ch} not in raw channel")

        return full_data

    def build_region_nodes(self, data, func_areas):
        B, C, T, F = data.shape
        num_regions = len(func_areas)
        region_data = np.zeros((B, num_regions, T, F), dtype=data.dtype)

        for region_idx, area in enumerate(func_areas):
            region_data[:, region_idx] = data[:, area].mean(axis=1)

        return region_data

    def node_mask(self, data, mask_type="random", mask_ratio=0.4):
        raw_nodes = 17
        B, S, N, F = data.shape
        if mask_type == "node":
            idx = torch.randperm(raw_nodes)[0:int(raw_nodes * mask_ratio)]
            data[:, :, idx, :] = 0

        elif mask_type == "band":
            band_idx = torch.randperm(F)[:int(F * mask_ratio)]
            data[:, :, :, band_idx] = 0

        elif mask_type == "random":
            B, S, N, F = data.shape
            mask = (np.random.rand(N, F) > mask_ratio).astype(np.float32)
            mask = np.broadcast_to(mask, (B, S, N, F))
            data = data * mask

        return data

    def _split(self, dataset_name):
        stage1_size = 531
        stage2_size = 177
        test_size = 177

        if dataset_name == "stage1":
            self.data = self.data[:, :stage1_size]
            self.label_1 = self.label_1[:, :stage1_size]
            self.label_2 = self.label_2[:, :stage1_size]
            self.length = stage1_size

        elif dataset_name == "stage2":
            self.data = self.data[:, stage1_size:stage1_size + stage2_size]
            self.label = self.label[:, stage1_size:stage1_size + stage2_size]
            self.length = stage2_size

        elif dataset_name == "test":
            self.data = self.data[:, stage1_size + stage2_size:]
            self.label = self.label[:, stage1_size + stage2_size:]
            self.length = test_size
        else:
            raise ValueError("dataset_name should be stage1 or stage2 or test")

    
    def _normalize(self, method='minmax'):
        if method == 'minmax':
            for i in range(self.data.shape[0]):
                for j in range(5):
                    minn = np.min(self.data[i, :531, :, j])
                    maxx = np.max(self.data[i, :531, :, j])
                    self.data[i, :, :, j] = (self.data[i, :, :, j] - minn) / (maxx - minn)

        if method == 'gaussian':
            for i in range(self.data.shape[0]):
                for j in range(5):
                    mean = np.mean(self.data[i, :531, :, j])
                    std = np.std(self.data[i, :531, :, j])
                    self.data[i, :, :, j] = (self.data[i, :, :, j] - mean) / std

    def coordination(self):
        coordination =[[-56, 56, -58, 58, -56, 56, -12, 12, -12, 0, 12, -16, 0, 16, -16, 0, 16],
                       [24, 24, 0, 0, -24, -24, -16, -16, -32, -32, -32, -48, -48, -48, -64, -64, -64]]

        return np.array(coordination)

    def get_coordination(self):
        func_areas = self.func_areas
        coordination = self.coordination()

        if self.dataset_name == "stage1":
            attn_mask = torch.full((17 + len(func_areas), 17 + len(func_areas)), True, dtype=torch.bool)
        else:
            attn_mask = torch.full((17 + len(func_areas) + 1, 17 + len(func_areas) + 1), True, dtype=torch.bool)

        for i in range(len(func_areas)):
            for j in range(len(func_areas)):
                attn_mask[17 + i, 17 + j] = False

        for i, func_area in enumerate(func_areas):
            for j in func_area:
                attn_mask[17 + i, j] = False
                attn_mask[j, 17 + i] = False
        if self.dataset_name != "stage1":
            for i in range(17, 17 + len(func_areas) + 1):
                attn_mask[17 + len(func_areas), i] = False
                attn_mask[i, 17 + len(func_areas)] = False
        self.attn_mask = attn_mask
        self.coordination = area_gather(coordination, func_areas)

    def __len__(self):
        if self.dataset_name=="stage1":
            return self.label_1.shape[0]
        return self.label.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float)
        
        if self.dataset_name == "stage1":
            y_1 = torch.tensor(self.label_2[idx], dtype=torch.long)
            y_2 = torch.tensor(self.label_1[idx], dtype=torch.float)
            y=[y_1, y_2]

        else:
            y = torch.tensor(self.label[idx], dtype=torch.long)

        return x, y

    def freeup(self):
        pass

    def load(self):
        pass

def area_gather(coordination, areas):
    supernode_coordination = np.zeros([coordination.shape[0], len(areas)])
    for idx, area in enumerate(areas):
        for i in area:
            for j in range(coordination.shape[0]):
                supernode_coordination[j][idx] += coordination[j][i] / len(area)

    res = np.concatenate((coordination, supernode_coordination), axis=1)
    for i in range(res.shape[0]):
        arr = res[i]
        rank = rankdata(arr, method="dense") - 1
        res[i] = rank
    return res
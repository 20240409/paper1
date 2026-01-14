import numpy as np
import torch
from dataset.SEEDVIG import SEEDVIG
from torch.utils.data import DataLoader
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Optional
from model.stage_2 import Stage2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

def fit(
        func_area: Optional[list] = None,
        batch_size: int = 128,
        depth: int = 4,
        encoder_dim: int = 16,
        num_heads: int = 8,
        num_classes: int = 3,
        aggregation_type: Optional[str] = None,
):
    random_list = np.load("../random_list.npy").tolist()
    regions = len(func_area)
    data_yuanben=[]
    data_raw = []
    data_pred = []
    label_true = []

    for sub in range(23):
        test_dataset = SEEDVIG(dataset_name="test", normalize="minmax", subject_idx=sub, rand_list=random_list,
                               func_areas=func_area)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        attn_mask = test_dataset.attn_mask
        pe_coordination = test_dataset.coordination

        stage2 = Stage2(channel_num=17 + regions, attn_mask=attn_mask, pe_coordination=pe_coordination,
                            encoder_dim=encoder_dim, regions=regions,
                            num_heads=num_heads, depth=depth, num_class=num_classes, func_area=func_area,
                            aggregation_type=aggregation_type).to(device)

        stage2_dict = torch.load("The path to the stage2_dict you saved")
        stage2.load_state_dict(stage2_dict)

        loss_fn = nn.CrossEntropyLoss()
        loss_res = 0
        y_true, y_pred = [], []
        stage2.eval()

        with torch.no_grad():  
            for d, l in test_loader:
                label_true.append(l.cpu().numpy())  
                data_yuanben.append(d.cpu().numpy())
                d = d.to(device)
                l = l.to(device).long()
                output = stage2(d)
                output[0] = output[0].to(torch.float)
                output[1] = output[1].to(torch.float)
                data_raw.append(output[1].cpu().numpy())
                data_pred.append(output[0].cpu().numpy())
                preds_class = torch.argmax(output[0], dim=1)
                y_true.extend(l.cpu().numpy())
                y_pred.extend(preds_class.cpu().numpy())

                loss = loss_fn(output[0], l)
                loss_res += loss

    data_yuanben=np.concatenate(data_yuanben,axis=0)
    data_raw = np.concatenate(data_raw, axis=0)
    data_pred = np.concatenate(data_pred, axis=0)
    label_true = np.concatenate(label_true, axis=0)

    return data_yuanben,data_raw, data_pred, label_true

def plot_tsne_comparison(data_yuanben,data_raw, data_pred, labels):
    data_yuanben_flat = data_yuanben.reshape(data_yuanben.shape[0], -1)
    data_raw_flat = data_raw.reshape(data_raw.shape[0], -1)
    n_samples = min(5000, len(data_raw_flat))
    indices = np.random.choice(len(data_raw_flat), n_samples, replace=False)

    data_yuanben_sampled = data_yuanben_flat[indices]
    data_raw_sampled = data_raw_flat[indices]
    data_pred_sampled = data_pred[indices]
    labels_sampled = labels[indices]

    tsne_yuanben = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    data_tsne_yuanben = tsne_yuanben.fit_transform(data_yuanben_sampled)
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    data_tsne_raw = tsne_raw.fit_transform(data_raw_sampled)
    tsne_pred = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    data_tsne_pred = tsne_pred.fit_transform(data_pred_sampled)
    colors = ['#16EAE0', '#FF2501', '#27E818']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    scatter1 = ax1.scatter(data_tsne_yuanben[:, 0], data_tsne_yuanben[:, 1],
                           c=labels_sampled,
                           cmap=plt.get_cmap('viridis', 3),
                           alpha=0.7, s=15)
    ax1.axis('off')
    scatter2 = ax2.scatter(data_tsne_raw[:, 0], data_tsne_raw[:, 1],
                           c=labels_sampled,
                           cmap=plt.get_cmap('viridis', 3),
                           alpha=0.7, s=15)
    
    ax2.axis('off')
    scatter3 = ax3.scatter(data_tsne_pred[:, 0], data_tsne_pred[:, 1],
                           c=labels_sampled,
                           cmap=plt.get_cmap('viridis', 3),
                           alpha=0.7, s=15)
    ax3.axis('off')
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_ticks([0.333333, 1, 1.666666])
    cbar3.set_ticklabels(['Awake', 'Tired', 'Drowsy'])
    cbar3.ax.tick_params(labelsize=30)  

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    func_area =[[0, 2, 4], [1, 3, 5], [6, 8], [7, 10], [9, 12, 15], [11, 14], [13, 16]]

    result = fit(func_area=func_area, batch_size=128, depth=4, encoder_dim=16,
                 num_heads=8,aggregation_type="prototype-attention")

    if result is not None:
        data_yuanben, data_raw, data_pred, label_true = result
        plot_tsne_comparison(data_yuanben,data_raw, data_pred, label_true)
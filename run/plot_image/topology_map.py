import numpy as np
import torch
from dataset.SEEDVIG import SEEDVIG
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional
import mne
import matplotlib as mpl
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

    data_raw=[]
    data_patch = []
    data_pred = []
    label_true = []

    for sub in [2]:
        test_dataset = SEEDVIG(dataset_name="test", normalize="minmax", subject_idx=sub, rand_list=random_list,
                               func_areas=func_area)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        attn_mask = test_dataset.attn_mask
        pe_coordination = test_dataset.coordination
        class_data = {0: [], 1: [], 2: []}

        for x, y in test_loader:
            for cls in [0, 1, 2]:
                idx = (y == cls).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    class_data[cls].append(x[idx])

        for cls in class_data:
            if len(class_data[cls]) > 0:
                class_data[cls] = torch.cat(class_data[cls], dim=0)  
            else:
                class_data[cls] = torch.empty((0, 17, 5))
        
        for cls in class_data:
            print(f"Class {cls} data shape: {class_data[cls].shape}")

        stage2 = Stage2(channel_num=17 + regions, attn_mask=attn_mask, pe_coordination=pe_coordination,
                            encoder_dim=encoder_dim, regions=regions,
                            num_heads=num_heads, depth=depth, num_class=num_classes, func_area=func_area,
                            aggregation_type=aggregation_type).to(device)

        stage2_dict = torch.load("The path to the stage2_dict you saved")
        stage2.load_state_dict(stage2_dict)
        y_true, y_pred = [], []
        stage2.eval()

        with torch.no_grad():  
            for d, l in test_loader:
                label_true.append(l.cpu().numpy())  
                data_raw.append(d.cpu().numpy())
                d = d.to(device)
                l = l.to(device).long()
                output = stage2(d)
                output[0] = output[0].to(torch.float)
                output[1] = output[1].to(torch.float)
                data_patch.append(output[1].cpu().numpy())
                data_pred.append(output[0].cpu().numpy())
                preds_class = torch.argmax(output[0], dim=1)
                y_true.extend(l.cpu().numpy())
                y_pred.extend(preds_class.cpu().numpy())

    return data_raw,data_patch, data_pred, label_true

if __name__ == "__main__":
    func_areas = [
        [[0, 2, 4], [1, 3, 5], [6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]
    ]

    for i in range(len(func_areas)):
        func_area = func_areas[i]
        data_raw,data_patch, data_pred, label_true = fit(func_area=func_area, batch_size=128, depth=4, encoder_dim=16,
                     num_heads=8, aggregation_type="prototype-attention")

        data_patch = [[abs(x) for x in sub] for sub in data_patch]
        label_array = np.concatenate(label_true, axis=0)  
        data_patch_array = np.concatenate(
            [np.array(sub) for sub in data_patch],
            axis=0
        )

        class0_idx = np.where(label_array == 0)[0]
        class1_idx = np.where(label_array == 1)[0]
        class2_idx = np.where(label_array == 2)[0]

        patch_class0 = data_patch_array[class0_idx,:17]  
        patch_class1 = data_patch_array[class1_idx,:17]
        patch_class2 = data_patch_array[class2_idx,:17]

        sample0 = patch_class0[0]  
        sample1 = patch_class1[7]
        sample2 = patch_class2[1]
        
        sample0_for_plot = sample0.mean(axis=1)  
        sample1_for_plot = sample1.mean(axis=1)
        sample2_for_plot = sample2.mean(axis=1)
        

        ch_names = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
                    'P1', 'Pz', 'P2', 'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
        
        all_samples = np.concatenate([sample0_for_plot, sample1_for_plot, sample2_for_plot])
        vmin, vmax = np.min(all_samples), np.max(all_samples)
        titles = ['Awake', 'Tired', 'Drowsy']
        samples_for_plot = [sample0_for_plot, sample1_for_plot, sample2_for_plot]
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0})

        fig.subplots_adjust(
            left=0.105,  
            right=0.88,  
            bottom=0.15,
            top=0.85
        )

        ims = []
        fig.text(
            0.08, 0.5,  
            'Subject 3',  
            va='center',  
            ha='center',  
            fontsize=14,
        )

        for k, data_plot in enumerate(samples_for_plot):
            im, _ = mne.viz.plot_topomap(
                data_plot,
                info,
                axes=axes[k],
                show=False,
                cmap='RdBu_r',
                vlim=(vmin, vmax),
                contours=6,
                res=256
            )
            ims.append(im)
            axes[k].set_title(titles[k], fontsize=18)

        cax = fig.add_axes([0.90, 0.22, 0.015, 0.50])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=14)

        plt.show()

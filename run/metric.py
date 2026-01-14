from typing import Optional
import torch
from dataset.SEEDVIG import SEEDVIG
import numpy as np
import os
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model.stage_1 import Stage1
from model.stage_2 import Stage2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

def safe_save(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)

def fit(
        func_area: list = None,
        batch_size: int = 128,
        depth: int = 4,
        encoder_dim: int = 16,
        num_heads: int = 8,
        num_classes: int = 3,
        random_list: Optional[list] = None,
        mask_type: Optional[str] = None,
        aggregation_type: Optional[str] = None,
):
    regions = len(func_area)
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    for sub in range(23):
        stage1_dataset = SEEDVIG(dataset_name="stage1", normalize="minmax", mask_type=mask_type, rand_list=random_list,
                                 subject_idx=sub, func_areas=func_area)
        stage1_loader = DataLoader(stage1_dataset, batch_size=batch_size, shuffle=True)

        attn_mask = stage1_dataset.attn_mask
        pe_coordination = stage1_dataset.coordination

        stage1_net = Stage1(channel_num=17 + regions, encoder_dim=encoder_dim, depth=depth, num_heads=num_heads,
                            attn_mask=attn_mask,
                            pe_coordination=pe_coordination, regions=regions, func_area=func_area,
                            aggregation_type=aggregation_type).to(device)

        optimizer = optim.Adam(stage1_net.parameters(), lr=0.005)
        loss_fn_1 = nn.CrossEntropyLoss()
        loss_fn_2 = nn.MSELoss()

        epoch = 100
        loss_res = 0
        for epo in range(epoch):
            loss_res = 0
            stage1_net.train()
            for d, l in stage1_loader:
                d = d.to(device)
                l[0] = l[0].to(device)
                l[1] = l[1].to(device)
                output = stage1_net(d)
                output[0] = output[0].to(device).to(torch.float)
                output[1] = output[1].to(device).to(torch.float)
                loss_1 = loss_fn_1(output[0], l[0])
                loss_2 = loss_fn_2(output[1], l[1])

                norm_loss1 = loss_1 / (loss_1.detach() + loss_2.detach())
                norm_loss2 = loss_2 / (loss_1.detach() + loss_2.detach())

                loss = norm_loss1 * loss_1 + norm_loss2 * loss_2
                loss_res += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            safe_save(stage1_net.encoder.state_dict(),
                      f"save_all/{mask_type}_{aggregation_type}/stage1_dict/best_dict_{sub}.pth")
        print(f"Successfully saved stage1 parameters to stage1_dict.    loss:{loss_res}")

        stage2_dataset = SEEDVIG(dataset_name="stage2", normalize="minmax", subject_idx=sub, rand_list=random_list,
                                 func_areas=func_area)
        stage2_loader = DataLoader(stage2_dataset, batch_size=batch_size, shuffle=True)
        attn_mask = stage2_dataset.attn_mask
        pe_coordination = stage2_dataset.coordination

        test_dataset = SEEDVIG(dataset_name="test", normalize="minmax", subject_idx=sub, rand_list=random_list,
                               func_areas=func_area)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        stage2 = Stage2(channel_num=17 + regions, attn_mask=attn_mask, pe_coordination=pe_coordination,
                        encoder_dim=encoder_dim, regions=regions,
                        num_heads=num_heads, depth=depth, num_class=num_classes, func_area=func_area,
                        aggregation_type=aggregation_type).to(device)

        stage1_dict = torch.load(
            f"save_all/{mask_type}_{aggregation_type}/stage1_dict/best_dict_{sub}.pth")
        stage2.encoder.load_state_dict(stage1_dict)

        for parm in stage2.encoder.parameters():
            parm.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, stage2.parameters()), lr=0.005)
        loss_fn = nn.CrossEntropyLoss()
        epoch = 50
        best_metrics = (0, 0, 0, 0)
        for epo in range(epoch):
            loss_res = 0
            correct = 0

            stage2.train()
            for d, l in stage2_loader:
                d = d.to(device)
                l = l.to(device).long()
                output = stage2(d).to(torch.float)
                pred_class = torch.argmax(output, dim=1)
                correct += (pred_class == l).sum().item()
                loss = loss_fn(output, l)
                loss_res += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            loss_res = 0

            stage2.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for d, l in test_loader:
                    d = d.to(device)
                    l = l.to(device).long()
                    output = stage2(d).to(torch.float)
                    preds_class = torch.argmax(output, dim=1)
                    y_true.extend(l.cpu().numpy())
                    y_pred.extend(preds_class.cpu().numpy())

                    loss = loss_fn(output, l)
                    loss_res += loss

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            best_metrics = (acc, prec, rec, f1)
        safe_save(stage2.state_dict(),
                  f"save_all/{mask_type}_{aggregation_type}/best_dict/best_dict_{sub}.pth")

        print(
            f"subject:{sub}   best_acc:{best_metrics[0]:.4f}  prec:{best_metrics[1]:.4f}  rec:{best_metrics[2]:.4f}  f1:{best_metrics[3]:.4f}  total_loss:{loss_res:.4f}")

        all_acc.append(best_metrics[0])
        all_prec.append(best_metrics[1])
        all_rec.append(best_metrics[2])
        all_f1.append(best_metrics[3])

    avg_acc = np.mean(all_acc)
    avg_prec = np.mean(all_prec)
    avg_rec = np.mean(all_rec)
    avg_f1 = np.mean(all_f1)

    print(f"\n acc={avg_acc:.4f}, prec={avg_prec:.4f}, rec={avg_rec:.4f}, f1={avg_f1:.4f}\n")

    return avg_acc, avg_prec, avg_rec, avg_f1

def run(
        mask_type="node",
        aggregation_type="prototype-attention",
):
    func_area = [[0, 2, 4], [1, 3, 5], [6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]
    random_list = np.load("./random_list.npy").tolist()

    avg_acc, avg_prec, avg_rec, avg_f1 = fit(func_area=func_area, batch_size=128,
                                                          depth=4,
                                                          encoder_dim=16, num_heads=8,
                                                          random_list=random_list,
                                                          mask_type=mask_type,
                                                          aggregation_type=aggregation_type)

    print(f"avg_acc:{avg_acc}  avg_prec:{avg_prec}  avg_rec:{avg_rec}  avg_f1:{avg_f1}")


if __name__ == "__main__":
    run()

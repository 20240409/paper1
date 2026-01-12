import torch.nn as nn

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
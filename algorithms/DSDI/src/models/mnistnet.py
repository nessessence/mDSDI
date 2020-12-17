import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.ds_z_fc = nn.Linear(196, 196)
        self.di_z_fc = nn.Linear(196, 196)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        di_z = self.di_z_fc(x)
        ds_z = self.ds_z_fc(x)

        return x, di_z, ds_z

def mnistnet(**kwargs):
    model = MnistNet(**kwargs)

    return model
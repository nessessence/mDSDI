import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)

        self.ds_z_fc = nn.Linear(120, 120)
        self.di_z_fc = nn.Linear(120, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))

        di_z = F.relu(self.di_z_fc(x))
        ds_z = F.relu(self.ds_z_fc(x))

        return x, di_z, ds_z

def mnistnet(**kwargs):
    model = MnistNet(**kwargs)

    return model
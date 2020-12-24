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

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)

        return x

def mnistnet(**kwargs):
    model = MnistNet(**kwargs)

    return model
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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
        x_flatten = torch.flatten(x, 1)
        di_z = self.di_z_fc(x_flatten)
        ds_z = self.ds_z_fc(x_flatten)

        return x, di_z, ds_z, x_flatten

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 2, stride=2)
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        
        return x_hat


class MNIST_Autoencoder(nn.Module):
    def __init__(self):
        super(MNIST_Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z, z_i, z_s, z_flatten = self.encoder(x)
        x_hat = self.decoder(z)
        return z_flatten, z_i, z_s, x_hat   

def mnistnet(**kwargs):
    model = MNIST_Autoencoder(**kwargs)

    return model
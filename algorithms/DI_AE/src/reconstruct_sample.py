import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.DI_AE.src.dataloaders import dataloader_factory
from algorithms.DI_AE.src.models import model_factory
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torchvision

class GradReverse(torch.autograd.Function):
    iter_num = 0
    alpha = 10
    low = 0.0
    high = 1.0
    max_iter = 3000

    @staticmethod
    def forward(ctx, x):
        GradReverse.iter_num += 1
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        coeff = np.float(2.0 * (GradReverse.high - GradReverse.low) / (1.0 + np.exp(-GradReverse.alpha * GradReverse.iter_num / GradReverse.max_iter))
                         - (GradReverse.high - GradReverse.low) + GradReverse.low)
        return -coeff * grad_output

class Domain_Discriminator(nn.Module):
    def __init__(self, feature_dim, domain_classes):
        super(Domain_Discriminator, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, domain_classes)
        )

    def forward(self, di_z):
        y = self.class_classifier(GradReverse.apply(di_z))
        return y

class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, classes)
    
    def forward(self, z):
        y = self.classifier(z)
        return y 

invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

class Trainer_DI_AE:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.plot_writer = "algorithms/" + self.args.algorithm + "/results/plots/" + self.args.exp_name + "_" + exp_idx + "/"
        self.train_loaders = [DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[0]], domain_label = 0), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[1]], domain_label = 1), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[2]], domain_label = 2), batch_size = self.args.batch_size, shuffle = True)]
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_val_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.target_test_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        
        self.model = model_factory.get_model(self.args.model)().to(self.device)
        self.classifier = Classifier(feature_dim = self.args.feature_dim, classes = self.args.n_classes).to(self.device)
        self.domain_discriminator = Domain_Discriminator(feature_dim = self.args.feature_dim, domain_classes = 3).to(self.device)
        
        self.checkpoint_name = "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + exp_idx

    def generate(self):
        checkpoint = torch.load(self.checkpoint_name + '.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.domain_discriminator.load_state_dict(checkpoint['domain_discriminator_state_dict'])

        self.model.eval()
        self.classifier.eval()
        self.domain_discriminator.eval()

        self.train_iter_loaders = []
        for train_loader in self.train_loaders:
            self.train_iter_loaders.append(iter(train_loader))

        for iteration in range(self.args.iterations):
            samples, labels, domain_labels = [], [], []

            for idx in range(len(self.train_iter_loaders)):
                if (iteration % len(self.train_iter_loaders[idx])) == 0:
                    self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
                train_loader = self.train_iter_loaders[idx]

                itr_samples, itr_labels, itr_domain_labels = train_loader.next()

                samples.append(itr_samples)
                labels.append(itr_labels)
                domain_labels.append(itr_domain_labels)

            samples = torch.cat(samples, dim=0).to(self.device)
            labels = torch.cat(labels, dim=0).to(self.device)
            domain_labels = torch.cat(domain_labels, dim=0).to(self.device)    
            
            di_z, x_hat = self.model(samples)

            if iteration % self.args.step_eval == 0:

                for i in range(49):
                    # define subplot
                    plt.subplot(7, 7, 1 + i)
                    # turn off axis
                    plt.axis('off')
                    # plot raw pixel data
                    plt.imshow((invTrans(samples[i]).detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))

                plt.savefig(self.plot_writer + "IMAGE_{}.png".format(iteration), bbox_inches='tight', dpi=600)

                for i in range(49):
                    # define subplot
                    plt.subplot(7, 7, 1 + i)
                    # turn off axis
                    plt.axis('off')
                    # plot raw pixel data
                    plt.imshow((invTrans(x_hat[i]).detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))

                plt.savefig(self.plot_writer + "GEN_IMAGE_{}.png".format(iteration), bbox_inches='tight', dpi=600)
                exit()
                
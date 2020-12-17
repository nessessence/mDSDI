import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.DSDI.src.dataloaders import dataloader_factory
from algorithms.DSDI.src.models import model_factory
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

class Disentangle_GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return - grad_output

class Disentangle_Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Disentangle_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear((int(feature_dim) * 2), feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2)
        )

    def forward(self, di_z, ds_z):
        z = torch.cat((di_z, ds_z), dim = 1)
        y = self.discriminator(GradReverse.apply(z))
        return y

class Mask_Domain_Generator(nn.Module):
    def __init__(self, feature_in, feature_out):
        super(Mask_Domain_Generator, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Linear(feature_in, feature_out),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.mask_generator(z)

class Mask_Domain_Classifier(nn.Module):
    def __init__(self, feature_dim, domain_classes):
        super(Mask_Domain_Classifier, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, domain_classes)
        )

    def forward(self, mask):
        y = self.class_classifier(mask)
        return y

class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(int(feature_dim * 2), classes)
    
    def forward(self, di_z, ds_z):
        z = torch.cat((di_z, ds_z), dim = 1)
        y = self.classifier(z)
        return y 

class ZS_Domain_Classifier(nn.Module):
    def __init__(self, feature_dim, domain_classes):
        super(ZS_Domain_Classifier, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, domain_classes)
        )

    def forward(self, ds_z):
        y = self.class_classifier(ds_z)
        return y

invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

class Trainer_DSDI:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "_" + exp_idx + "/")
        self.train_loaders = [DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[0]], domain_label = 0), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[1]], domain_label = 1), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[2]], domain_label = 2), batch_size = self.args.batch_size, shuffle = True)]
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_val_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.target_test_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        
        self.model = model_factory.get_model(self.args.model)().to(self.device)
        self.classifier = Classifier(feature_dim = self.args.feature_dim, classes = self.args.n_classes).to(self.device)
        self.zs_domain_classifier = ZS_Domain_Classifier(feature_dim = self.args.feature_dim, domain_classes = 3).to(self.device)
        self.domain_discriminator = Domain_Discriminator(feature_dim = self.args.feature_dim, domain_classes = 3).to(self.device)
        self.mask_domain_generator = Mask_Domain_Generator(feature_in = self.args.feature_dim, feature_out = self.args.feature_dim).to(self.device)
        self.mask_domain_classifier = Mask_Domain_Classifier(feature_dim = self.args.feature_dim, domain_classes = 3).to(self.device)
        self.disentangle_discriminator = Disentangle_Discriminator(feature_dim = self.args.feature_dim).to(self.device)

        self.model.encoder.di_z_fc.weight.requires_grad = False
        self.model.encoder.di_z_fc.bias.requires_grad = False
        disentangle_optimizer = list(filter(lambda p: p.requires_grad, self.model.encoder.parameters())) + list(self.disentangle_discriminator.parameters())
        self.disentangle_optimizer = torch.optim.SGD(disentangle_optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)

        for param in self.model.encoder.parameters():
            param.requires_grad = True
        optimizer = list(self.model.parameters()) + list(self.classifier.parameters()) + list(self.domain_discriminator.parameters()) + list(self.zs_domain_classifier.parameters()) + list(self.mask_domain_generator.parameters())

        self.optimizer = torch.optim.SGD(optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)
        
        mask_optimizer = list(self.mask_domain_generator.parameters()) + list(self.mask_domain_classifier.parameters())
        self.mask_optimizer = torch.optim.SGD(mask_optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)

        self.criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.MSELoss()
        self.disentangle_scheduler = StepLR(self.optimizer, step_size=self.args.iterations * 0.8)
        self.scheduler = StepLR(self.optimizer, step_size=self.args.iterations * 0.8)
        self.mask_scheduler = StepLR(self.mask_optimizer, step_size=self.args.iterations * 0.8)

        self.checkpoint_name = "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + exp_idx
        self.val_loss_min = np.Inf

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def train(self):
        self.model.train()
        # self.model.bn_eval()
        self.classifier.train()
        self.zs_domain_classifier.train()
        self.domain_discriminator.train()
        self.mask_domain_classifier.train()
        self.mask_domain_generator.train()
        self.disentangle_discriminator.train()

        n_class_corrected = 0
        n_domain_class_corrected = 0
        n_mask_domain_class_corrected = 0
        n_zs_domain_class_corrected = 0
        n_dtg_class_corrected = 0

        total_classification_loss = 0
        total_dc_loss = 0
        total_maskd_loss = 0
        total_zsc_loss = 0
        total_disentangle_loss = 0
        total_reconstruction_loss = 0
        total_samples = 0
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
            
            z, di_z, ds_z, x_hat= self.model(samples)

            # # Correlation Matrix
            # mdi_z, ddi_z = torch.mean(di_z, 0), torch.std(di_z, 0)          # Size M
            # mds_z, dds_z = torch.mean(ds_z, 0), torch.std(ds_z, 0)           # Size N

            # di_z_n = (di_z - mdi_z[None, :]) # BxM
            # ds_z_n = (ds_z - mds_z[None, :]) # BxN
            # C = di_z_n[:, :, None] * ds_z_n[:,None,:]              # BxMxN
            
            # target_cr = torch.zeros(C.shape[0], C.shape[1], C.shape[2]).to(self.device)
            # disentangle_loss = nn.MSELoss()(C, target_cr)
            # total_disentangle_loss += disentangle_loss.item()

            # Adversarial Training
            real_dtg = self.disentangle_discriminator(di_z, ds_z)
            fake_dtg = self.disentangle_discriminator(di_z, ds_z[torch.randperm(ds_z.size()[0])])
            real_label = torch.zeros(real_dtg.shape[0]).to(self.device, dtype=torch.int64)
            fake_label = torch.ones(fake_dtg.shape[0]).to(self.device, dtype=torch.int64)
            rf_dtg = torch.cat((real_dtg, fake_dtg))
            rf_label = torch.cat((real_label, fake_label))
            disentangle_loss = self.criterion(rf_dtg, rf_label)
            total_disentangle_loss += disentangle_loss.item()
            _, rf_dtg = torch.max(rf_dtg, 1)
            n_dtg_class_corrected += (rf_dtg == rf_label).sum().item()

            self.disentangle_optimizer.zero_grad()
            disentangle_loss.backward()
            self.disentangle_optimizer.step()
            self.disentangle_scheduler.step()

            z, di_z, ds_z, x_hat= self.model(samples)
            reconstruction_loss = self.reconstruction_criterion(x_hat, samples)
            total_reconstruction_loss += reconstruction_loss.item()

            di_predicted_domain = self.domain_discriminator(di_z)
            predicted_domain_di_loss = self.criterion(di_predicted_domain, domain_labels)
            total_dc_loss += predicted_domain_di_loss.item()
            
            ds_predicted_classes = self.zs_domain_classifier(ds_z)
            predicted_domain_ds_loss = self.criterion(ds_predicted_classes, domain_labels)
            total_zsc_loss += predicted_domain_ds_loss.item()

            mask = self.mask_domain_generator(z)
            ds_z = torch.mul(ds_z, mask)
            predicted_classes = self.classifier(di_z, ds_z)
            classification_loss = self.criterion(predicted_classes, labels)
            total_classification_loss += classification_loss.item()

            total_loss = classification_loss + 0.5 * predicted_domain_di_loss + 0.5 * predicted_domain_ds_loss + 0.5 * reconstruction_loss

            _, ds_predicted_classes = torch.max(ds_predicted_classes, 1)
            n_zs_domain_class_corrected += (ds_predicted_classes == domain_labels).sum().item()
            _, di_predicted_domain = torch.max(di_predicted_domain, 1)
            n_domain_class_corrected += (di_predicted_domain == domain_labels).sum().item()
            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
            total_samples += len(samples)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            z, di_z, ds_z, x_hat = self.model(samples)
            mask = self.mask_domain_generator(z)
            mask_predicted_domain = self.mask_domain_classifier(mask)
            predicted_domain_mask_loss = self.criterion(mask_predicted_domain, domain_labels)
            total_maskd_loss += predicted_domain_mask_loss.item()
            
            _, mask_predicted_domain = torch.max(mask_predicted_domain, 1)
            n_mask_domain_class_corrected += (mask_predicted_domain == domain_labels).sum().item()
            self.mask_optimizer.zero_grad()
            predicted_domain_mask_loss.backward()
            self.mask_optimizer.step()
            self.mask_scheduler.step()

            if iteration % self.args.step_eval == 0:
                self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Accuracy/domainAT_train', 100. * n_domain_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Accuracy/domainMask_train', 100. * n_mask_domain_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Accuracy/dtg_train', 100. * n_dtg_class_corrected / int(total_samples * 2), iteration)
                self.writer.add_scalar('Accuracy/domainZS_train', 100. * n_zs_domain_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Loss/train', total_classification_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/domainAT_train', total_dc_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/domainMask_train', total_maskd_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/domainZS_train', total_zsc_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/reconstruction', total_reconstruction_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/disentangle', total_disentangle_loss / total_samples, iteration)
                logging.info('Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}'.format(iteration, self.args.iterations,
                    n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
                self.evaluate(iteration)

                for i in range(49):
                    # define subplot
                    plt.subplot(7, 7, 1 + i)
                    # turn off axis
                    plt.axis('off')
                    # plot raw pixel data
                    plt.imshow((invTrans(samples[i]).detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))

                plt.savefig("algorithms/DSDI/results/plots/IMAGE_{}.png".format(iteration), bbox_inches='tight', dpi=600)

                for i in range(49):
                    # define subplot
                    plt.subplot(7, 7, 1 + i)
                    # turn off axis
                    plt.axis('off')
                    # plot raw pixel data
                    plt.imshow((invTrans(x_hat[i]).detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))

                plt.savefig("algorithms/DSDI/results/plots/GEN_IMAGE_{}.png".format(iteration), bbox_inches='tight', dpi=600)
                
            n_class_corrected = 0
            n_domain_class_corrected = 0
            n_mask_domain_class_corrected = 0
            n_zs_domain_class_corrected = 0
            n_dtg_class_corrected = 0

            total_dc_loss = 0
            total_classification_loss = 0
            total_maskd_loss = 0
            total_zsc_loss = 0
            total_disentangle_loss = 0
            total_reconstruction_loss = 0
            total_samples = 0
    
    def evaluate(self, n_iter):
        self.model.eval()
        self.classifier.eval()
        self.zs_domain_classifier.eval()
        self.domain_discriminator.eval()
        self.mask_domain_classifier.eval()
        self.mask_domain_generator.eval()
        self.disentangle_discriminator.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.val_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                z, di_z, ds_z, x_hat = self.model(samples)
                mask = self.mask_domain_generator(z)
                ds_z = torch.mul(ds_z, mask)

                predicted_classes = self.classifier(di_z, ds_z)

                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        self.writer.add_scalar('Accuracy/validate', 100. * n_class_corrected / len(self.val_loader.dataset), n_iter)
        self.writer.add_scalar('Loss/validate', total_classification_loss / len(self.val_loader.dataset), n_iter)
        logging.info('Val set: Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}'.format(n_class_corrected, len(self.val_loader.dataset),
            100. * n_class_corrected / len(self.val_loader.dataset), total_classification_loss / len(self.val_loader.dataset)))

        val_loss = total_classification_loss / len(self.val_loader.dataset)
        self.model.train()
        # self.model.bn_eval()
        self.classifier.train()
        self.zs_domain_classifier.train()
        self.domain_discriminator.train()
        self.mask_domain_classifier.train()
        self.mask_domain_generator.train()
        self.disentangle_discriminator.train()

        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            torch.save({'model_state_dict': self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'zs_domain_classifier_state_dict': self.zs_domain_classifier.state_dict(),
                'domain_discriminator_state_dict': self.domain_discriminator.state_dict(),
                'mask_domain_classifier_state_dict': self.mask_domain_classifier.state_dict(),
                'mask_domain_generator_state_dict': self.mask_domain_generator.state_dict(),
                'disentangle_discriminator_state_dict': self.disentangle_discriminator.state_dict()
                }, self.checkpoint_name + '.pt')
    
    def test(self):
        checkpoint = torch.load(self.checkpoint_name + '.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.zs_domain_classifier.load_state_dict(checkpoint['zs_domain_classifier_state_dict'])
        self.domain_discriminator.load_state_dict(checkpoint['domain_discriminator_state_dict'])
        self.mask_domain_classifier.load_state_dict(checkpoint['mask_domain_classifier_state_dict'])
        self.mask_domain_generator.load_state_dict(checkpoint['mask_domain_generator_state_dict'])
        self.disentangle_discriminator.load_state_dict(checkpoint['disentangle_discriminator_state_dict'])

        self.model.eval()
        self.classifier.eval()
        self.zs_domain_classifier.eval()
        self.domain_discriminator.eval()
        self.mask_domain_classifier.eval()
        self.mask_domain_generator.eval()
        self.disentangle_discriminator.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                z, di_z, ds_z, x_hat = self.model(samples)
                mask = self.mask_domain_generator(z)
                ds_z = torch.mul(ds_z, mask)
                
                predicted_classes = self.classifier(di_z, ds_z)

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))
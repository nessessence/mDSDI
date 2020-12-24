import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.DI.src.dataloaders import dataloader_factory
from algorithms.DI.src.models import model_factory
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

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

class Trainer_DI:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "_" + exp_idx + "/")
        self.train_loaders = [DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[0]], domain_label = 0), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[1]], domain_label = 1), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[2]], domain_label = 2), batch_size = self.args.batch_size, shuffle = True)]
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_val_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.target_test_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        
        self.zi_model = model_factory.get_model(self.args.model)().to(self.device)

        self.classifier = Classifier(feature_dim = self.args.feature_dim, classes = self.args.n_classes).to(self.device)
        self.domain_discriminator = Domain_Discriminator(feature_dim = self.args.feature_dim, domain_classes = 3).to(self.device)

        optimizer = list(self.zi_model.parameters()) + list(self.classifier.parameters()) + list(self.domain_discriminator.parameters())
        self.optimizer = torch.optim.SGD(optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)
        
        # disentangle_optimizer = list(self.zs_model.parameters()) + list(self.disentangle_discriminator.parameters())
        # self.disentangle_optimizer = torch.optim.SGD(disentangle_optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)

        self.criterion = nn.CrossEntropyLoss()
        # self.adversarial_loss = nn.BCELoss()
        # self.disentangle_scheduler = StepLR(self.disentangle_optimizer, step_size=self.args.iterations)
        self.scheduler = StepLR(self.optimizer, step_size=self.args.iterations)

        self.checkpoint_name = "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + exp_idx
        self.val_loss_min = np.Inf

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def train(self):        
        self.zi_model.train()
        # self.model.bn_eval()
        self.classifier.train()
        self.domain_discriminator.train()

        n_class_corrected = 0
        n_domain_class_corrected = 0
      
        total_classification_loss = 0
        total_dc_loss = 0
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
            
            di_z = self.zi_model(samples)
            
            di_predicted_domain = self.domain_discriminator(di_z)
            predicted_domain_di_loss = self.criterion(di_predicted_domain, domain_labels)
            total_dc_loss += predicted_domain_di_loss.item()
            
            predicted_classes = self.classifier(di_z)
            classification_loss = self.criterion(predicted_classes, labels)
            total_classification_loss += classification_loss.item()

            total_loss = classification_loss + 0.5 * predicted_domain_di_loss

            _, di_predicted_domain = torch.max(di_predicted_domain, 1)
            n_domain_class_corrected += (di_predicted_domain == domain_labels).sum().item()
            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
           
            total_samples += len(samples)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
                        
            if iteration % self.args.step_eval == 0:
                self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Accuracy/domainAT_train', 100. * n_domain_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Loss/train', total_classification_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/domainAT_train', total_dc_loss / total_samples, iteration)
                logging.info('Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}'.format(iteration, self.args.iterations,
                    n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
                self.evaluate(iteration)
                
            n_class_corrected = 0
            n_domain_class_corrected = 0
       
            total_dc_loss = 0
            total_classification_loss = 0
            total_samples = 0
    
    def evaluate(self, n_iter):
        self.zi_model.eval()
        self.classifier.eval()
        self.domain_discriminator.eval()
        
        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.val_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                di_z = self.zi_model(samples)

                predicted_classes = self.classifier(di_z)

                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        self.writer.add_scalar('Accuracy/validate', 100. * n_class_corrected / len(self.val_loader.dataset), n_iter)
        self.writer.add_scalar('Loss/validate', total_classification_loss / len(self.val_loader.dataset), n_iter)
        logging.info('Val set: Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}'.format(n_class_corrected, len(self.val_loader.dataset),
            100. * n_class_corrected / len(self.val_loader.dataset), total_classification_loss / len(self.val_loader.dataset)))

        val_loss = total_classification_loss / len(self.val_loader.dataset)
        
        self.zi_model.train()
        # self.model.bn_eval()
        self.classifier.train()
        self.domain_discriminator.train()

        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            torch.save({
                'zi_model_state_dict': self.zi_model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'domain_discriminator_state_dict': self.domain_discriminator.state_dict(),
            }, self.checkpoint_name + '.pt')
    
    def test(self):
        checkpoint = torch.load(self.checkpoint_name + '.pt')
        self.zi_model.load_state_dict(checkpoint['zi_model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.domain_discriminator.load_state_dict(checkpoint['domain_discriminator_state_dict'])

        self.zi_model.eval()
        self.classifier.eval()
        self.domain_discriminator.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                di_z = self.zi_model(samples)
                
                predicted_classes = self.classifier(di_z)

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))
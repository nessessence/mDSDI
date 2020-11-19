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

class Domain_Classifier(nn.Module):
    def __init__(self, feature_dim, domain_classes):
        super(Domain_Classifier, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, domain_classes)
        )

    def forward(self, di_z):
        y = self.class_classifier(GradReverse.apply(di_z))
        return y

class Mask_Domain_Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Mask_Domain_Generator, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Linear(feature_dim, 256),
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
        self.classifier = nn.Linear(feature_dim, classes)
    
    def forward(self, di_z, ds_z):
        z = torch.cat((di_z, ds_z), dim =1)
        y = self.classifier(z)
        return y 

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
        self.domain_classifier = Domain_Classifier(feature_dim = 256, domain_classes = 3).to(self.device)
        self.mask_domain_generator = Mask_Domain_Generator(feature_dim = 512).to(self.device)
        self.mask_domain_classifier = Mask_Domain_Classifier(feature_dim = 256, domain_classes = 3).to(self.device)

        optimizer = list(self.model.parameters()) + list(self.classifier.parameters()) + list(self.domain_classifier.parameters())

        self.optimizer = torch.optim.SGD(optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)
        
        mask_optimizer = list(self.mask_domain_generator.parameters()) + list(self.mask_domain_classifier.parameters())
        self.mask_optimizer = torch.optim.SGD(mask_optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)

        self.criterion = nn.CrossEntropyLoss()
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
        self.domain_classifier.train()
        self.mask_domain_classifier.train()
        self.mask_domain_generator.train()

        n_class_corrected = 0
        total_classification_loss = 0
        total_dc_loss = 0
        total_maskd_loss = 0
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
                
            di_z, ds_z = self.model(samples)
            di_predicted_domain = self.domain_classifier(di_z)
            predicted_domain_di_loss = self.criterion(di_predicted_domain, domain_labels)
            total_dc_loss += predicted_domain_di_loss.item()
            
            z = torch.cat((di_z, ds_z), dim =1)
            mask = self.mask_domain_generator(z)
            ds_z = torch.mul(ds_z, mask)
            predicted_classes = self.classifier(di_z, ds_z)
            classification_loss = self.criterion(predicted_classes, labels)
            total_classification_loss += classification_loss.item()

            total_loss = classification_loss + 0.1 * predicted_domain_di_loss

            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
            total_samples += len(samples)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            di_z, ds_z= self.model(samples)
            z = torch.cat((di_z, ds_z), dim =1)
            mask = self.mask_domain_generator(z)
            mask_predicted_domain = self.mask_domain_classifier(mask)
            predicted_domain_mask_loss = self.criterion(mask_predicted_domain, domain_labels)
            total_maskd_loss += predicted_domain_mask_loss.item()
            
            self.mask_optimizer.zero_grad()
            predicted_domain_mask_loss.backward()
            self.mask_optimizer.step()
            self.mask_scheduler.step()

            if iteration % self.args.step_eval == 0:
                self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Loss/train', total_classification_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/domainAT_train', total_dc_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/domainMask_train', total_maskd_loss / total_samples, iteration)
                logging.info('Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}'.format(iteration, self.args.iterations,
                    n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
                self.evaluate(iteration)
                if self.args.self_test:
                    self.self_test(iteration)
                
            n_class_corrected = 0
            total_dc_loss = 0
            total_classification_loss = 0
            total_maskd_loss = 0
            total_samples = 0
    
    def evaluate(self, n_iter):
        self.model.eval()
        self.classifier.eval()
        self.domain_classifier.eval()
        self.mask_domain_classifier.eval()
        self.mask_domain_generator.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.val_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                di_z, ds_z = self.model(samples)
                z = torch.cat((di_z, ds_z), dim =1)
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
        self.domain_classifier.train()
        self.mask_domain_classifier.train()
        self.mask_domain_generator.train()

        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            torch.save({'model_state_dict': self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'domain_classifier_state_dict': self.domain_classifier.state_dict(),
                'mask_domain_classifier_state_dict': self.mask_domain_classifier.state_dict(),
                'mask_domain_generator_state_dict': self.mask_domain_generator.state_dict(),
                }, self.checkpoint_name + '.pt')
    
    def self_test(self, n_iter):
        self.model.eval()
        self.classifier.eval()
        self.domain_classifier.eval()
        self.mask_domain_classifier.eval()
        self.mask_domain_generator.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                di_z, ds_z = self.model(samples)
                z = torch.cat((di_z, ds_z), dim =1)
                mask = self.mask_domain_generator(z)
                ds_z = torch.mul(ds_z, mask)

                predicted_classes = self.classifier(di_z, ds_z)
                
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()

        self.writer.add_scalar('Accuracy/test', 100. * n_class_corrected / len(self.test_loader.dataset), n_iter)
        self.writer.add_scalar('Loss/test', total_classification_loss / len(self.test_loader.dataset), n_iter)
        logging.info('Self test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))
        
        self.model.train()
        # self.model.bn_eval()
        self.classifier.train()
        self.domain_classifier.train()
        self.mask_domain_classifier.train()
        self.mask_domain_generator.train()
    
    def test(self):
        checkpoint = torch.load(self.checkpoint_name + '.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.domain_classifier.load_state_dict(checkpoint['domain_classifier_state_dict'])
        self.mask_domain_classifier.load_state_dict(checkpoint['mask_domain_classifier_state_dict'])
        self.mask_domain_generator.load_state_dict(checkpoint['mask_domain_generator_state_dict'])
        
        self.model.eval()
        self.classifier.eval()
        self.domain_classifier.eval()
        self.mask_domain_classifier.eval()
        self.mask_domain_generator.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                di_z, ds_z = self.model(samples)
                z = torch.cat((di_z, ds_z), dim =1)
                mask = self.mask_domain_generator(z)
                ds_z = torch.mul(ds_z, mask)
                
                predicted_classes = self.classifier(di_z, ds_z)

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))
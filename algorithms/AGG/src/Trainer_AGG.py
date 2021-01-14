import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.AGG.src.dataloaders import dataloader_factory
from algorithms.AGG.src.models import model_factory
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, z):
        y = self.classifier(z)
        return y   

def set_tr_val_samples_labels(meta_filenames):
    sample_tr_paths, class_tr_labels, sample_val_paths, class_val_labels = [], [], [], []

    for idx_domain, meta_filename in enumerate(meta_filenames):
        column_names = ['filename', 'class_label']
        data_frame = pd.read_csv(meta_filename, header = None, names = column_names, sep='\s+')
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)

        split_idx = int(len(data_frame) * 0.9)
        sample_tr_paths.append(data_frame["filename"][:split_idx])
        class_tr_labels.append(data_frame["class_label"][:split_idx])

        sample_val_paths.extend(data_frame["filename"][split_idx:])
        class_val_labels.extend(data_frame["class_label"][split_idx:])
            
    return sample_tr_paths, class_tr_labels, sample_val_paths, class_val_labels

def set_test_samples_labels(meta_filenames):
    sample_paths, class_labels = [], []
    for idx_domain, meta_filename in enumerate(meta_filenames):
        column_names = ['filename', 'class_label']
        data_frame = pd.read_csv(meta_filename, header = None, names = column_names, sep='\s+')
        sample_paths.extend(data_frame["filename"])
        class_labels.extend(data_frame["class_label"])
            
    return sample_paths, class_labels

class Trainer_AGG:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "_" + exp_idx + "/")
        
        src_tr_sample_paths, src_tr_class_labels, src_val_sample_paths, src_val_class_labels = set_tr_val_samples_labels(self.args.src_train_meta_filenames)
        test_sample_paths, test_class_labels = set_test_samples_labels(self.args.target_test_meta_filenames)

        self.train_loaders = [DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, sample_paths = src_tr_sample_paths[0], class_labels = src_tr_class_labels[0], domain_label = 0), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, sample_paths = src_tr_sample_paths[1], class_labels = src_tr_class_labels[1], domain_label = 1), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, sample_paths = src_tr_sample_paths[2], class_labels = src_tr_class_labels[2], domain_label = 2), batch_size = self.args.batch_size, shuffle = True)]
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, sample_paths = src_val_sample_paths, class_labels = src_val_class_labels), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, sample_paths = test_sample_paths, class_labels = test_class_labels), batch_size = self.args.batch_size, shuffle = True)

        self.model = model_factory.get_model(self.args.model)().to(self.device)
        self.classifier = Classifier(feature_dim = self.args.feature_dim, classes = self.args.n_classes).to(self.device)

        optimizer = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.SGD(optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=self.args.iterations, gamma=0.1)

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

        n_class_corrected = 0
        total_classification_loss = 0
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
            
            samples = torch.cat(samples, dim=0)
            labels = torch.cat(labels, dim=0)
            domain_labels = torch.cat(domain_labels, dim=0)

            indexes = torch.randperm(samples.shape[0])
            samples = samples[indexes].to(self.device)
            labels = labels[indexes].to(self.device)
            domain_labels = domain_labels[indexes].to(self.device)
            
            predicted_classes = self.classifier(self.model(samples))
            classification_loss = self.criterion(predicted_classes, labels)
            total_classification_loss += classification_loss.item()

            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
            total_samples += len(samples)

            self.optimizer.zero_grad()
            classification_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if iteration % self.args.step_eval == 0:
                self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Loss/train', total_classification_loss / total_samples, iteration)
                logging.info('Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}'.format(iteration, self.args.iterations,
                    n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
                self.evaluate(iteration)
            
            n_class_corrected = 0
            total_classification_loss = 0
            total_samples = 0
    
    def evaluate(self, n_iter):
        self.model.eval()
        self.classifier.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.val_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
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

        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            torch.save({'model_state_dict': self.model.state_dict(), 'classifier_state_dict': self.classifier.state_dict()}, self.checkpoint_name + '.pt')

    def test(self):
        checkpoint = torch.load(self.checkpoint_name + '.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.model.eval()
        self.classifier.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))
import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.JiGen.src.dataloaders import dataloader_factory
from algorithms.JiGen.src.models import model_factory
from torch.optim.lr_scheduler import StepLR

class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, z):
        y = self.classifier(z)
        return y

class JiGen_Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(JiGen_Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, z):
        y = self.classifier(z)
        return y

class Trainer_JiGen:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "_" + exp_idx + "/")
        self.train_loaders = [DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[0]], n_jig_classes = self.args.n_jig_classes, domain_label = 0), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[1]], n_jig_classes = self.args.n_jig_classes, domain_label = 1), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[2]], n_jig_classes = self.args.n_jig_classes, domain_label = 2), batch_size = self.args.batch_size, shuffle = True)]
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, n_jig_classes = self.args.n_jig_classes, meta_filenames = self.args.src_val_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, n_jig_classes = self.args.n_jig_classes, meta_filenames = self.args.target_test_meta_filenames), batch_size = self.args.batch_size, shuffle = True)

        self.model = model_factory.get_model(self.args.model)().to(self.device)
        self.classifier = Classifier(feature_dim = self.args.feature_dim, classes = self.args.n_classes).to(self.device)
        self.jigen_classifier = JiGen_Classifier(feature_dim = self.args.feature_dim, classes = self.args.n_jig_classes + 1).to(self.device)

        optimizer = list(self.model.parameters()) + list(self.classifier.parameters()) + list(self.jigen_classifier.parameters())
        self.optimizer = torch.optim.SGD(optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=self.args.iterations * 0.8)

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
        self.jigen_classifier.train()

        n_class_corrected = 0
        total_classification_loss = 0
        total_jigen_loss = 0
        total_samples = 0
        self.train_iter_loaders = []
        for train_loader in self.train_loaders:
            self.train_iter_loaders.append(iter(train_loader))

        for iteration in range(self.args.iterations):
            samples, labels, domain_labels, jigen_labels = [], [], [], []

            for idx in range(len(self.train_iter_loaders)):
                if (iteration % len(self.train_iter_loaders[idx])) == 0:
                    self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
                train_loader = self.train_iter_loaders[idx]

                itr_samples, itr_labels, itr_domain_labels, itr_jigen_labels = train_loader.next()
                samples.append(itr_samples)
                labels.append(itr_labels)
                domain_labels.append(itr_domain_labels)
                jigen_labels.append(itr_jigen_labels)
            
            samples = torch.cat(samples, dim=0).to(self.device)
            labels = torch.cat(labels, dim=0).to(self.device)
            domain_labels = torch.cat(domain_labels, dim=0).to(self.device)
            jigen_labels = torch.cat(jigen_labels, dim = 0).to(self.device)
            
            latten_z = self.model(samples)
            predicted_classes = self.classifier(latten_z)
            classification_loss = self.criterion(predicted_classes[jigen_labels == 0], labels[jigen_labels == 0])
            total_classification_loss += classification_loss.item()

            predicted_jigen = self.jigen_classifier(latten_z)
            jigen_loss = self.criterion(predicted_jigen, jigen_labels)
            total_jigen_loss += jigen_loss.item()
                
            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
            total_samples += len(samples)

            total_loss = classification_loss + jigen_loss * self.args.lambda_jigen
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if iteration % self.args.step_eval == 0:
                self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, iteration)
                self.writer.add_scalar('Loss/train', total_classification_loss / total_samples, iteration)
                self.writer.add_scalar('Loss/jigen_train', total_jigen_loss / total_samples, iteration)

                logging.info('Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}'.format(iteration, self.args.iterations,
                    n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
                self.evaluate(iteration)
                if self.args.self_test:
                    self.self_test(iteration)
            
            n_class_corrected = 0
            total_classification_loss = 0
            total_jigen_loss = 0
            total_samples = 0
            
    def evaluate(self, n_iter):
        self.model.eval()
        self.classifier.eval()
        self.jigen_classifier.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels, jigen_labels) in enumerate(self.val_loader):
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
        self.jigen_classifier.train()

        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            torch.save({'model_state_dict': self.model.state_dict(), 'classifier_state_dict': self.classifier.state_dict()}, self.checkpoint_name + '.pt')

    def self_test(self, n_iter):
        self.model.eval()
        self.classifier.eval()
        self.jigen_classifier.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels, jigen_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
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
        self.jigen_classifier.train()

    def test(self):
        checkpoint = torch.load(self.checkpoint_name + '.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.model.eval()
        self.classifier.eval()
        self.jigen_classifier.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels, jigen_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))
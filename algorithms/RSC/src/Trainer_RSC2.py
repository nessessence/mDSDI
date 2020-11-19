import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.RSC.src.dataloaders import dataloader_factory
from algorithms.RSC.src.models import model_factory
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models

class Trainer_RSC:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "/")
        self.train_loaders = [DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[0]], domain_label = 0), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[1]], domain_label = 1), batch_size = self.args.batch_size, shuffle = True),
            DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[2]], domain_label = 2), batch_size = self.args.batch_size, shuffle = True)]
        
        # self.train_loader = DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_train_meta_filenames), num_workers=4, pin_memory=True, batch_size = self.args.batch_size, shuffle = True)
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_val_meta_filenames), num_workers=4, pin_memory=True, batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.target_test_meta_filenames), num_workers=4, pin_memory=True, batch_size = self.args.batch_size, shuffle = True)

        self.model = model_factory.get_model(self.args.model)(classes = self.args.n_classes).to(self.device)

        optimizer = list(self.model.parameters())
        self.optimizer = torch.optim.SGD(optimizer, lr = self.args.learning_rate, weight_decay = self.args.weight_decay, momentum = self.args.momentum, nesterov = False)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=self.args.iterations * 0.8)

        self.checkpoint_name = "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name
        self.val_loss_min = np.Inf
        self.best_test = 0

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def train(self):
        self.model.train()

        n_class_corrected = 0
        total_classification_loss = 0
        total_samples = 0
        # self.train_iter_loaders = []
        # for train_loader in self.train_loaders:
        #     self.train_iter_loaders.append(iter(train_loader))

        for epoch in range(30):
            for iteration, (samples, labels) in enumerate(self.train_loaders):
                samples, labels = samples.to(self.device), labels.to(self.device)
            
        # for iteration in range(self.args.iterations):
        #     samples, labels, domain_labels = [], [], []

        #     for idx in range(len(self.train_iter_loaders)):
        #         if (iteration % len(self.train_iter_loaders[idx])) == 0:
        #             self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
        #         train_loader = self.train_iter_loaders[idx]

        #         itr_samples, itr_labels, itr_domain_labels = train_loader.next()
        #         samples.append(itr_samples)
        #         labels.append(itr_labels)
        #         domain_labels.append(itr_domain_labels)
                self.optimizer.zero_grad()
                
                predicted_classes = self.model(samples, labels, True)
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
                total_samples += len(samples)

                classification_loss.backward()
                self.optimizer.step()

                # if iteration % self.args.step_eval == 0:
            # self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, iteration)
            # self.writer.add_scalar('Loss/train', total_classification_loss / total_samples, iteration)
            logging.info('Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}'.format(epoch, 30,
                    n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
            self.evaluate(epoch)
            # if self.args.self_test:
            self.self_test(epoch)
            
            self.scheduler.step()
            n_class_corrected = 0
            total_classification_loss = 0
            total_samples = 0
    
    def evaluate(self, n_iter):
        self.model.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.val_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes = self.model(samples, labels, False)
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

        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_name + '.pt')

    def self_test(self, n_iter):
        self.model.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes = self.model(samples, labels, False)
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()

        test_acc = 100. * n_class_corrected / len(self.test_loader.dataset)
        if self.best_test < test_acc:
            self.best_test = test_acc

        self.writer.add_scalar('Accuracy/test', 100. * n_class_corrected / len(self.test_loader.dataset), n_iter)
        self.writer.add_scalar('Loss/test', total_classification_loss / len(self.test_loader.dataset), n_iter)
        logging.info('Self test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))
        
        self.model.train()

    def test(self):
        checkpoint = torch.load(self.checkpoint_name + '.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes = self.model(samples, labels, False)
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))
        
        logging.info('Best test set: Accuracy: {}/{} ({:.2f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            self.best_test))
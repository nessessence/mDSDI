import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pickle
from SSDG.DGFAS import DG_model, Discriminator
from dataset_configs import dataset_configs
from get_loader import get_dataset

def tsne_plot(embedded_features, labels, domain_labels):
    def unique(list1):
        unique_list = []
        for x in list1:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    labels = np.asarray(labels)
    domain_labels = np.asarray(domain_labels)
    tsne_model = TSNE(n_components=2, init='pca')
    X_2d = tsne_model.fit_transform(embedded_features)
    
    label_target_names = unique(labels)
    domain_label_target_names = unique(domain_labels)
    label_target_ids = range(len(label_target_names))
    domain_label_target_ids = range(len(domain_label_target_names))
    colors=['red','green','blue','black','brown','grey','orange','yellow','pink','cyan','magenta']

    #Class tSNE
    plt.figure(figsize=(16, 16))
    for i, label in zip(label_target_ids, label_target_names):
        plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=colors[i], label=label)

    plt.legend(loc=2, fontsize = 'x-small')
    plt.savefig('class_tSNE.png')

    #Domain tSNE
    plt.figure(figsize=(16, 16))
    for i, label in zip(domain_label_target_ids, domain_label_target_names):
        plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=colors[i], label=label)

    plt.legend(loc=2, fontsize = 'x-small')
    plt.savefig('domain_tSNE.png')

def get_embedded_train_set(loader, model):
    X_out, Y_out = [], []
    for batch_idx, (images, labels, domain_labels, video_ids) in enumerate(loader):
        psuedo_labels = []
        for idx, label in enumerate(labels):
            if label == 0:
                psuedo_labels.append(0)
            else:
                psuedo_labels.append(domain_labels[idx] + 1)
        labels = torch.Tensor(psuedo_labels)
        if torch.cuda.is_available():
            images= images.cuda()
            labels=labels.cuda()
        cls_out, feature = model(images, True)
        X_out += feature.tolist()
        Y_out += labels.tolist()
    return X_out, Y_out

def main():
    model = DG_model('resnet18').to('cuda')
    model.load_state_dict(torch.load("results/checkpoints/resnet18/9999_checkpoint.pt"))
    model.eval()
    domain_ids = {}
    src_train_dataloader_fake, src_train_dataloader_real, validate_loaders, domain_ids['source'], domain_ids['target'] = get_dataset('SSDG', dataset_configs['VinSmart'], [4], 128)
    X_total, Y_total = [], []
    with torch.no_grad():
        X_test, Y_test = get_embedded_train_set(src_train_dataloader_real[0], model)
        X_total += X_test
        
        Y_total, X_total = (list(t) for t in zip(*sorted(zip(Y_total, X_total))))
        print("Ziped!")
        # with open("Y_total.pickle", "wb") as fp:
        #     pickle.dump(Y_total, fp)
        # with open("X_total.pickle", "wb") as fp:
        #     pickle.dump(X_total, fp)
        tsne_plot(X_total,Y_total)
main()
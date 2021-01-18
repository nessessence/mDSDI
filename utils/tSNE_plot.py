import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pickle

def plot_TNSE(X_2d_tr, X_2d_test, tr_labels, test_labels, label_target_names, filename):
    colors=['red','green','blue','black','brown','grey','orange','yellow','pink','cyan','magenta']
    plt.figure(figsize=(16, 16))
    for i, label in zip(range(len(label_target_names)), label_target_names):
        plt.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], c=colors[i], marker='.', label=label)

    for i, label in zip(range(len(label_target_names)), label_target_names):
        plt.scatter(X_2d_test[test_labels == i, 0], X_2d_test[test_labels == i, 1], c=colors[i], marker='2', label=label)

    plt.legend(loc=2, fontsize = 'x-small')
    plt.savefig(filename)

def tsne_plot(Zi_out, Zs_out, labels, domain_labels, idx_split):
    def unique(list1):
        unique_list = []
        for x in list1:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    Z_out = []
    for idx in range(len(Zi_out)):
        Z_out.append(Zi_out[idx] + Zs_out[idx])

    labels = np.asarray(labels)
    domain_labels = np.asarray(domain_labels)
    tr_labels, test_labels = labels[:idx_split], labels[idx_split:]
    tr_domain_labels, test_domain_labels = domain_labels[:idx_split], domain_labels[idx_split:]
    label_target_names = unique(labels)
    domain_label_target_names = unique(domain_labels)

    tsne_model = TSNE(n_components=2, init='pca')
    Z_2d = tsne_model.fit_transform(Z_out)
    Zi_2d = tsne_model.fit_transform(Zi_out)
    Zs_2d = tsne_model.fit_transform(Zs_out)

    Z_2d_tr, Z_2d_test = Z_2d[:idx_split], Z_2d[idx_split:]
    Zi_2d_tr, Zi_2d_test = Zi_2d[:idx_split], Zi_2d[idx_split:]
    Zs_2d_tr, Zs_2d_test = Zs_2d[:idx_split], Zs_2d[idx_split:]
    tr_labels, test_labels = labels[:idx_split], labels[idx_split:]
    tr_domain_labels, test_domain_labels = domain_labels[:idx_split], domain_labels[idx_split:]
    
    plot_TNSE(Z_2d_tr, Z_2d_test, tr_labels, test_labels, label_target_names, 'Z_class_tSNE.png')
    plot_TNSE(Z_2d_tr, Z_2d_test, tr_domain_labels, test_domain_labels, domain_label_target_names, 'Z_domain_tSNE.png')

    plot_TNSE(Zi_2d_tr, Zi_2d_test, tr_labels, test_labels, label_target_names, 'Zi_class_tSNE.png')
    plot_TNSE(Zi_2d_tr, Zi_2d_test, tr_domain_labels, test_domain_labels, domain_label_target_names, 'Zi_domain_tSNE.png')

    plot_TNSE(Zs_2d_tr, Zs_2d_test, tr_labels, test_labels, label_target_names, 'Zs_class_tSNE.png')
    plot_TNSE(Zs_2d_tr, Zs_2d_test, tr_domain_labels, test_domain_labels, domain_label_target_names, 'Zs_domain_tSNE.png')

def main():
    with open ('Zi_out', 'rb') as fp:
        Zi_out = pickle.load(fp)
    with open ('Zs_out', 'rb') as fp:
        Zs_out = pickle.load(fp)
    with open ('Y_out', 'rb') as fp:
        Y_out = pickle.load(fp)
    with open ('Y_domain_out', 'rb') as fp:
        Y_domain_out = pickle.load(fp)
    
    # with open ('Zi_test', 'rb') as fp:
    #     Zi_test = pickle.load(fp)
    # with open ('Zs_test', 'rb') as fp:
    #     Zs_test = pickle.load(fp)
    # with open ('Y_test', 'rb') as fp:
    #     Y_test = pickle.load(fp)
    # with open ('Y_domain_test', 'rb') as fp:
    #     Y_domain_test = pickle.load(fp)

    # for i in range(len(Y_domain_test)):
    #     Y_domain_test[i] = 3

    # Zi_out += Zi_test
    # Zs_out += Zs_test
    # Y_out += Y_test
    # Y_domain_out += Y_domain_test
    print(len(Y_out))
    exit()

    idx_split = len(Zi_out)
    tsne_plot(Zi_out, Zs_out, Y_out, Y_domain_out, idx_split)

main()
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from random import random

class PACSDataloader(Dataset):
    def __init__(self, src_path, meta_filenames, n_jig_classes, domain_label = -1):
        def get_train_transformers():
            image_size = 222
            min_scale = 0.8
            max_scale = 1.0
            jitter = 0.4
            random_horiz_flip = 0.5
            tile_random_grayscale = 0.1
            img_tr = [transforms.RandomResizedCrop((int(image_size), int(image_size)), (min_scale, max_scale))]
            if random_horiz_flip > 0.0:
                img_tr.append(transforms.RandomHorizontalFlip(random_horiz_flip))
            if jitter > 0.0:
                img_tr.append(transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=min(0.5, jitter)))

            tile_tr = []
            if tile_random_grayscale:
                tile_tr.append(transforms.RandomGrayscale(tile_random_grayscale))
            tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

            return transforms.Compose(img_tr), transforms.Compose(tile_tr)
            
        self._image_transformer, self._augment_tile = get_train_transformers()
        self.grid_size = 3
        self.permutations = self.set_permutations(n_jig_classes)
        self.bias_whole_image = 0.9
        self.src_path = src_path
        self.domain_label = domain_label
        self.sample_paths, self.class_labels = self.set_samples_labels(meta_filenames)
        
    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile
    
    def set_permutations(self, n_jig_classes):
        all_perm = np.load('algorithms/JiGen/permutations_%d.npy' % (n_jig_classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm
    
    def set_samples_labels(self, meta_filenames):
        sample_paths, class_labels = [], []
        for idx_domain, meta_filename in enumerate(meta_filenames):
            column_names = ['filename', 'class_label']
            data_frame = pd.read_csv(meta_filename, header = None, names = column_names, sep='\s+')
            sample_paths.extend(data_frame["filename"])
            class_labels.extend(data_frame["class_label"] - 1)
            
        return sample_paths, class_labels

    def get_image(self, sample_path):
        img = Image.open(sample_path).convert('RGB')
        return self._image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index]
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(sample, n)

        order = np.random.randint(len(self.permutations) + 1)
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            sample = tiles
        else:
            sample = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            
        sample = torch.stack(sample, 0)
        sample = torchvision.utils.make_grid(sample, self.grid_size, padding=0)

        return sample, class_label, self.domain_label, int(order)

class PACS_Test_Dataloader(PACSDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        def get_val_transformer():
            image_size = 222
            img_tr = [transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            return transforms.Compose(img_tr)
        self._image_transformer = get_val_transformer()

    def __getitem__(self, index):
        sample = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index]
        return sample, class_label, self.domain_label, int(0)
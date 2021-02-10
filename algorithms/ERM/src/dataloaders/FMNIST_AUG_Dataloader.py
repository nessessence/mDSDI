import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FMNISTAUGDataloader(Dataset):
    def __init__(self, src_path, sample_paths, class_labels, domain_label = -1):
        self.image_transformer = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomCrop(28, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.2860], [0.3530]),
        ])
        self.src_path = src_path
        self.domain_label = domain_label
        self.sample_paths, self.class_labels = sample_paths, class_labels
        
    def get_image(self, sample_path):
        img = Image.open(sample_path)
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index]
        
        return sample, class_label, self.domain_label

class FMNIST_AUG_Test_Dataloader(FMNISTAUGDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self.image_transformer = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.2860], [0.3530]),
        ])
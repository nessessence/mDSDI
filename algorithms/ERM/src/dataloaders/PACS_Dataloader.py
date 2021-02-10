import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PACSDataloader(Dataset):
    def __init__(self, src_path, sample_paths, class_labels):
        self.image_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.src_path = src_path
        self.sample_paths, self.class_labels = sample_paths, class_labels

    def get_image(self, sample_path):
        img = Image.open(sample_path).convert('RGB')
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index] - 1
        
        return sample, class_label

class PACS_Test_Dataloader(PACSDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
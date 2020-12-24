import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PACSDataloader(Dataset):
    def __init__(self, src_path, meta_filenames, domain_label = -1):
        def get_train_transformers():
            image_size = 224
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
            img_tr.append(transforms.RandomGrayscale(tile_random_grayscale))
            img_tr.append(transforms.ToTensor())
            img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            return transforms.Compose(img_tr)
            
        self._image_transformer = get_train_transformers()
        self.src_path = src_path
        self.domain_label = domain_label
        self.sample_paths, self.class_labels = self.set_samples_labels(meta_filenames)
        
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
        
        return sample, class_label, self.domain_label

class PACS_Test_Dataloader(PACSDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._image_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
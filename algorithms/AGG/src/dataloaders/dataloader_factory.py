from algorithms.AGG.src.dataloaders.PACS_Dataloader import PACSDataloader, PACS_Test_Dataloader
from algorithms.AGG.src.dataloaders.MNIST_Dataloader import MNISTDataloader, MNIST_Test_Dataloader
from algorithms.AGG.src.dataloaders.MNIST_AUG_Dataloader import MNISTAUGDataloader, MNIST_AUG_Test_Dataloader
from algorithms.AGG.src.dataloaders.Colored_MNIST_Dataloader import ColoredMNISTDataloader, Colored_MNIST_Test_Dataloader
from algorithms.AGG.src.dataloaders.FMNIST_Dataloader import FMNISTDataloader, FMNIST_Test_Dataloader
from algorithms.AGG.src.dataloaders.FMNIST_AUG_Dataloader import FMNISTAUGDataloader, FMNIST_AUG_Test_Dataloader
from algorithms.AGG.src.dataloaders.DomainNet_Dataloader import DomainNetDataloader, DomainNet_Test_Dataloader
from algorithms.AGG.src.dataloaders.OfficeHome_Dataloader import OfficeHomeDataloader, OfficeHome_Test_Dataloader

train_dataloaders_map = {
    'PACS': PACSDataloader,
    'DomainNet': DomainNetDataloader,
    'FMNIST': FMNISTDataloader,
    'MNIST': MNISTDataloader,
    'FMNIST_AUG': FMNISTAUGDataloader,
    'MNIST_AUG': MNISTAUGDataloader,
    'OfficeHome': OfficeHomeDataloader,
    'Colored_MNIST': ColoredMNISTDataloader
}

test_dataloaders_map = {
    'PACS': PACS_Test_Dataloader,
    'DomainNet': DomainNet_Test_Dataloader,
    'FMNIST': FMNIST_Test_Dataloader,
    'MNIST': MNIST_Test_Dataloader,
    'FMNIST_AUG': FMNIST_AUG_Test_Dataloader,
    'MNIST_AUG': MNIST_AUG_Test_Dataloader,
    'OfficeHome': OfficeHome_Test_Dataloader,
    'Colored_MNIST': Colored_MNIST_Test_Dataloader
}

def get_train_dataloader(name):
    if name not in train_dataloaders_map:
        raise ValueError('Name of train dataloader unknown %s' % name)

    def get_dataloader_fn(**kwargs):
        return train_dataloaders_map[name](**kwargs)

    return get_dataloader_fn

def get_test_dataloader(name):
    if name not in test_dataloaders_map:
        raise ValueError('Name of test dataloader unknown %s' % name)

    def get_dataloader_fn(**kwargs):
        return test_dataloaders_map[name](**kwargs)

    return get_dataloader_fn
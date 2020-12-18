from algorithms.DSDI_AE.src.dataloaders.PACS_Dataloader import PACSDataloader, PACS_Test_Dataloader
from algorithms.DSDI_AE.src.dataloaders.MNIST_Dataloader import MNISTDataloader, MNIST_Test_Dataloader

train_dataloaders_map = {
    'PACS': PACSDataloader,
    'MNIST': MNISTDataloader
}

test_dataloaders_map = {
    'PACS': PACS_Test_Dataloader,
    'MNIST': MNIST_Test_Dataloader
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
from algorithms.DSDI_AE.src.models import resnet_ae
from algorithms.DSDI_AE.src.models import resnet
from algorithms.DSDI_AE.src.models import mnistnet

nets_map = {
    'mnistnet': mnistnet.mnistnet,
    'resnet18': resnet_ae.resnet18,
    'resnet50': resnet_ae.resnet50,
    'mask_18': resnet.resnet18,
    'mask_50': resnet.resnet50,
}

def get_model(name):
    if name not in nets_map:
        raise ValueError('Name of model unknown %s' % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn
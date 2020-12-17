from algorithms.DSDI.src.models import resnet
from algorithms.DSDI.src.models import mnistnet
from algorithms.DSDI.src.models import resnet_ae
from algorithms.DSDI.src.models import mnistnet_ae

nets_map = {
    'mnistnet': mnistnet.mnistnet,
    'mnistnet_ae': mnistnet_ae.mnistnet,
    'resnet18': resnet.resnet18,
    'resnet50': resnet.resnet50,
    'resnet18_ae': resnet_ae.resnet18
}

def get_model(name):
    if name not in nets_map:
        raise ValueError('Name of model unknown %s' % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn
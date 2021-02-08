import os
import argparse
import json
import logging
import torch
import numpy as np
import random

from algorithms.AGG.src.Trainer_AGG import Trainer_AGG
from algorithms.DSDI.src.Trainer_DSDI import Trainer_DSDI

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

algorithms_map = {
    'AGG': Trainer_AGG,
    'DSDI': Trainer_DSDI
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help = "Path to configuration file")
    parser.add_argument("--exp_idx", help = "Index of experiment")
    parser.add_argument("--gpu_idx", help = "Index of GPU")
    bash_args = parser.parse_args()
    with open(bash_args.config, "r") as inp:
        args = argparse.Namespace(**json.load(inp))
        
    os.environ["CUDA_VISIBLE_DEVICES"] = bash_args.gpu_idx        
        
    # set_random_seed(args.seed_value)
    logging.basicConfig(filename = "algorithms/" + args.algorithm + "/results/logs/" + args.exp_name + "_" + bash_args.exp_idx + '.log', filemode = 'w', level = logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = algorithms_map[args.algorithm](args, device, bash_args.exp_idx)
    trainer.train()
    trainer.test()
    # trainer.save_plot()
    print("Finished!")
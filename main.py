from options.base_options import BaseOptions, reset_weight
from trainer import trainer
import torch
import os
import numpy as np
import random


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


seeds = [100]

layers_GCN = [15]

def main(args):
    if args.type_model in ['GCN', 'Cheby']:
        layers = layers_GCN

    acc_test_layers = []
    outs_layers = []
    for layer in layers:
        args.num_layers = layer
        if args.type_norm == 'group':
            args = reset_weight(args)
        acc_test_seeds = []
        outs_seeds = []
        for seed in seeds:
            args.random_seed = seed
            set_seed(args)
            trnr = trainer(args)
            acc_test, outs = trnr.train_compute_MI()
            acc_test_seeds.append(acc_test)
            outs_seeds.append(outs)
        avg_acc_test = np.mean(acc_test_seeds)

        acc_test_layers.append(avg_acc_test)
        import pandas as pd
        outs_seeds = pd.DataFrame(outs_seeds)
        outs_seeds = np.array(outs_seeds)
        outs_layers.append(outs_seeds.mean(0))

    print(f'experiment results of {args.type_norm} applied in {args.type_model} on dataset {args.dataset}')
    print('number of layers: ', layers)
    print('test accuracies: ', acc_test_layers)
    print("Mean of inner_sim, intra_sim, r_class: ", outs_layers)


if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)

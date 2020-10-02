import torch
import os
from Attack.foolbox_lib import FBAttack
import pickle
import argparse

from Module import MODULELIST
from utils import common_dumpinfo, common_set_random_seed
from utils import EPSILONS, ATTACKLIST, EPSILON_NUM


def arg_praser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default=2, type=int)
    parser.add_argument('--module', default=1, type=int, choices=[0, 1])
    parser.add_argument('--L', default=0, type=int, choices=[0, 1])
    args = parser.parse_args()
    return args


def main():
    args = arg_praser()
    device = torch.device(args.device)
    attack_type = ATTACKLIST[args.L]
    ModuleClass = MODULELIST[args.module]

    train_batch = 256
    max_size = 1000
    if args.module == 0:
        test_batch = 2000
    else:
        test_batch = 1000

    module = ModuleClass(device, train_batch, test_batch, max_size)
    norm_loader = module.norm_loader
    if not os.path.isdir('./Adversary/'):
        os.mkdir('./Adversary/')

    if max_size is not None:
        file_name = './Adversary/' + module.__class__.__name__ + '_' + attack_type + '_ALL.adv'
    else:
        file_name = './Adversary/' + module.__class__.__name__ + '_' + attack_type + '_ALL.adv'
    print(file_name)
    epsilon = EPSILONS[module.__class__.__name__][attack_type]
    epsilon = [epsilon]

    common_dumpinfo(module, attack_type, epsilon)
    attack = FBAttack(module.model, epsilon, device)
    adversary_samples = attack.generate_adv(norm_loader, attack_type)
    with open(file_name, 'wb') as f:
        pickle.dump(adversary_samples, f)


if __name__ == '__main__':
    common_set_random_seed(10)
    main()

import os
import torch
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

from Module import MODULELIST
from utils import EPSILONS, ATTACKID, EPSILON_NUM, ATTACKLIST
from utils import MyDataSet, common_set_random_seed, common_arg_praser
from Detector import Adv2Detector, VinallaDetector
from Attack.foolbox_lib import FBAttack
from utils import common_dumpinfo, common_construct_loader_list

from experiment import construct_loader, construct_res, cal_res


def get_adv_stastic(module, device, attack_key, epsions, dist_order, dataloader):
    adv_detector = Adv2Detector(module.model, device, attack_key, epsions, dist_order)
    score = adv_detector.run_detector(dataloader)
    return score


def extract_attackdist(attack_list, module, attack_l, epsilon, loader_list, device, dist_order):
    results = {}
    for defense in attack_list:
        score_mat = []
        common_dumpinfo(module, attack_l, epsilon)
        for data_loader in loader_list:
            score = get_adv_stastic(module, device, defense, epsilon, dist_order, data_loader)
            score_mat.append(score.reshape([-1, 1]))
        save_name = 'statistics/' + attack_l + '_' + module.__class__.__name__ + '_' + defense + '.csv'
        results[defense] = score_mat.copy()
        score_mat = np.concatenate(score_mat, axis=1)
        np.savetxt(save_name, score_mat, delimiter=',')
        y = np.ones_like(score_mat)
        y[:, 0] = 0
        score_list, y_list = construct_res(score_mat, y)
        score = cal_res(score_list, y_list)
        print(score)
    return results


def main():
    args = common_arg_praser()
    device = torch.device(args.device)
    attack_l = ATTACKLIST[args.L]
    if attack_l == 'L2':
        dist_order = 2
        defense_list = FBAttack.L2Attack
    else:
        dist_order = np.inf
        defense_list = FBAttack.LinfAttack
    ModuleClass = MODULELIST[args.module]

    train_batch = 256
    max_size = 1000
    test_batch = 1000
    module = ModuleClass(device, train_batch, test_batch, max_size)
    norm_loader = module.norm_loader
    max_size = len(norm_loader.dataset)
    epsilon = EPSILONS[module.__class__.__name__][attack_l] / (EPSILON_NUM)
    epsilon = [epsilon * i for i in range(1, EPSILON_NUM + 1)]
    file_name = './Adversary/' + module.__class__.__name__ + '_' + attack_l + '_ALL.adv'
    with open(file_name, 'rb') as f:
        adversary_samples = pickle.load(f)
    if attack_l == 'L2':
        attack_list = FBAttack.L2Attack
    else:
        attack_list = FBAttack.LinfAttack

    loader_list = common_construct_loader_list(norm_loader, adversary_samples, attack_list)
    if not os.path.isdir('statistics'):
        os.mkdir('statistics')
    results = extract_attackdist(attack_list, module, attack_l, epsilon, loader_list, device, dist_order)
    torch.save(results, 'feature/attackdist_' + module.__class__.__name__ + '_' + attack_l + '.fea')


if __name__ == '__main__':
    common_set_random_seed(10)
    main()

import os
import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from torch.utils.data import DataLoader
import argparse

from Module import MODULELIST
from utils import ATTACKLIST, EPSILONS, EPSILON_NUM
from utils import common_arg_praser
from utils import common_dumpinfo, common_construct_loader_list
from Detector.LID.LIDDetector import LIDDetector
from Detector.KD.KDDetector import KDDetector
from Detector.MC.MCDropout import MCDetecor
from Detector.Vinalla.VinallaDetector import VinallaDetector
from Detector.Mahalanobis.Mahalanobis import MahalanobisDetector
from Attack.foolbox_lib import FBAttack

BANDWIDTHS = {'MNIST_Module': 3.7926, 'CIFAR10_Module': 0.26, 'SVHN_Module': 1.0}


def test_auc(x_adv, x_norm):
    x = np.concatenate([x_adv, x_norm], axis=0)
    x = x.reshape([len(x), -1])
    y = np.zeros([len(x)])
    y[:len(x_adv)] = 1
    y = y.reshape([-1, 1])
    m = LogisticRegression()
    m.fit(x, y)
    pred = m.predict_proba(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred[:, 1], pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    return auc_score


def extract_lid(adversary_samples, attack_list, epsion_id, module, device, norm_loader, std, k):
    lid = LIDDetector(module.model, device, norm_loader, std, k=k)
    for attack_type in attack_list:
        attack_dataset = adversary_samples[attack_type]['adv'][epsion_id]
        x_norm, x_adv, x_noise = lid.feature_extractor(attack_dataset)
        x_norm = np.concatenate(x_norm, axis=0)
        x_adv = np.concatenate(x_adv, axis=0)
        x_noise = np.concatenate(x_noise, axis=0)
        score = test_auc(x_adv, x_norm)
        print('LID', attack_type, epsion_id, score)
        save_file = 'feature/' + attack_type + '_' + module.__class__.__name__ + '_' + str(epsion_id) + '_lid.fea'
        torch.save([x_norm, x_adv, x_noise], save_file)


# def extract_kd(adversary_samples, attack_list, epsion_id, module, device, test_loader, std, bandwidth, batch):
#     kd = KDDetector(module.model, device, module.train_loader, test_loader, std, module.class_num, bandwidth)
#     for attack_type in attack_list:
#         attack_dataset = adversary_samples[attack_type]['adv'][epsion_id]
#         adv_loader = DataLoader(attack_dataset, batch_size=batch)
#         x_adv, x_norm, x_noise = kd.feature_extractor(adv_loader)
#         score = test_auc(x_adv, x_norm)
#         print('KD', attack_type, epsion_id, score)
#         save_file = 'feature/' + attack_type + '_' + module.__class__.__name__ + '_' + str(epsion_id) + '_kd.fea'
#         torch.save([x_norm, x_adv, x_noise], save_file)


def extract_kd(data_loader_list, model, train_loader, device, std, class_num, bandwidth):
    results = []
    kd = KDDetector(model, device, train_loader, std, class_num, bandwidth)
    for data_loader in data_loader_list:
        kd_score = kd.feature_extractor(data_loader)
        results.append(kd_score)
    return results


def extract_mc(data_loader_list, model, device, iter_time):
    results = []
    mc = MCDetecor(model, device, iter_time)
    for data_loader in data_loader_list:
        mc_score = mc.feature_extractor(data_loader)
        results.append(mc_score)
    return results


def extract_vinalla(data_loader_list, model, device, ):
    results = []
    vinalla = VinallaDetector(model, device)
    for data_loader in data_loader_list:
        score = vinalla.feature_extractor(data_loader)
        results.append(score)
    return results


def extract_mahalan(loader_list, model, train_loader, device, class_num, attack_list):
    ma = MahalanobisDetector(model, device, train_loader, loader_list, class_num, attack_list)
    results = {}
    for i, defense_name in enumerate(attack_list):
        score_mat = ma.feature_extractor(i)
        results[defense_name] = score_mat
    return results


def main():
    args = common_arg_praser()
    device = torch.device(args.device)
    attack_l = ATTACKLIST[args.L]
    ModuleClass = MODULELIST[args.module]

    train_batch = 1000
    max_size = 1000
    test_batch = 1000
    module = ModuleClass(device, train_batch, test_batch, max_size)
    norm_loader = module.norm_loader
    std = 0.01

    if attack_l == 'L2':
        attack_list = FBAttack.L2Attack
    else:
        attack_list = FBAttack.LinfAttack

    file_name = './Adversary/' + module.__class__.__name__ + '_' + attack_l + '_ALL.adv'
    with open(file_name, 'rb') as f:
        adversary_samples = pickle.load(f)

    data_loader_list = common_construct_loader_list(norm_loader, adversary_samples, attack_list)

    if not os.path.isdir('feature'):
        os.mkdir('feature')

    ma = extract_mahalan(
        data_loader_list, module.model, module.train_loader,
        device, module.class_num, attack_list=attack_list
    )
    torch.save(ma, 'feature/ma_' + module.__class__.__name__ + '_' + attack_l + '.fea')

    vinalla = extract_vinalla(data_loader_list, module.model, module.device)
    torch.save(vinalla, 'feature/vinalla_' + module.__class__.__name__ + '_' + attack_l + '.fea')

    mc = extract_mc(data_loader_list, module.model, module.device, iter_time=500)
    torch.save(mc, 'feature/mc_' + module.__class__.__name__ + '_' + attack_l + '.fea')

    kd = extract_kd(
        data_loader_list, module.model, module.train_loader,
        device, std, module.class_num, BANDWIDTHS[module.__class__.__name__]
    )
    torch.save(kd, 'feature/kd_' + module.__class__.__name__ + '_' + attack_l + '.fea')


if __name__ == '__main__':
    main()

import torch
import os
from Attack.foolbox_lib import FBAttack
from Module.Cifar10 import CIFAR10_Module
from Module.Mnist import MNIST_Module
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import auc
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import EPSILONS, ATTACKID, MyDataSet, common_set_random_seed
from Detector import Adv2Detector, VinallaDetector
from Detector import KDDetector, LIDDetector
from Attack.foolbox_lib import FBAttack
from Module import MODULELIST
from utils import ATTACKLIST
from utils import common_get_auc


def construct_loader(adversary_samples, norm_loader, batch, epsion_id, max_size, attack_type):
    x, y = [], []
    for data in norm_loader:
        if torch.is_tensor(data):
            images = data
        else:
            images = data[0]
        x.append(images)
        y.append(torch.zeros([len(images), 1]))

    if attack_type is None:
        for ky in adversary_samples:
            label_id = ATTACKID[ky]
            new_x = adversary_samples[ky]['adv'][epsion_id]
            x.append(new_x[:max_size])
            y.append(torch.ones([len(new_x[:max_size]), 1]) * label_id)
    else:
        label_id = ATTACKID[attack_type]
        new_x = adversary_samples[attack_type]['adv'][epsion_id]
        x.append(new_x[:max_size])
        y.append(torch.ones([len(new_x[:max_size]), 1]) * label_id)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    dataset = MyDataSet(x, y)
    dataloader = DataLoader(dataset, batch_size=batch)
    return dataloader, y.detach().cpu().numpy()


def construct_res(score, y):
    adv_num = score.shape[1]
    score_list, y_list = [], []
    for i in range(1, adv_num):
        new_score = score[:, [0, i]]
        new_y = y[:, [0, i]]
        score_list.append(new_score)
        y_list.append(new_y)
    return score_list, y_list


def cal_res(score_list, y_list):
    result = []
    for i, score in enumerate(score_list):
        y = y_list[i].reshape([-1])
        fpr, tpr, thresholds = metrics.roc_curve(y, score.reshape([-1]), pos_label=1)
        val = metrics.auc(fpr, tpr)
        result.append(max(val, 1 - val))
    return result


def cross_auc(norm_train, adv_train, norm_test, adv_test):
    def construct_data(norm, adv):
        x = np.concatenate([norm, adv], axis=0)
        x = x.reshape([len(x), -1])
        y = np.zeros([len(x)])
        y[:len(adv)] = 1
        return x, y
    x_train, y_train = construct_data(norm_train, adv_train)
    x_test, y_test = construct_data(norm_test, adv_test)
    m = LogisticRegression()
    m.fit(x_train, y_train)
    pred = m.predict_proba(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred[:, 1], pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    return max(auc_score, 1-auc_score)


def get_vinalla_res(module, device, dataloader, y, max_size, adv_list):
    vinalla_detector = VinallaDetector(module.model, device, )
    score = vinalla_detector.run_detector(dataloader)
    score = score.reshape([-1, max_size]).T
    y = y.reshape([-1, max_size]).T
    data_num, dimension = score.shape
    score_list, y_list = construct_res(score, y, dimension)
    result = cal_res(score_list, y_list)
    assert len(result) == len(adv_list)
    print('vinalla')
    for i in range(len(result)):
        print(list(adv_list.keys())[i], result[i])
    mix_res = cal_res([score.reshape([-1])], [y.reshape([-1])])
    print('mix', mix_res)
    print('---------------------------')


def get_kd_res(module, attack_list, epsion_id=0):
    norm, adv, noise = [], [], []
    for attack_type in attack_list:
        file_name = 'feature/' + attack_type + '_' + module.__class__.__name__ + '_' + str(epsion_id) + '_kd.fea'
        results = torch.load(file_name)
        x_norm, x_adv, x_noise = results
        norm.append(x_norm)
        adv.append(x_adv)
        noise.append(x_noise)
    for i, defense_type in enumerate(attack_list):
        print('-------------', defense_type, '-------------')
        for j, attack_type in enumerate(attack_list):
            score = cross_auc(norm[i], adv[i], norm[j], adv[j])
            print(attack_type, score)
        mix_norm = np.concatenate(norm, axis=0)
        mix_adv = np.concatenate(adv, axis=0)
        score = cross_auc(norm[i], adv[i], mix_norm, mix_adv)
        print('mixed', score)


def get_lid_res(module, attack_list, epsion_id=0):
    norm, adv, noise = [], [], []
    for attack_type in attack_list:
        file_name = 'feature/' + attack_type + '_' + module.__class__.__name__ + '_' + str(epsion_id) + '_lid.fea'
        results = torch.load(file_name)
        x_norm, x_adv, x_noise = results
        norm.append(x_norm)
        adv.append(x_adv)
        noise.append(x_noise)
    for i, defense_type in enumerate(attack_list):
        print('-------------', defense_type, '-------------')
        for j, attack_type in enumerate(attack_list):
            score = cross_auc(norm[i], adv[i], norm[j], adv[j])
            print(attack_type, score)
        mix_norm = np.concatenate(norm, axis=0)
        mix_adv = np.concatenate(adv, axis=0)
        score = cross_auc(norm[i], adv[i], mix_norm, mix_adv)
        print('mixed', score)


def get_adv_res(module, attack_l, attack_list):
    score_database = []
    for defense in attack_list:
        file_name = 'statistics/' + attack_l + '_' + module.__class__.__name__ + '_' + defense + '.csv'
        score = np.loadtxt(file_name, delimiter=',')
        score_database.append(score)

    for i, defense_type in enumerate(attack_list):
        print('-------------', defense_type, '-------------')
        y = np.ones_like(score_database[i])
        data_num, adv_num = score_database[i].shape
        y[:, 0] = 0
        score_list, y_list = construct_res(score_database[i], y)
        result = cal_res(score_list, y_list)
        for j, attack_name in enumerate(attack_list):
            print(attack_name, result[j])
        mix_score = [score_database[i][:, iii] for iii in range(adv_num)]
        mix_score = np.concatenate(mix_score, axis=0)
        mix_y = np.ones_like(mix_score)
        mix_y[:data_num] = 0
        mix_res = cal_res([mix_score], [mix_y])
        print('mixed', mix_res[0])


def main():
    device = torch.device(1)
    train_batch = 256
    data_index = 1
    DATA = MODULELIST[data_index]

    if DATA.__name__ == 'MNIST_Module':
        test_batch = 5000
        max_size = None
    else:
        test_batch = 1000
        max_size = 2000
    for attack_l in ATTACKLIST:
        if attack_l == 'L2':
            attack_list = FBAttack.L2Attack
        else:
            attack_list = FBAttack.LinfAttack
        epsion_id = -1
        module = DATA(device, train_batch, test_batch, max_size)
        norm_loader = module.norm_loader
        max_size = len(norm_loader.dataset)
        file_name = './Adversary/' + module.__class__.__name__ + '_' + attack_l + '_ALL.adv'
        with open(file_name, 'rb') as f:
            adversary_samples = pickle.load(f)

        print('----------------VINALLA----------------------')
        get_vinalla_res(module, device, dataloader, y, max_size, attack_list)
        print('----------------KD----------------------------')
        get_kd_res(module, attack_list)
        print('----------------LID----------------------------')
        get_lid_res(module, attack_list)
        print('----------------AttackDist----------------------------')
        get_adv_res(module, attack_l, attack_list)


if __name__ == '__main__':
    common_set_random_seed(10)
    main()

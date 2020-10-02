import torch
import numpy as np
import os

from Module import MODULELIST
from utils import ATTACKLIST
from Metric import EvaluationMetric
from Attack.foolbox_lib import FBAttack


def normalize_data(attack_list, tmp):
    res = {}
    for k in attack_list:
        res[k] = tmp
    return res


def main():
    for attack_id in range(2):
        for data_id in range(2):
            DATA = MODULELIST[data_id]
            attack_l = ATTACKLIST[attack_id]
            metric_num = 5
            tool_num = 5
            if attack_l == 'L2':
                attack_list = FBAttack.L2Attack
            else:
                attack_list = FBAttack.LinfAttack
            results = np.zeros([tool_num * (len(attack_list) + 1), metric_num * len(attack_list)])

            ma = torch.load('feature/ma_' + DATA.__name__ + '_' + attack_l + '.fea')
            tool = torch.load('feature/attackdist_' + DATA.__name__ + '_' + attack_l + '.fea')

            for k in tool:
                tool[k] = [1 - v for v in tool[k]]

            tmp = torch.load('feature/vinalla_' + DATA.__name__ + '_' + attack_l + '.fea')
            vinalla = normalize_data(attack_list, tmp)
            tmp = torch.load('feature/mc_' + DATA.__name__ + '_' + attack_l + '.fea')
            mc = normalize_data(attack_list, tmp)
            tmp = torch.load('feature/kd_' + DATA.__name__ + '_' + attack_l + '.fea')
            kd = normalize_data(attack_list, tmp)

            tool_score = [vinalla, mc, kd, ma, tool]

            for j, defense in enumerate(attack_list):
                for m in range(tool_num):
                    score_mat = tool_score[m][defense]
                    score_mat = [s.reshape([-1, 1]) for s in score_mat]
                    score_mat = np.concatenate(score_mat, axis=1)
                    metric = EvaluationMetric(score_mat, attack_list)
                    for n in range(metric_num):
                        for i in range(len(attack_list) + 1):
                            results[int(i * tool_num + m)][int(n * len(attack_list) + j)] = \
                                metric.metric[n][i]

            results = np.around(results, decimals=3)
            print(results)
            if not os.path.isdir('table'):
                os.mkdir('table')
            save_file = 'table/' + DATA.__name__ + '_' + attack_l + '.csv'
            np.savetxt(save_file, results, delimiter=',')


if __name__ == '__main__':
    main()
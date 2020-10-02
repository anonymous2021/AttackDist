import numpy as np
import matplotlib.pyplot as plt

from utils import ATTACKLIST
from Attack.foolbox_lib import FBAttack
from Module import MODULELIST
from experiment import cal_res


def load_data(file_name):
    try:
        results = np.loadtxt(file_name, delimiter=',')
    except:
        with open(file_name, 'r') as f:
            d = f.readlines()
            d = [ddd.split() for ddd in d]
            for i, ddd in enumerate(d):
                for j, v in enumerate(ddd):
                    d[i][j] = float(d[i][j])
            results = np.array(d)
            print()
    return results


def main():
    for m in MODULELIST:
        for attack_l in ATTACKLIST:
            if attack_l == 'L2':
                defense_list = FBAttack.L2Attack
            else:
                defense_list = FBAttack.LinfAttack
            for defense in defense_list:
                file_name = attack_l + '_' + m.__name__ + '_' + defense + '.csv'
                print(file_name)
                results = load_data(file_name)
                # results = results.T
                # np.savetxt(file_name, results, delimiter=',')
                data_num, dimension = results.shape
                for j in range(1, dimension):
                    # plt.plot(results[:, 0], 'r*')
                    # plt.plot(results[:, j], 'b*')
                    # plt.show()
                    score = np.concatenate([results[:, 0], results[:, j]])
                    y_list = np.zeros_like(score)
                    y_list[data_num:] = 1
                    auc = cal_res([score], [y_list])
                    print(defense, list(defense_list.keys())[j - 1], auc)


if __name__ == '__main__':
    main()
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import  matplotlib.pyplot as plt


class EvaluationMetric:
    name_list = [
        'auroc',
         'aupr',
        'tnr_0.99',
        #'tnr_0.95',
        'tnr_0.90',
        'acc'
    ]
    def __init__(self, score_mat, attack_list):
        '''
        :param score_mat: the distance matrix,
        N x D, N is the number of data,
        D is the number of attacks
        :param truth_mat: the ground truth
        '''
        truth_mat = np.ones_like(score_mat)
        truth_mat[:, 0] = 0

        assert score_mat.shape[1] == len(attack_list) + 1

        self.score_mat = score_mat
        self.truth_mat = truth_mat
        self.attack_list = list(attack_list.keys())
        self.attack_list.append('MixAttack')
        self.data_num, self.adv_num = self.score_mat.shape
        self.score_list, self.y_list = self.construct_score()
        self.metric = self.evaluate()

    def evaluate(self):
        my_metric = [
            self.tnr_rate(tpr_tresh=0.99),
            #self.tnr_rate(tpr_tresh=0.95),
            self.tnr_rate(tpr_tresh=0.90),
            self.auc_score(),
            self.pr_score(),
            self.acc_score(),
        ]
        return my_metric

    def dump_results(self):
        for key in self.metric:
            print('-------------', key,'-------------')
            for i, attack in enumerate(self.attack_list):
                res = self.metric[key][i]
                res = np.around(np.array(res), decimals=4)
                print(attack, res)

    def construct_score(self):
        score_list, y_list = [], []
        for i in range(1, self.adv_num):
            new_score = self.score_mat[:, [0, i]]
            new_y = self.truth_mat[:, [0, i]]
            score_list.append(new_score)
            y_list.append(new_y)
        mix_score = [self.score_mat[:, i] for i in range(self.adv_num)]
        mix_score = np.concatenate(mix_score, axis=0)
        mix_y = np.ones_like(mix_score)
        mix_y[:self.data_num] = 0
        score_list.append(mix_score)
        y_list.append(mix_y)
        return score_list, y_list

    def auc_score(self):
        result = []
        for i, score in enumerate(self.score_list):
            score = score.reshape([-1])
            y = self.y_list[i].reshape([-1])
            val = metrics.roc_auc_score(y, score)
            result.append(max(val, 1 - val))
        return result

    def pr_score(self):
        result = []
        for i, score in enumerate(self.score_list):
            score = score.reshape([-1])
            y = self.y_list[i].reshape([-1])
            val = metrics.average_precision_score(y, score, pos_label=1)
            result.append(val)
        return result

    def tnr_rate(self, tpr_tresh=0.95):
        def cal_tnr(pred, threshs, truth):
            #TNR = TN / (FP+TN) TPR = TP/(TP + FN)
            tnr, tpr = [], []
            for t in threshs:
                # a = confusion_matrix(truth, (pred > t))
                # tn = a[1, 1]
                # fp = a[0, 1]
                tn = np.sum(((pred > t) == 0) * (truth == 0))
                fp = np.sum(((pred > t) == 1) * (truth == 0))
                fn = np.sum(((pred > t) == 0) * (truth == 1))
                tp = np.sum(((pred > t) == 1) * (truth == 1))
                tnr.append(tn / (fp + tn + 1e-8))
                tpr.append(tp / (tp + fn + 1e-8))
            return tnr, tpr

        def find_first(tpr_array):
            for i in range(len(tpr_array)):
                if tpr_array[i] > tpr_tresh:
                    return i
            return len(tpr_array) - 1

        result = []
        for i, score in enumerate(self.score_list):
            score = score.reshape([-1])
            y = self.y_list[i].reshape([-1])
            _, _, thresholds = metrics.roc_curve(y, score, pos_label=1)
            tnr, tpr = cal_tnr(score, thresholds, y)
            index = find_first(tpr)
            result.append(tnr[index])
        return result

    def acc_score(self):
        result = []
        for i, score in enumerate(self.score_list):
            max_acc = 0
            score = score.reshape([-1])
            y = self.y_list[i].reshape([-1])
            thresh_holds = np.sort(score)
            for t in thresh_holds:
                acc = ((score > t) == y)
                acc = np.sum(acc) / len(score)
                max_acc = max(acc, max_acc)
            result.append(max_acc)
        return result


if __name__ == '__main__':
    score = []
import torch
import numpy as np
from tqdm import tqdm

from Detector.Base import BaseDetector
from Attack.foolbox_lib import FBAttack
from scipy.spatial.distance import pdist, cdist, squareform
from utils import common_ten2numpy

import torch
from sklearn.linear_model import LogisticRegression


class MahalanobisDetector(BaseDetector):
    def __init__(self, model, device, train_loader, loader_list, class_num, attack_list):
        super(MahalanobisDetector, self).__init__(model, device)
        self.hidden_num = model.layer_num
        self.train_loader = train_loader
        self.class_num = class_num
        self.loader_list = loader_list  # test_loader  + list of adv loader
        self.attack_name = attack_list

        self.u_list, self.std_value = self.preprocess()
        assert len(loader_list) == len(attack_list) + 1

        self.train_feature = self.extract_metric(self.train_loader)
        self.feature_list = [self.extract_metric(loader) for loader in loader_list]
        self.lr_list = self.train_logic_list()

    def train_logic_list(self):
        print('training logic regression')

        def train_logic(x_train, y_train):
            logic_regression = LogisticRegression()
            logic_regression.fit(x_train, y_train)
            return logic_regression

        lr_list = []
        neg_x = self.train_feature
        for i, pos_x in enumerate(self.feature_list):
            if i == 0:
                continue
            x = np.concatenate([neg_x, pos_x], axis=0)
            y = np.ones([len(x)])
            y[:len(neg_x)] = 0

            lr = train_logic(x, y)
            lr_list.append(lr)
        return lr_list

    def preprocess(self):
        fx, y = self.get_penultimate()
        u_list, std_list = [], []
        std_value = []
        for target in range(self.class_num):
            fx_tar = [f[torch.where(y == target)[0]] for f in fx]
            mean_val = [torch.mean(f, dim=0) for f in fx_tar]
            norm_vec = [fx_tar[i] - mean_val[i] for i in range(self.hidden_num)]
            std_val = [vec.T.mm(vec) for vec in norm_vec]
            u_list.append(mean_val)
            std_list.append(std_val)

        for i in range(self.hidden_num):
            new_std = sum([f[i] for f in std_list]) / len(y)
            new_std = torch.inverse(new_std)
            std_value.append(new_std)
        return u_list, std_value

    def get_penultimate(self):
        res, y_list = [[] for _ in range(self.hidden_num)], []
        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            self.model.to(self.device)
            fx = self.model.get_feature(x)
            for lay_id in range(self.hidden_num):
                res[lay_id].append(fx[lay_id])
            y_list.append(y)
        res = [torch.cat(f, dim=0) for f in res]
        y_list = torch.cat(y_list, dim=0).detach().cpu()
        return res, y_list

    def extract_metric(self, data_loader):
        fx = [[] for _ in range(self.hidden_num)]
        for i, data in enumerate(data_loader):
            if torch.is_tensor(data):
                x = data
            else:
                x = data[0]
            x = x.to(self.device)
            self.model.to(self.device)
            new_fx = self.model.get_feature(x)
            for lay_id in range(self.hidden_num):
                fx[lay_id].append(new_fx[lay_id])
        fx = [torch.cat(f, dim=0) for f in fx]
        score = [[] for _ in range(self.hidden_num)]
        for target in range(self.class_num):
            tmp = [(fx[l] - self.u_list[target][l]).mm(self.std_value[l]) for l in range(self.hidden_num)]
            tmp = [tmp[l].mm((fx[l] - self.u_list[target][l]).T) for l in range(self.hidden_num)]
            tmp = [tmp[l].diagonal().reshape([-1, 1]) for l in range(self.hidden_num)]
            for l in range(self.hidden_num):
                score[l].append(-tmp[l])
        score = [torch.cat(s, dim=1) for s in score]
        score = [common_ten2numpy(torch.max(s, dim=1)[0]) for s in score]
        score = np.concatenate(score, axis=0).T.reshape([-1, self.hidden_num])
        return score

    def feature_extractor(self, defense_id):
        score_mat = []
        for feature in self.feature_list:
            score = self.lr_list[defense_id].predict_proba(feature)[:, 1]
            score_mat.append(score)
        return score_mat

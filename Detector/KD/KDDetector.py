import torch
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
import multiprocessing as mp
from torch.utils.data import DataLoader

from Detector.Base import BaseDetector
from utils import common_predict

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}


class KDDetector(BaseDetector):
    @staticmethod
    def score_point(tup):
        x, kde = tup
        return kde.score_samples(np.reshape(x, (1, -1)))[0]

    @staticmethod
    def score_samples(kdes, samples, preds, n_jobs=10):
        if n_jobs is not None:
            p = mp.Pool(n_jobs)
        else:
            p = mp.Pool()
        results = np.asarray(
            p.map(
                KDDetector.score_point,
                [(x, kdes[int(i)]) for x, i in zip(samples, preds)]
            )
        )
        p.close()
        p.join()
        return results

    def get_deep_representations(self, X, batch_size=256):
        # last hidden layer is always at index -4
        output = self.model.get_feature(X)
        return output

    @staticmethod
    def score_point(tup):
        """
        TODO
        :param tup:
        :return:
        """
        x, kde = tup

        return kde.score_samples(np.reshape(x, (1, -1)))[0]

    @staticmethod
    def score_samples(kdes, samples, preds, n_jobs=None):
        """
        TODO
        :param kdes:
        :param samples:
        :param preds:
        :param n_jobs:
        :return:
        """
        if n_jobs is not None:
            p = mp.Pool(n_jobs)
        else:
            p = mp.Pool()
        results = np.asarray(
            p.map(
                KDDetector.score_point,
                [(x, kdes[int(i)]) for x, i in zip(samples, preds)]
            )
        )
        p.close()
        p.join()
        return results

    def get_deep_representation(self, dataloader):
        res = []
        for data in dataloader:
            if torch.is_tensor(data):
                preds = self.model.get_deep_representation(data.to(self.device)).detach().cpu()
            else:
                preds = self.model.get_deep_representation(data[0].to(self.device)).detach().cpu()
            res.append(preds)
        return torch.cat(res, dim=0).detach().cpu().numpy()

    def feature_extractor(self, data_loader):
        x_feature = self.get_deep_representation(data_loader)
        print('Computing model predictions...')
        _, preds, _ = common_predict(data_loader, self.model, self.device)
        preds = preds.detach().cpu().numpy()
        densities = self.score_samples(self.kdes, x_feature, preds)
        return densities

    # def feature_extractor(self, adv_loader):
    #     x_test_normal_features = self.get_deep_representation(self.test_loader)
    #     x_test_noisy_features = self.get_deep_representation(self.noise_loader)
    #     x_test_adv_features = self.get_deep_representation(adv_loader)
    #
    #     # Get model predictions
    #     print('Computing model predictions...')
    #     _, preds_test_normal, _ = common_predict(self.test_loader, self.model, self.device)
    #     _, preds_test_noisy, _ = common_predict(self.noise_loader, self.model, self.device)
    #     _, preds_test_adv, _ = common_predict(adv_loader, self.model, self.device)
    #     preds_test_normal = preds_test_normal.detach().cpu().numpy()
    #     preds_test_noisy = preds_test_noisy.detach().cpu().numpy()
    #     preds_test_adv = preds_test_adv.detach().cpu().numpy()
    #     # Get density estimates
    #     print('computing densities...')
    #     densities_normal = self.score_samples(
    #         self.kdes,
    #         x_test_normal_features,
    #         preds_test_normal
    #     )
    #     densities_noisy = self.score_samples(
    #         self.kdes,
    #         x_test_noisy_features,
    #         preds_test_noisy
    #     )
    #     densities_adv = self.score_samples(
    #         self.kdes,
    #         x_test_adv_features,
    #         preds_test_adv
    #     )
    #     return densities_adv, densities_normal, densities_noisy

    def train_kde(self):
        x_train_features = self.get_deep_representation(self.train_loader)
        print('Training KDEs...')
        class_inds = {}
        for i in range(self.class_num):
            class_inds[i] = np.where(self.norm_y == i)[0]
        kdes = {}
        for i in range(self.class_num):
            kdes[i] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth) \
                .fit(x_train_features[class_inds[i]])
        return kdes

    def __init__(self, model, device, train_loader, std, class_num, bandwidth):
        super(KDDetector, self).__init__(model, device)
        self.train_loader = train_loader
        self.std = std

        self.class_num = class_num
        self.bandwidth = bandwidth
        self.norm_y = self.train_loader.dataset.y.detach().cpu().numpy()
        self.kdes = self.train_kde()
        self.noise_dataset = self.get_noise_dataset(train_loader, std)
        self.noise_loader = DataLoader(self.noise_dataset, batch_size=200)
        self.noise_feature = self.feature_extractor(self.noise_loader)

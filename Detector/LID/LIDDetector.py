import torch
import numpy as np
from tqdm import tqdm

from Detector.Base import BaseDetector
from Attack.foolbox_lib import FBAttack
from scipy.spatial.distance import pdist, cdist, squareform


class LIDDetector(BaseDetector):
    def calculate_score(self, tgt_mat, norm_mat):
        data = np.asarray(tgt_mat, dtype=np.float32)
        batch = np.asarray(norm_mat, dtype=np.float32)
        k = min(self.k, len(data))
        f = lambda v: -k / np.sum(np.log(v / v[-1]))
        a = cdist(batch, data)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
        a = np.apply_along_axis(f, axis=1, arr=a)
        return a

    def feature_extractor(self, adv_dataset):
        x_norm, x_adv, x_noise = [], [], []
        for iii, norm_batch in tqdm(enumerate(self.norm_loader)):
            if not torch.is_tensor(norm_batch):
                norm_batch = norm_batch[0]
            batch_size = len(norm_batch)
            st = iii * self.norm_loader.batch_size
            ed = st + batch_size
            adv_batch = adv_dataset[st:ed]
            noise_batch = self.noise_dataset[st:ed]
            norm_x, adv_x, noise_x = \
                np.zeros([batch_size, self.layer_num]), np.zeros([batch_size, self.layer_num]), np.zeros([batch_size, self.layer_num])
            norm_activation = self.model.get_activation(norm_batch.to(self.device))
            adv_activation = self.model.get_activation(adv_batch.reshape(norm_batch.shape).to(self.device))
            noise_activation = self.model.get_activation(noise_batch.reshape(norm_batch.shape).to(self.device))
            for i in range(self.layer_num):
                norm_a = norm_activation[i].reshape([batch_size, -1])
                adv_a = adv_activation[i].reshape([batch_size, -1])
                noise_a = noise_activation[i].reshape([batch_size, -1])

                norm_x[:, i] = self.calculate_score(norm_a, norm_a)
                adv_x[:, i] = self.calculate_score(adv_a, norm_a)
                noise_x[:, i] = self.calculate_score(noise_a, norm_a)
            x_norm.append(norm_x)
            x_noise.append(noise_x)
            x_adv.append(adv_x)
        return x_norm, x_adv, x_noise

    def __init__(self, model, device, norm_loader, std, k):
        super(LIDDetector, self).__init__(model, device)
        self.norm_loader = norm_loader
        self.std = std
        self.noise_dataset = self.get_noise_dataset(norm_loader, std)

        self.layer_num = model.layer_num
        self.k = k
        self.data_size = len(self.norm_loader.dataset)



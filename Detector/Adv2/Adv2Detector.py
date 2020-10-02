import torch
from torch.utils.data import DataLoader
import numpy as np

from Detector.Base import BaseDetector
from Attack.foolbox_lib import FBAttack
from utils import common_dataset2loader
from utils import common_Dist
from utils import common_loader2x


class Adv2Detector(BaseDetector):
    def __init__(self, model, device, attack_key, epsilons, order):
        super(Adv2Detector, self).__init__(model, device)
        self.attack_key = attack_key
        self.epsilons = epsilons
        self.order = order
        self.fb = FBAttack(model, epsilons, device)

    def get_success(self, x, success):
        num, data_num = len(success), len(success[0])
        new_x = x[-1]
        for i in range(data_num):
            for j in range(num - 1):
                if success[j][i] == 1:
                    new_x[i] = x[j][i]
                    break
        return new_x

    def run_detector(self, dataloader):
        adv_1, success, _ = self.fb.run_attack(self.attack_key, dataloader)
        adv_1 = self.get_success(adv_1, success)
        #adv_loader = common_dataset2loader(
        #    adv_1,  self.model, dataloader.batch_size, self.device)
        #adv_2, success, _ = self.fb.run_attack(self.attack_key, adv_loader)
        #adv_2 = self.get_success(adv_2, success)
        x_0 = common_loader2x(dataloader)

        d_1 = common_Dist(x_0, adv_1, self.order).reshape([-1])
        #d_2 = common_Dist(adv_1, adv_2, self.order).reshape([-1])
        #ist = d_1.astype(np.float64) / (d_2.astype(np.float64) + 1e-12)
        return d_1

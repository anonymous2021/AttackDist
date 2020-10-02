import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import foolbox as fb
import datetime

from Attack.foolbox_lib import FBAttack
from Module import MNIST_Module, CIFAR10_Module
from Detector.MC.MCDropout import MCDetecor

'''
 Test function only
'''

if __name__ == '__main__':
    attack_type = 'L2'
    device = torch.device(2)
    train_batch = 256
    test_batch = 1000
    max_size = 1000

    module = CIFAR10_Module(device, train_batch, test_batch, max_size)
    norm_loader = module.norm_loader

    detector = MCDetecor(module.model, module.device, 2)
    res = detector.run_detector(norm_loader)
    print(res)
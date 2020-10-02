import torch

from Module import MNIST_Module
from utils import EPSILONS
from Attack.foolbox_lib import FBAttack

def main():
    device = torch.device(1)
    train_batch = 256
    max_size = 1000
    test_batch = 1000
    attack_type = 'L2'

    module = MNIST_Module(device, train_batch, test_batch, max_size)
    norm_loader = module.norm_loader
    epsilon = EPSILONS[module.__class__.__name__][attack_type] / 50
    epsilon = [epsilon*i for i in range(51)]

    attack = FBAttack(module.model, epsilon, device)
    adversary_samples = attack.generate_adv(norm_loader, attack_type)

from abc import ABCMeta, abstractmethod
import torch


class BaseDetector:
    __metaclass__ = ABCMeta

    @staticmethod
    def get_noise_dataset(norm_loader, std):
        norm_dataset = norm_loader.dataset.x
        noise_dataset = norm_dataset + torch.randn(size=norm_dataset.shape) * std
        noise_dataset = torch.clamp(noise_dataset, 0, 1)

        if len(noise_dataset.shape) == 3:
            noise_dataset = noise_dataset.unsqueeze(1)

        return noise_dataset

    def __init__(self, model, device):
        self.model = model
        self.device = device


    @abstractmethod
    def feature_extractor(self, data_loader):
        pass
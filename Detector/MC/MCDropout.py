import torch

from utils import common_predict
from Detector.Base import BaseDetector


class MCDetecor(BaseDetector):
    def __init__(self, model, device, iter_time):
        super(MCDetecor, self).__init__(model, device)
        self.iter_time = iter_time

    def get_origin_label(self, data_loader):
        self.model.eval()
        _, pred_list, _ = common_predict(data_loader, self.model, self.device)
        return pred_list

    def feature_extractor(self, data_loader):
        origin_pred = self.get_origin_label(data_loader)
        label_chg = torch.zeros([len(data_loader.dataset)])
        self.model.train()
        for _ in range(self.iter_time):
            _, pred_list, _ = common_predict(data_loader, self.model, self.device, is_eval=False)
            label_chg += (origin_pred == pred_list)
        self.model.eval()
        return (label_chg / self.iter_time).detach().cpu().numpy()
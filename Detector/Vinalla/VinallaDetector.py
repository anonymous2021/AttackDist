import torch

from Detector.Base import BaseDetector


class VinallaDetector(BaseDetector):
    def __init__(self, model, device):
        super(VinallaDetector, self).__init__(model, device)

    def feature_extractor(self, data_loader):
        result = []
        for data in data_loader:
            if torch.is_tensor(data):
                images = data
            else:
                images, _ = data
            images = images.to(self.device)
            preds = self.model(images)
            preds = torch.nn.functional.softmax(preds, dim=1)
            preds, _ = torch.max(preds, dim=1)
            result.append(preds.detach().cpu())
        return torch.cat(result, dim=0).detach().cpu().numpy()

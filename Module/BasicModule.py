from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader, Subset
from utils import common_predict, common_ten2numpy
import torch
import os
from torchvision import transforms

from utils import MyDataSet


class BasicModule:
    __metaclass__ = ABCMeta

    def __init__(self, device, train_batch, test_batch, max_size):
        self.device = device
        self.train_batch_size = train_batch
        self.test_batch_size = test_batch
        self.max_size = max_size
        self.model = self.get_model()
        self.class_num = 10
        self.train_loader = None
        self.test_loader = None
        if not os.path.isdir('../Result/'):
            os.mkdir('../Result/')
        if not os.path.isdir('../Result/' + self.__class__.__name__):
            os.mkdir('../Result/' + self.__class__.__name__)

    def get_model(self):
        model = self.load_model()
        model.to(self.device)
        model.eval()
        print('model name is ', model.__class__.__name__)
        return model

    @abstractmethod
    def load_model(self):
        return None

    def get_hiddenstate(self, dataloader, device):
        sub_num = self.model.sub_num
        hidden_res, label_res = [[] for _ in sub_num], []
        for x, y in dataloader:
            x = x.to(device)
            res = self.model.get_hidden(x)
            for i, r in enumerate(res):
                hidden_res[i].append(r)
            label_res.append(y)
        hidden_res = [torch.cat(tmp, dim=0) for tmp in hidden_res]
        return hidden_res, sub_num, torch.cat(label_res)

    def get_loader(self, train_db, test_db):
        train_loader = DataLoader(
            train_db, batch_size=self.train_batch_size, shuffle=False)
        test_loader = DataLoader(
            test_db, batch_size=self.test_batch_size, shuffle=False)

        def loader2dataset(data_loader, max_size):
            norm_x, norm_y = [], []
            for data in data_loader:
                x, y = data
                norm_x.append(x)
                norm_y.append(y)
            norm_x = torch.cat(norm_x)
            norm_y = torch.cat(norm_y)
            if max_size is not None:
                norm_x = norm_x[:max_size]
                norm_y = norm_y[:max_size]
            norm_db = MyDataSet(norm_x, norm_y)
            return norm_db

        train_db = loader2dataset(train_loader, None)
        test_db = loader2dataset(test_loader, None)
        norm_db = loader2dataset(test_loader, self.max_size)

        train_loader = DataLoader(
            train_db, batch_size=self.train_batch_size, shuffle=False)
        test_loader = DataLoader(
            test_db, batch_size=self.test_batch_size, shuffle=False)
        norm_loader = DataLoader(
            norm_db, batch_size=self.test_batch_size,  shuffle=False)

        return train_loader, test_loader, norm_loader

    def get_information(self):
        self.train_pred_pos, self.train_pred_y, self.train_y = \
            common_predict(self.train_loader, self.model, self.device)

        self.test_pred_pos, self.test_pred_y, self.test_y = \
            common_predict(self.test_loader, self.model, self.device)


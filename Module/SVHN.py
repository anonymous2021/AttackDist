from model.svhn.covnet import CovNet
from Module.BasicModule import BasicModule
from utils import common_cal_accuracy
import torch
import numpy as np
import scipy.io as sio

from utils import MyDataSet


class SVHN_Module(BasicModule):
    def __init__(self, device, train_batch, test_batch, max_size, dump_info=False):
        super(SVHN_Module, self).__init__(device, train_batch, test_batch, max_size)
        self.train_loader, self.test_loader, self.norm_loader = self.load_data()
        self.input_shape = (3, 32, 32)
        self.class_num = 10
        if dump_info:
            self.get_information()
            self.test_acc = common_cal_accuracy(self.test_pred_y, self.test_y)
            self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)

            print('construct the module', self.__class__.__name__,
                  'the accuracy is %0.3f, %0.3f' % (self.train_acc, self.test_acc))

    def load_model(self):
        model = CovNet()
        state = torch.load('model_weight/svhn.h5', map_location=self.device)
        model.load_state_dict(state['net'])
        model.eval()
        return model

    def load_data(self):
        train_db = sio.loadmat('newdata/svhn_train.pt')
        test_db = sio.loadmat('newdata/svhn_test.pt')
        X_train = torch.tensor(np.transpose(train_db['X'], axes=[3, 2, 0, 1]) / 255, dtype=torch.float)
        X_test = torch.tensor(np.transpose(test_db['X'], axes=[3, 2, 0, 1]) / 255, dtype=torch.float)
        y_train = torch.tensor(np.reshape(train_db['y'], (-1,)) - 1, dtype=torch.long)
        y_test = torch.tensor(np.reshape(test_db['y'], (-1,)) - 1, dtype=torch.long)

        train_db = MyDataSet(X_train, y_train)
        test_db = MyDataSet(X_test, y_test)
        return self.get_loader(train_db, test_db)

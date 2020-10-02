from model.mnist.lenet import LeNet5
from Module.BasicModule import BasicModule
from utils import common_cal_accuracy
import torch


class MNIST_Module(BasicModule):
    def __init__(self, device, train_batch, test_batch, max_size, dump_info=False):
        super(MNIST_Module, self).__init__(device, train_batch, test_batch, max_size)
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
        model = LeNet5()
        state = torch.load('model_weight/mnist.h5', map_location=self.device)
        model.load_state_dict(state['net'])
        print('model accuracy is', state['acc'])
        model.eval()
        return model

    def load_data(self):
        train_db = torch.load('newdata/mnist' + '_train.pt')
        test_db = torch.load('newdata/mnist' + '_test.pt')
        return self.get_loader(train_db, test_db)

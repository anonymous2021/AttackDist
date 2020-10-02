import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
import numpy as np
import random
import argparse

EPSILONS = {
    'SVHN_Module': {
        'L2': 1.0,
        'Linf': 4.0e-2
    },
    'CIFAR10_Module': {
        'L2': 5.0e-1,
        'Linf': 1.5e-2
    },
    'MNIST_Module': {
        'L2': 3.0,
        'Linf': 2.5e-1
    }
}
EPSILON_NUM = 40
ATTACKLIST = [
    'L2CarliniWagnerAttack',
    'L2DeepFoolAttack',
    'L2BrendelBethgeAttack',
    'LinfProjectedGradientDescentAttack',
    'LinfBasicIterativeAttack',
    'LinfFastGradientAttack',
]
ATTACKID = {}
for i, a in enumerate(ATTACKLIST):
    ATTACKID[a] = i + 1
ATTACKLIST = ['L2', 'Linf']


class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


def common_set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def common_predict_y(dataset, model, device, batch_size=32):  # Todo : modeify the batch_size to a large number
    model.to(device)
    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=None,
    )
    pred_pos, pred_y, y_list = [], [], []
    for i, (x, y_label) in enumerate(data_loader):
        output = model(x.to(device))
        pos, y = torch.max(output, dim=1)
        pred_pos.append(output.detach())
        pred_y.append(y.detach())
        y_list.append(y_label)

    return torch.cat(pred_pos, dim=0).to(device), \
           torch.cat(pred_y, dim=0).view([-1]).to(device), \
           torch.cat(y_list, dim=0).view([-1]).to(device)


def common_predict(data_loader, model, device, is_eval=True):
    pred_pos, pred_list, y_list = [], [], []
    model.to(device)
    if is_eval:
        model.eval()
    for i, data in enumerate(data_loader):
        if torch.is_tensor(data):
            x, y = data, data
        else:
            x, y = data
        torch.cuda.empty_cache()
        x = x.to(device)
        output = model(x)
        pos, pred_y = torch.max(output, dim=1)
        pred_list.append(pred_y.detach())
        pred_pos.append(output.detach())
        y_list.append(y.detach())
    return torch.cat(pred_pos, dim=0).cpu(), torch.cat(pred_list, dim=0).cpu(), torch.cat(y_list, dim=0).cpu()


def common_get_auc(y_test, y_score, name=None):
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    if name is not None:
        print(name, 'auc is ', roc_auc)
    return roc_auc


def common_plotROC(y_test, y_score, file_name=None):
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
    print(file_name, 'auc is ', roc_auc)
    return roc_auc


def common_get_accuracy(ground_truth, oracle_pred, threshhold=0.1):
    oracle_pred = (oracle_pred > threshhold)
    pos_acc = (np.sum((oracle_pred == 1) * (ground_truth == 1))) / (np.sum(oracle_pred == 1) + 1)
    neg_acc = (np.sum((oracle_pred == 0) * (ground_truth == 1))) / (np.sum(oracle_pred == 0) + 1)
    coverage = (np.sum((oracle_pred == 1) * (ground_truth == 1))) / (np.sum(ground_truth == 1) + 1)
    print(threshhold, pos_acc, neg_acc, coverage)


def common_get_xy(dataset, batch_size, device):
    x, y = [], []
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    for i, data in enumerate(data_loader):
        x.append(data[0])
        y.append(data[1])
    return torch.cat(x, dim=0).cpu(), torch.cat(y, dim=0).cpu()


def common_cal_accuracy(pred_y, y):
    tmp = (pred_y.view([-1]) == y.view([-1]))
    acc = torch.sum(tmp.float()) / len(y)
    return acc


def common_load_corroptions():
    dir_name = './data/cifar_10/CIFAR-10-C/'
    y = np.load(dir_name + 'labels.npy')
    y = torch.tensor(y, dtype=torch.long)
    for file_name in os.listdir(dir_name):
        if file_name != 'labels.npy':
            x = np.load(dir_name + file_name)
            yield x, y, file_name.split('.')[0]


def common_get_maxpos(pos: torch.Tensor):
    test_pred_pos, _ = torch.max(F.softmax(pos, dim=1), dim=1)
    return common_ten2numpy(test_pred_pos)


def common_ten2numpy(a: torch.Tensor):
    return a.detach().cpu().numpy()


def common_dataset2loader(dataset, model, batch, device):
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    pred_list = []
    for x in dataloader:
        x = x.to(device)
        pred_y = model(x)
        _, pred_y = torch.max(pred_y, dim=1)
        pred_list.append(pred_y)
    pred_list = torch.cat(pred_list, dim=0).cpu()
    dataset = MyDataSet(dataset, pred_list)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    return dataloader


def common_Dist(vec_1, vec_2, order):
    vec_1 = vec_1.reshape([len(vec_1), -1]).numpy()
    vec_2 = vec_2.reshape([len(vec_2), -1]).numpy()
    return np.linalg.norm(vec_1 - vec_2, order, axis=1)


def common_loader2x(dataloader):
    x = []
    for data in dataloader:
        if torch.is_tensor(data):
            image = data
        else:
            image = data[0]
        x.append(image)
    return torch.cat(x, dim=0)


def common_dumpinfo(module, attack_l, epsilon_value):
    print('--------------------------')
    print('dataset:', module.__class__.__name__)
    print('attack_l:', attack_l)
    print('epsilon:', epsilon_value)
    print('--------------------------')


def common_construct_loader_list(norm_loader, adversarys, attack_list):
    loader_list = [norm_loader]
    for key in attack_list:
        x = adversarys[key]['adv'][0]
        data_loader = DataLoader(x, batch_size=norm_loader.batch_size)
        assert len(norm_loader.dataset) == len(x)
        loader_list.append(data_loader)
    return loader_list


def common_arg_praser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default=2, type=int)
    parser.add_argument('--module', default=0, type=int, choices=[0, 1], help='0 is mnist, 1 is cifar')
    parser.add_argument('--L', default=0, type=int, choices=[0, 1], help='0 is L2 and 1 is Linf')
    parser.add_argument('--batch', default=500, type=int, help='batch size')
    args = parser.parse_args()
    return args
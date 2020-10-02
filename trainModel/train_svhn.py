import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
import scipy.io as sio

import os
import argparse

from utils import MyDataSet
from model.svhn.covnet import *
from trainModel.train_utils import loadTrainData, parse_args
from trainModel.train_utils import train_model, test_model


def generate_dataloader():
    train_db = sio.loadmat('../newdata/svhn_train.pt')
    test_db = sio.loadmat('../newdata/svhn_test.pt')
    X_train = torch.tensor(np.transpose(train_db['X'], axes=[3, 2, 0, 1]) / 255, dtype=torch.float)
    X_test = torch.tensor(np.transpose(test_db['X'], axes=[3, 2, 0, 1]) / 255, dtype=torch.float)
    y_train = torch.tensor(np.reshape(train_db['y'], (-1,)) - 1, dtype=torch.long)
    y_test = torch.tensor(np.reshape(test_db['y'], (-1,)) - 1, dtype=torch.long)

    train_db = MyDataSet(X_train, y_train)
    test_db = MyDataSet(X_test, y_test)
    train_loader = DataLoader(train_db, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=5000)
    return train_loader, test_loader


def main():
    args = parse_args()
    device = torch.device(args.device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    train_loader, test_loader = generate_dataloader()
    # Model
    print('==> Building model..')
    net = CovNet()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../model_weight'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('../model_weight/svhn.h5')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                           momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch, start_epoch+100):
        train_model(epoch, net, train_loader, optimizer, criterion, device)
        acc = test_model(net, test_loader, criterion, device)
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('../model_weight/'):
                os.mkdir('../model_weight/')
            torch.save(state, '../model_weight/svhn.h5')
            best_acc = acc


if __name__ == '__main__':
    main()

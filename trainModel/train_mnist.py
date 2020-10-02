import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model.mnist.lenet import *
from trainModel.train_utils import loadTrainData, parse_args
from trainModel.train_utils import train_model, test_model


def generate_dataloader():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader, test_loader = loadTrainData(
        'mnist', 64, 3000, transform_train, transform_test)
    return train_loader, test_loader


def main():
    args = parse_args()
    device = torch.device(args.device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..')
    train_loader, test_loader = generate_dataloader()

    print('==> Building model..')
    net = LeNet5()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../model_weight'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('../model_weight/mnist.h5')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch, start_epoch+200):
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
            torch.save(state, '../model_weight/mnist.h5')
            best_acc = acc


if __name__ == '__main__':
    main()

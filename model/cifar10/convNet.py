import torch


class LeNet_Cifar(torch.nn.Module):
    def __init__(self):
        super(LeNet_Cifar, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=2, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)

        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.4)

        self.fc1 = torch.nn.Linear(4608, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 10)

        self.layer_num = 2

    def forward(self, x):
        x = self.get_deep_representation(x)
        x = self.fc3(x)
        return x

    def get_deep_representation(self, x):
        data_num = len(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool_1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool_2(x)
        x = x.view(data_num, -1)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

    def get_activation(self, x):
        data_num = len(x)
        res = []
        x = self.conv1(x)
        res.append(x.detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        res.append(x.detach().cpu().numpy())
        x = self.max_pool_1(x)
        res.append(x.detach().cpu().numpy())
        x = self.conv2(x)
        res.append(x.detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        res.append(x.detach().cpu().numpy())
        x = self.max_pool_2(x)
        res.append(x.detach().cpu().numpy())

        x = x.view(data_num, -1)
        x = self.fc1(x)
        res.append(x.detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        res.append(x.detach().cpu().numpy())
        x = self.fc2(x)
        res.append(x.detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        res.append(x.detach().cpu().numpy())
        return res

    def get_feature(self, x):
        data_num = len(x)
        res = []

        x = self.conv1(x)
        #res.append(x.detach().cpu().reshape([data_num, -1]))
        x = self.relu(x)

        x = self.conv2(x)
        #res.append(x.detach().cpu().reshape([data_num, -1]))
        x = self.relu(x)
        x = self.max_pool_1(x)

        x = self.conv3(x)
        #res.append(x.detach().cpu().reshape([data_num, -1]))
        x = self.relu(x)

        x = self.conv4(x)
        #res.append(x.detach().cpu().reshape([data_num, -1]))
        x = self.relu(x)

        x = self.max_pool_2(x)
        x = x.view(data_num, -1)

        x = self.fc1(x)
        res.append(x.detach().cpu().reshape([data_num, -1]))
        x = self.relu(x)

        x = self.fc2(x)
        res.append(x.detach().cpu().reshape([data_num, -1]))
        return res


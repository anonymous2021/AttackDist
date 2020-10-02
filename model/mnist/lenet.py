import torch


class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

        self.dropout = torch.nn.Dropout(0.4)
        self.layer_num = 1

    def forward(self, x):
        x = self.get_deep_representation(x)
        x = self.fc3(x)
        return x

    def get_deep_representation(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.max_pool_1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.max_pool_2(x)
        x = self.dropout(x)

        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        return x

    def get_feature(self, x):
        data_num = len(x)
        res = []
        x = self.conv1(x)
        #res.append(x.detach().cpu().reshape([data_num, -1]))
        x = torch.nn.functional.relu(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        #res.append(x.detach().cpu().reshape([data_num, -1]))
        x = torch.nn.functional.relu(x)
        x = self.max_pool_2(x)

        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        #res.append(x.detach().cpu().reshape([data_num, -1]))
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        res.append(x.detach().cpu().reshape([data_num, -1]))
        x = torch.nn.functional.relu(x)
        return res


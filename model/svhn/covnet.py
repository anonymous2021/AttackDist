import torch


class CovNet(torch.nn.Module):
    def __init__(self):
        super(CovNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.batch_norm_1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0, bias=True)
        self.batch_norm_2 = torch.nn.BatchNorm2d(32)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)



        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(1152, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)
        self.layer_num = 10

    def forward(self, x):
        x = self.get_deep_representation(x)
        x = self.fc3(x)
        return x

    def get_deep_representation(self, x):
        data_num = len(x)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm_1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm_2(x)
        x = self.max_pool_2(x)
        x = self.dropout(x)

        x = x.view(data_num, -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        return x

    def get_activation(self, x):
        data_num = len(x)
        res = []
        x = self.conv1(x)
        res.append(x.detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        res.append(x.detach().cpu().numpy())
        x = self.batch_norm_1(x)
        x = self.max_pool_1(x)
        res.append(x.detach().cpu().numpy())
        x = self.conv2(x)
        res.append(x.detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        res.append(x.detach().cpu().numpy())
        x = self.batch_norm_2(x)
        x = self.max_pool_2(x)
        x = self.dropout(x)
        res.append(x.detach().cpu().numpy())

        x = x.view(data_num, -1)
        x = self.fc1(x)
        x = self.dropout(x)
        res.append(x.detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        res.append(x.detach().cpu().numpy())
        x = self.fc2(x)
        x = self.dropout(x)
        res.append(x.detach().cpu().numpy())
        x = torch.nn.functional.relu(x)
        res.append(x.detach().cpu().numpy())
        return res


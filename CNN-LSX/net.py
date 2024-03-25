import torch.nn.functional as f
import torch.nn as nn
from torch import Tensor
import torch

import global_vars

## Define the NN architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1)
        # linear layer (n_hidden -> hidden_2)
        # self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc2 = nn.Linear(1, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function

        # x = f.relu(self.fc1(x))
        x = self.fc1(x)

        # # add dropout layer
        # x = self.dropout(x)
        # # add hidden layer, with relu activation function
        x = f.sigmoid(self.fc2(x))
        # add dropout layer
        # x = self.dropout(x)
        # add output layer
        # x = self.fc2(x)
        return x


class Net(nn.Module):
    # Use the same architecture as https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self, accepts_additional_explanations: bool = False):
        super().__init__()
        self.accepts_additional_explanations = accepts_additional_explanations
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, global_vars.N_CLASS)

        self.enc = None # Dummy

        # I'm not sure why there's a warning here. It's still there when downloading their notebook,
        # so it might be a problem with the tutorial.
        # [here](https://stackoverflow.com/questions/48132786/why-is-this-warning-expected-type-int-matched-generic-type-t-got-dict)
        # it sounds like it's a PyCharm Issue.
        # ("My guess would be the analytics that give this warning are not sharp enough.")
        # I let PyCharm ignore it for now.

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.avg_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        self.enc = x
        x = self.fc1(x)
        x = f.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class MedNet(nn.Module):
    # Use the same architecture as https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self, accepts_additional_explanations: bool = False):
        super().__init__()
        self.accepts_additional_explanations = accepts_additional_explanations
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, global_vars.N_CLASS)

        self.enc = None # Dummy

        # I'm not sure why there's a warning here. It's still there when downloading their notebook,
        # so it might be a problem with the tutorial.
        # [here](https://stackoverflow.com/questions/48132786/why-is-this-warning-expected-type-int-matched-generic-type-t-got-dict)
        # it sounds like it's a PyCharm Issue.
        # ("My guess would be the analytics that give this warning are not sharp enough.")
        # I let PyCharm ignore it for now.

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.avg_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        self.enc = x
        x = self.fc1(x)
        x = f.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class Net2(nn.Module):
    # Use the same architecture as https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self, accepts_additional_explanations: bool = False):
        super().__init__()
        self.accepts_additional_explanations = accepts_additional_explanations
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.batchnorm = nn.BatchNorm2d(32)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # I'm not sure why there's a warning here. It's still there when downloading their notebook,
        # so it might be a problem with the tutorial.
        # [here](https://stackoverflow.com/questions/48132786/why-is-this-warning-expected-type-int-matched-generic-type-t-got-dict)
        # it sounds like it's a PyCharm Issue.
        # ("My guess would be the analytics that give this warning are not sharp enough.")
        # I let PyCharm ignore it for now.

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.last_conv = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.last_conv(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4*4*50)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 10)
        self.fc4 = nn.Linear(64, 10)
        # defining the 20% dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, 28*28)
        print(x.shape)

        x = self.dropout(f.relu(self.fc1(x)))
        x = self.dropout(f.relu(self.fc2(x)))
        x = self.dropout(f.relu(self.fc3(x)))
        # not using dropout on output layer
        x = f.log_softmax(self.fc3(x), dim=1)
        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class Net4(nn.Module):

    def __init__(self):
        super(Net4, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class CIFAR100_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 120)
        self.fc3 = nn.Linear(120, 100)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CIFAR10_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
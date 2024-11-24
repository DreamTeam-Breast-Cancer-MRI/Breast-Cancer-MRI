from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax


class CNN(Module):
    def __init__(self, num_channels=1, out_size=2):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(
            in_channels=num_channels, out_channels=20, kernel_size=(5, 5)
        )
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()

        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = Linear(in_features=781250, out_features=500)
        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=500, out_features=out_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 50 * 125 * 125)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)

        return x

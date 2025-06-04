import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights, alexnet, AlexNet_Weights, resnet50, \
    ResNet50_Weights


class CNNModel(torch.nn.Module):
    def __init__(self, output_num):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, 2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, output_num))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x

class LeafCNN1(torch.nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """

    def __init__(self, num_classes):
        super(LeafCNN1, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 2048)
        self.output = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class LeNet(LeafCNN1):
    """
    CNN model used in "(ICML 21)  Personalized Federated Learning using Hypernetworks":
    a LeNet-based (LeCun et al., 1998) network with two convolution and two fully connected layers.
    """

    def __init__(self, num_classes, n_kernels=32, in_channels=3, fc_factor=1, fc_factor2=1):
        super(LeNet, self).__init__(num_classes)
        in_channels = in_channels
        self.n_kernels = n_kernels
        self.fc_factor = fc_factor
        self.fc_factor2 = fc_factor2
        self.conv1 = torch.nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = torch.nn.Linear(2 * n_kernels * 5 * 5, 120 * self.fc_factor)
        self.fc2 = torch.nn.Linear(120 * self.fc_factor, 84 * self.fc_factor2)
        self.output = torch.nn.Linear(84 * self.fc_factor2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2 * self.n_kernels * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # 使用预训练权重初始化 ResNet50 模型
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # ResNet50 最后一层全连接层的输入通道数为 2048
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)



class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.model = vgg16(weights=VGG16_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # 修改最后一个全连接层

    def forward(self, x):
        return self.model(x)

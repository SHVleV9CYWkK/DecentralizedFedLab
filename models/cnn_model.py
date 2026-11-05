import math

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

def convert_bn_to_gn(module, groups=32):
    """递归把 BatchNorm 替换为 GroupNorm（WC 规范 §8-2：模型平均破坏 BN running stats）。"""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            channels = child.num_features
            setattr(module, name, nn.GroupNorm(math.gcd(groups, channels), channels))
        else:
            convert_bn_to_gn(child, groups)


class TinyViT(torch.nn.Module):
    """timm tiny_vit_11m_224（BN→GN 转换，从随机初始化训练）。

    TinyViT 的卷积嵌入阶段含 BatchNorm，去中心化模型平均会破坏其 running stats
    （WC 规范 §8-2），故统一替换为 GroupNorm。层级结构对输入尺寸自适应，
    Tiny-ImageNet 的 64×64 输入可直接前向，无需 Resize。
    """
    def __init__(self, num_classes, groups=32):
        super(TinyViT, self).__init__()
        import timm
        self.model = timm.create_model('tiny_vit_11m_224', pretrained=False,
                                       num_classes=num_classes)
        convert_bn_to_gn(self.model, groups)
        self._disable_attention_bias_cache()

    def _disable_attention_bias_cache(self):
        """禁用 timm TinyViT 在 eval 模式下的 attention bias 缓存。

        timm 的实现把带计算图的张量缓存进 attention_bias_cache，eval 模式下
        多次 backward（L̂ 探针、平稳性梯度评测）会触发
        "backward through the graph a second time"。改为每次前向重新索引。
        """
        import types

        def _fresh_biases(module, device):
            return module.attention_biases[:, module.attention_bias_idxs]

        for module in self.model.modules():
            if hasattr(module, 'attention_bias_cache'):
                module.get_attention_biases = types.MethodType(_fresh_biases, module)

    def forward(self, x):
        return self.model(x)


class ResNet18GN(torch.nn.Module):
    """ResNet18 + GroupNorm（替换全部 BatchNorm），卷积/FC 层用 ImageNet 预训练初始化。

    去中心化模型平均（每轮混合 / 加入热启动）会破坏 BN 的 running statistics
    （WC 规范 §8-2），联邦实验统一改用 GN。预训练权重只能部分迁移：BN 的
    running_mean/var 在 GN 中不存在（直接 strict 加载会崩），且 BN 的 affine
    参数语义（批统计尺度）与 GN（组尺度）不同——故只加载卷积与 FC 权重，
    归一化层保持新初始化（GN 迁移的标准做法）。
    """
    def __init__(self, num_classes, groups=32):
        super(ResNet18GN, self).__init__()
        def norm_layer(channels):
            return nn.GroupNorm(num_groups=math.gcd(groups, channels), num_channels=channels)
        self.model = resnet18(weights=None, norm_layer=norm_layer)
        pretrained = ResNet18_Weights.DEFAULT.get_state_dict(progress=False)
        own = self.model.state_dict()
        transfer = {k: v for k, v in pretrained.items()
                    if k in own and own[k].shape == v.shape and not self._is_norm_key(k)}
        self.model.load_state_dict(transfer, strict=False)
        self.model.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _is_norm_key(key):
        # 归一化层参数不迁移：bn1./ .bnN. / downsample.1 是 resnet 中全部 norm 位置
        parts = key.split('.')
        return any(p.startswith('bn') for p in parts) or \
            ('downsample' in parts and parts[parts.index('downsample') + 1] == '1')

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

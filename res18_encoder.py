import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, Normalize
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn.functional as F
from paddle.metric import Accuracy
from collections import OrderedDict
import paddle
import paddle.nn as nn
from paddle.vision.models import resnet18
from encoder import Encoder


class ResidualBlock(nn.Layer):
    """残差块（ResNet基础组件）"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2D(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CRNN(nn.Layer):
    def __init__(self, n_classes, input_shape=(3, 42, 130)):
        super().__init__()
        self.input_shape = input_shape

        # 修改后的ResNet18结构（适配输入尺寸）
        self.cnn = nn.Sequential(
            nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),

            # conv2_x: 1 blocks
            ResidualBlock(64, 64),
            # conv3_x: 2 blocks
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),

            # conv4_x: 2 blocks
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),

            # conv5_x: 2 blocks
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2D((1, 16))
        )

        
        self.encoder = Encoder(in_dim=512,
                        hidden_dim=256,
                        n_head=8,
                        num_layers=2)
        
        self.fc = nn.Linear(512, n_classes)  # 双向LSTM输出需2*hidden_size

    def forward(self, x):
        # CNN部分
        x = self.cnn(x)
        x = x.transpose([0, 3, 1, 2])  # 将宽度维度转为序列长度
        x = x.reshape([x.shape[0], x.shape[1], -1])  # [batch_size, seq_len, features]
        x = self.encoder(x)
        x = self.fc(x)
        x = x.transpose([1, 0, 2])  # [seq_len, batch_size, features]
        return x

    def get_seq_len(self):
        """获取序列长度（适配ResNet输出）"""
        dummy_input = paddle.zeros([1] + list(self.input_shape))
        x = self.cnn(dummy_input)
        return x.shape[-1]  # 直接返回宽度维度作为序列长度
    
    def count_parameters(self):
        """计算模型参数量"""
        total_params = 0
        for name, param in self.named_parameters():
            param_count = param.numel()
            total_params += param_count
        return total_params